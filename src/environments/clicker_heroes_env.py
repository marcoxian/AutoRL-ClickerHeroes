from typing import Dict, List, Tuple, Optional, Any
import time
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np




from src.utils.screen_capture import ScreenCapture
from src.utils.input_controller import InputController
from src.utils.reward_extractor import RewardExtractor
from src.utils.smart_vision import SmartVisionController
from src.utils.window_finder import find_game_window


class ClickerHeroesEnv(gym.Env):
    """
    Gymnasium Environment for Clicker Heroes using visual screen state, virtual clicks, and scroll actions.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        region: Optional[Dict[str, int]] = None,
        target_size: Tuple[int, int] = (84, 84),
        max_steps: int = 200,
        action_delay: float = 0.05,
        mock_mode: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        :param region: Screen capture region bounding box {'top', 'left', 'width', 'height'}.
        :param target_size: Observation image dimensions (width, height).
        :param max_steps: Maximum steps per episode.
        :param action_delay: Sleep time in seconds after executing a virtual click.
        :param mock_mode: If True, uses synthetic frames and simulated mouse control.
        :param render_mode: Gymnasium render mode ("rgb_array" or "human").
        """
        super().__init__()

        self.target_size = target_size
        self.max_steps = max_steps
        self.mock_mode = mock_mode
        self.render_mode = render_mode

        # Define Observation Space: (C, H, W) RGB uint8
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, target_size[1], target_size[0]),
            dtype=np.uint8,
        )

        # Define Action Space: 8 Discrete Actions
        # 0: Click, 1: Upgrade, 2: Hero Skills, 3: Global Skills, 4: Next Zone, 5: Scroll Down, 6: Scroll Up, 7: No-Op
        self.action_space = spaces.Discrete(8)


        self.target_size = target_size
        self.channel_first = True

        # Initialize sub-modules
        self.screen_capture = ScreenCapture(
            region=region,
            target_size=target_size,
            channel_first=True,
            mock_mode=mock_mode,
        )

        self.input_controller = InputController(
            region=self.screen_capture.region,
            action_delay=action_delay,
            mock_mode=mock_mode,
        )
        self.reward_extractor = RewardExtractor(
            mock_mode=mock_mode,
        )

        self.current_step = 0
        self.current_zone = 1
        self.max_zone_reached = 1
        self.retreated_from_boss = False
        self.can_advance_to_new_zone = False
        self.steps_in_current_zone = 0
        self.advance_locked = False
        self.upgrades_purchased_since_retreat = 0
        self.steps_fighting_current_boss = 0
        self.boss_fight_start_time = 0.0
        self.max_zone_steps = 300  # ~60 segundos en la misma zona
        self.last_observation: Optional[np.ndarray] = None
        self.last_raw_rgb: Optional[np.ndarray] = None
        self.smart_vision = SmartVisionController()
        self.zone_detection_history: List[int] = []
        self.last_action_name: str = "Inicio"
        self.last_step_reward: float = 0.0
        self.episode_total_reward: float = 0.0
        self.has_boss_timer_active: bool = False
        self.last_event_msg: Optional[str] = None
        self.last_bought_hero_skills: int = 0  # Conteo estable de habilidades COMPRADA al inicio del episodio
        self.new_zone_candidate: Optional[int] = None  # Zona candidata a confirmar como nuevo récord
        self.new_zone_first_seen: float = 0.0  # Timestamp cuando se vio por primera vez la zona candidata
        self.zone_advance_rewarded: bool = False  # Evita doble recompensa por el mismo avance
        self.needs_to_press_a: bool = False  # Flag estricto que indica si la bota se ha apagado y necesita reactivarse
        self.global_steps_stuck: int = 0  # Steps since max zone was reached


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment state for a new episode.
        En modo EN VIVO preserva la zona alcanzada en el juego para mantener coherencia.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.zone_detection_history.clear()
        self.last_action_name = "Reset"
        self.last_step_reward = 0.0
        self.episode_total_reward = 0.0
        self.has_boss_timer_active = False
        self.last_event_msg = "🔄 [RESET] Nuevo episodio iniciado"

        if self.mock_mode or (options and options.get("hard_reset", False)):
            self.current_zone = 1
            self.max_zone_reached = 1
            self.retreated_from_boss = False
            self.can_advance_to_new_zone = False
            self.advance_locked = False
            self.upgrades_purchased_since_retreat = 0
            self.steps_fighting_current_boss = 0
            self.boss_fight_start_time = 0.0
            self.steps_in_current_zone = 0

        self.last_observation = None
        self.last_raw_rgb = None
        self.last_bought_hero_skills = 0
        self.new_zone_candidate = None
        self.new_zone_first_seen = 0.0
        self.zone_advance_rewarded = False
        self.needs_to_press_a = False
        self.scroll_depth = 0
        self.reward_extractor.reset()



        # Capturar fotograma en resolución nativa para OpenCV y redimensionar para la CNN
        raw_rgb = self.screen_capture.capture_raw_rgb()
        self.last_raw_rgb = raw_rgb

        # Sincronizar visualmente la zona real desde la pantalla con OpenCV
        if not self.mock_mode:
            detected_zone = self.smart_vision.detect_zone_number_from_screen(raw_rgb)
            if detected_zone is not None and 1 <= detected_zone <= 1000:
                # Filtrar glitches en el primer frame del reset:
                # Aceptar si acabamos de iniciar el script (todo en 1), si reseteamos (<= 5), o salto <= 50
                if (self.current_zone == 1 and self.max_zone_reached == 1) or detected_zone <= 5 or abs(detected_zone - self.current_zone) <= 50:
                    self.current_zone = detected_zone
                    self.max_zone_reached = max(self.max_zone_reached, self.current_zone)

        obs = cv2.resize(raw_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        if self.channel_first:
            obs = np.transpose(obs, (2, 0, 1))
        self.last_observation = obs

        info = {
            "is_mock": self.mock_mode,
            "region": self.screen_capture.region,
            "current_zone": self.current_zone,
            "max_zone_reached": self.max_zone_reached,
            "advance_locked": self.advance_locked,
        }

        # Al iniciar un episodio, nos aseguramos de pulsar 'a' para activar la progresión automática
        if not self.mock_mode:
            try:
                import pydirectinput
                pydirectinput.press('a')
            except Exception:
                pass

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes action with Native High-Res OpenCV Vision, Manual Progression, and Single Boss Retreat Lock.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid for action space {self.action_space}.")
        blocked_action_penalty = 0.0
        advance_bonus_reward = 0.0
        skill_bonus_reward = 0.0
        actual_action_executed = action
        
        self.global_steps_stuck += 1

        # 0. Detección y cierre automático de Popups (Ej: Reliquias o Anuncios)
        if not self.mock_mode and self.last_raw_rgb is not None:
            close_btn = self.smart_vision.detect_popup_close_button(self.last_raw_rgb)
            if close_btn is not None:
                x_rel, y_rel = close_btn
                self.input_controller.click(x_rel, y_rel, clicks=1)
                self.last_event_msg = "🚨 [POPUP DETECTADO] Cerrando ventana emergente automáticamente"
                time.sleep(0.5)
                # Recapturar la pantalla limpia antes de continuar con la acción de la IA
                self.last_raw_rgb = self.screen_capture.capture_raw_rgb()

        # 0.05 Detección de pestaña incorrecta (Mercenarios, Logros, etc.)
        # Si la IA ha hecho clic en otra pestaña que no es la de Héroes, volvemos automáticamente.
        if not self.mock_mode and self.last_raw_rgb is not None:
            if self.smart_vision.is_on_wrong_tab(self.last_raw_rgb):
                # Clic en la primera pestaña (icono de espada = Héroes) en la esquina superior izquierda
                self.input_controller.click(0.025, 0.055, clicks=1)
                self.last_event_msg = "🔀 [PESTAÑA INCORRECTA] Volviendo a la pestaña de Héroes automáticamente"
                print(f"\n{self.last_event_msg}\n")
                time.sleep(0.3)
                self.last_raw_rgb = self.screen_capture.capture_raw_rgb()

        # 0.1 Lógica de ASCENSIÓN FORZADA
        # Si la IA lleva estancada mucho tiempo (ej. 300 pasos sin récord de zona) y tiene al menos 10 almas pendientes
        if not self.mock_mode and self.last_raw_rgb is not None:
            if self.global_steps_stuck > 300 and self.max_zone_reached >= 135:
                souls_to_ascend = self.smart_vision.detect_hero_souls_to_ascend(self.last_raw_rgb)
                if souls_to_ascend is not None and souls_to_ascend >= 10:
                    self.last_event_msg = f"🔄 [ASCENSIÓN FORZADA] {souls_to_ascend} almas listas tras {self.global_steps_stuck} pasos atascado."
                    print(f"\n{self.last_event_msg}\n")
                    self.input_controller.force_ascension()
                    
                    # Reiniciar estado local tras ascensión
                    self.current_zone = 1
                    self.max_zone_reached = 1
                    self.global_steps_stuck = 0
                    self.steps_fighting_current_boss = 0
                    self.boss_fight_start_time = 0.0
                    self.scroll_depth = 0
                    self.needs_to_press_a = True
                    
                    # Pequeña penalización/recompensa. Recompensamos masivamente por ascender para que lo asocie si fuera RL, 
                    # pero como es macro, solo devolvemos el estado reseteado.
                    self.last_raw_rgb = self.screen_capture.capture_raw_rgb()
                    obs = cv2.resize(self.last_raw_rgb, self.target_size, interpolation=cv2.INTER_AREA)
                    if self.channel_first:
                        obs = np.transpose(obs, (2, 0, 1))
                    self.last_observation = obs
                    return obs, 500.0, False, False, {"is_mock": False, "current_zone": 1, "msg": self.last_event_msg}

        # Detección del estado del Jefe y cronometraje
        has_boss_timer = False
        # Solo comprobar el temporizador de jefe si matemáticamente estamos en un nivel de Jefe (múltiplo de 5)
        if self.current_zone % 5 == 0 and self.last_raw_rgb is not None:
            has_boss_timer = self.smart_vision.detect_boss_timer(self.last_raw_rgb)

        if has_boss_timer:
            if self.boss_fight_start_time == 0.0:
                self.boss_fight_start_time = time.time()
            self.steps_fighting_current_boss += 1
        else:
            self.steps_fighting_current_boss = 0
            self.boss_fight_start_time = 0.0

        # ----------------------------------------------------------------------
        # 1. ACCIÓN 4 (Avanzar a Siguiente Zona / Reintentar Boss)
        # ----------------------------------------------------------------------
        if action == InputController.ACTION_NEXT_ZONE_PROGRESS:
            if self.needs_to_press_a and not self.advance_locked:
                actual_action_executed = InputController.ACTION_NEXT_ZONE_PROGRESS
                blocked_action_penalty = 0.0
                self.needs_to_press_a = False  # Ya pulsó la 'a', se enciende la bota
            else:
                actual_action_executed = InputController.ACTION_NOOP
                blocked_action_penalty = -0.5

                if self.mock_mode:
                    # En modo Mock simulamos avance para pruebas unitarias
                    if self.current_zone < self.max_zone_reached:
                        self.current_zone += 1
                        self.steps_in_current_zone = 0
                        self.steps_fighting_current_boss = 0
                        self.boss_fight_start_time = 0.0
                    elif self.can_advance_to_new_zone:
                        is_boss_defeat = (self.current_zone % 5 == 0) or has_boss_timer
                        self.current_zone += 1
                        self.max_zone_reached = max(self.max_zone_reached, self.current_zone)
                        self.global_steps_stuck = 0
                        self.can_advance_to_new_zone = False
                        self.retreated_from_boss = False
                        self.steps_in_current_zone = 0
                        self.steps_fighting_current_boss = 0
                        self.boss_fight_start_time = 0.0
                        advance_bonus_reward = 100.0 if is_boss_defeat else 20.0


        # Incrementar contador de pasos en la zona actual
        self.steps_in_current_zone += 1

        upgrade_bonus_reward = 0.0
        real_upgrade_count = 0

        # Contar habilidades COMPRADA antes de ejecutar la acción (slot-based, estable)
        skills_bought_before = 0
        btns_before = []
        if actual_action_executed == InputController.ACTION_SMART_HERO_SKILLS and self.last_raw_rgb is not None:
            skills_bought_before = self.smart_vision.count_bought_hero_skills(self.last_raw_rgb)

        # Capturar botones +NV ANTES del clic para verificación post-acción
        if actual_action_executed == InputController.ACTION_SMART_UPGRADE and self.last_raw_rgb is not None:
            btns_before = self.smart_vision.detect_buyable_upgrade_buttons(self.last_raw_rgb)

        if actual_action_executed == InputController.ACTION_GLOBAL_SKILLS_BAR:
            self.last_event_msg = "⚡ [HABILIDADES ACTIVAS] Botones 1-9 ejecutados"

        # 2. Ejecutar acción de entrada virtual usando el fotograma a ALTA RESOLUCIÓN NATIVA para OpenCV
        clicked_coords = self.input_controller.execute_action(actual_action_executed, frame_rgb=self.last_raw_rgb)

        # 3. Simulate damage in mock mode for testing
        if self.mock_mode and actual_action_executed == InputController.ACTION_MONSTER_CLICK:
            self.reward_extractor.simulate_mock_damage(0.15)

        # Guardar explícitamente el fotograma de ANTES de la acción para la comparación de píxeles
        frame_before_action = self.last_raw_rgb
        
        # 4. Capturar nuevo estado: Fotograma nativo HD para OpenCV y (84, 84) para PyTorch CNN
        raw_rgb = self.screen_capture.capture_raw_rgb()
        self.last_raw_rgb = raw_rgb

        # B) Recompensar Scroll Efectivo (Enseñar a la IA a bajar)
        if actual_action_executed in [InputController.ACTION_SCROLL_DOWN, InputController.ACTION_SCROLL_UP] and clicked_coords is not None:
            # Comparamos la zona central (panel de héroes) para ver si realmente se movió la lista
            h, w = raw_rgb.shape[:2]
            y1, y2 = int(0.3 * h), int(0.9 * h)
            x1, x2 = int(0.05 * w), int(0.45 * w)
            roi_before_scroll = frame_before_action[y1:y2, x1:x2]
            roi_after_scroll = raw_rgb[y1:y2, x1:x2]
            
            if roi_before_scroll.size > 0 and roi_before_scroll.shape == roi_after_scroll.shape:
                diff_scroll = cv2.absdiff(roi_before_scroll, roi_after_scroll)
                changed_scroll = np.count_nonzero(diff_scroll > 15)
                
                if changed_scroll > 500: # El panel se ha movido (hay héroes nuevos)
                    if actual_action_executed == InputController.ACTION_SCROLL_DOWN:
                        self.scroll_depth += 1
                        # Recompensa directa e inmediata por bajar (tope de seguridad para evitar farmeo infinito)
                        if self.scroll_depth < 15:
                            advance_bonus_reward += 2.0
                        self.last_event_msg = f"📜 [SCROLL ABAJO] Profundidad: {self.scroll_depth} (+2.0 pts)"
                    else:
                        self.scroll_depth = max(0, self.scroll_depth - 1)
                        # Castigo fuerte por subir sin motivo
                        blocked_action_penalty = -2.0
                        self.last_event_msg = f"📜 [SCROLL ARRIBA] Profundidad: {self.scroll_depth} (-2.0 pts)"
                else:
                    # Chocó contra el fondo o el tope (no se movió nada)
                    blocked_action_penalty = -0.5
                    if actual_action_executed == InputController.ACTION_SCROLL_DOWN:
                        self.last_event_msg = f"🧱 [FONDO ALCANZADO] No se puede bajar más (-0.5 pts)"
                    else:
                        self.scroll_depth = 0
                        self.last_event_msg = f"🧱 [TOPE ALCANZADO] Arriba del todo (-0.5 pts)"

        # Verificación estricta DESPUÉS de la acción: ¿Realmente se compró algo?

        # A) Verificar compra de +NV
        # Comparamos la zona del botón antes y después del clic. Si el clic fue exitoso,
        # el texto del precio dentro del botón habrá cambiado. Si los píxeles son idénticos,
        # el clic falló o la compra fue rechazada por el juego.
        if actual_action_executed == InputController.ACTION_SMART_UPGRADE and btns_before and clicked_coords is not None:
            # Esperar suficiente tiempo para que el juego renderice el nuevo número (incluso si hay lag)
            time.sleep(0.25)
            raw_rgb = self.screen_capture.capture_raw_rgb() # Recapturar estado actualizado
            
            abs_x, abs_y = clicked_coords
            h, w = raw_rgb.shape[:2]
            
            # Extraer un ROI amplio que cubra TODO el botón (el precio está en la parte inferior)
            # El botón mide aprox 250x120 píxeles a resolución 2560x1440.
            y1, y2 = max(0, abs_y - 80), min(h, abs_y + 80)
            x1, x2 = max(0, abs_x - 150), min(w, abs_x + 150)
            
            roi_before = frame_before_action[y1:y2, x1:x2]
            roi_after = raw_rgb[y1:y2, x1:x2]
            
            if roi_before.size > 0 and roi_before.shape == roi_after.shape:
                diff = cv2.absdiff(roi_before, roi_after)
                # Contamos cuántos píxeles han cambiado significativamente (ignorando ruido leve)
                changed_pixels = np.count_nonzero(diff > 15)
                
                if changed_pixels > 5:
                    real_upgrade_count = 1
                    
                    # Si el grifo está cerrado por llevar demasiado tiempo estancado en la misma zona, anular puntos por mejoras
                    if self.steps_in_current_zone > self.max_zone_steps:
                        upgrade_bonus_reward = 0.0
                    else:
                        # Bonus multiplicador basado en la profundidad del scroll!
                        # Comprar en la superficie = +1.0 pts. Comprar a prof. 3 = +5.5 pts!
                        upgrade_bonus_reward = 1.0 + (self.scroll_depth * 1.5)
                        
                    msg = f"🛒 [MEJORA HÉROE] +NV comprado (+{upgrade_bonus_reward:.1f} pts)"
                    if self.advance_locked:
                        msg += f" | Mejoras: {self.upgrades_purchased_since_retreat + 1}/5"
                    self.last_event_msg = msg
                else:
                    self.last_event_msg = f"⚠️ [CLIC IGNORADO] El precio no cambió (diff={changed_pixels}), clic no registrado."
                    # [DEBUG] Guardar las imágenes para analizar por qué ha fallado
                    import os
                    debug_dir = r"C:\Users\marco\.gemini\antigravity-ide\brain\17f0ffef-b2cd-471d-8ad6-582303ff5e88\scratch"
                    if os.path.exists(debug_dir):
                        cv2.imwrite(os.path.join(debug_dir, "debug_roi_before.png"), roi_before)
                        cv2.imwrite(os.path.join(debug_dir, "debug_roi_after.png"), roi_after)

        # B) Verificar compra de habilidad de héroe: Tick verde ✔ nuevo
        if actual_action_executed == InputController.ACTION_SMART_HERO_SKILLS:
            skills_bought_after = self.smart_vision.count_bought_hero_skills(raw_rgb)
            if skills_bought_after > skills_bought_before:
                new_skills = skills_bought_after - skills_bought_before
                real_upgrade_count = new_skills
                upgrade_bonus_reward = 10.0 * new_skills
                self.last_event_msg = f"🌟 [HABILIDAD HÉROE] ¡Desbloqueada {new_skills} habilidad(es) (+{upgrade_bonus_reward:.1f} pts)!"

        if real_upgrade_count > 0:
            self.upgrades_purchased_since_retreat += real_upgrade_count
            # Al comprar 5 mejoras reales de héroes o habilidades, se desbloquea volver a desafiar al jefe
            if self.upgrades_purchased_since_retreat >= 5:
                if self.advance_locked:
                    self.last_event_msg = f"🔓 [CERROJO DESBLOQUEADO] ¡5 mejoras acumuladas! Desafío al jefe listo"
                self.advance_locked = False


        obs = cv2.resize(raw_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        if self.channel_first:
            obs = np.transpose(obs, (2, 0, 1))
        self.last_observation = obs

        # 5. Sincronización visual del nivel real mediante OpenCV / OCR con estabilidad temporal
        if not self.mock_mode and (self.current_step % 5 == 0 or actual_action_executed == InputController.ACTION_NEXT_ZONE_PROGRESS):
            detected_zone = self.smart_vision.detect_zone_number_from_screen(self.last_raw_rgb)
            if detected_zone is not None and 1 <= detected_zone <= 1000:
                self.zone_detection_history.append(detected_zone)
                if len(self.zone_detection_history) > 6:
                    self.zone_detection_history.pop(0)

                # Requiere que al menos 2 lecturas recientes confirmen el mismo nivel
                if self.zone_detection_history.count(detected_zone) >= 2:
                    # Si lee la misma zona 4 veces en el historial, confiamos plenamente aunque sea un salto grande
                    is_very_stable = self.zone_detection_history.count(detected_zone) >= 4
                    
                    # Evitar saltos de miles de niveles por errores consistentes de OCR
                    is_plausible_jump = (self.current_zone == 1 and detected_zone <= 200) or \
                                       (self.current_zone - 2 <= detected_zone <= self.current_zone + 3) or \
                                       (is_very_stable and abs(detected_zone - self.current_zone) <= 200)

                    if is_plausible_jump:
                        if detected_zone > self.current_zone:
                            # Actualizar zona interna pero NO dar recompensa todavía
                            self.steps_in_current_zone = 0
                            self.steps_fighting_current_boss = 0
                            self.boss_fight_start_time = 0.0

                            # ¿Es una zona NUEVA RÉCORD que nunca se ha alcanzado?
                            if detected_zone > self.max_zone_reached:
                                is_boss_defeat = ((detected_zone - 1) % 5 == 0) or (self.steps_fighting_current_boss > 0)
                                advance_bonus_reward += 200.0 if is_boss_defeat else 20.0
                                
                                self.max_zone_reached = detected_zone
                                self.current_zone = detected_zone
                                self.global_steps_stuck = 0
                                self.can_advance_to_new_zone = False
                                self.retreated_from_boss = False
                                self.advance_locked = False
                                self.last_event_msg = f"🏰 [¡NUEVO RÉCORD!] Avanza a Zona {detected_zone} (+{200.0 if is_boss_defeat else 20.0:.0f} pts)"
                            else:
                                # Volviendo a una zona ya alcanzada (ej: farm -> boss zone)
                                self.last_event_msg = f"↩️ [REGRESO] Volviendo a Zona {detected_zone} (ya alcanzada)"

                        elif detected_zone < self.current_zone:
                            # Filtro Anti-Glitches de OCR:
                            # El juego SÓLO te expulsa hacia atrás si pierdes contra un JEFE (Múltiplo de 5).
                            # Si estábamos en la zona 23 y leemos 22, es un 100% fallo de lectura del OCR.
                            if self.current_zone % 5 != 0:
                                # Ignoramos el error visual del OCR
                                detected_zone = self.current_zone
                            else:
                                self.steps_in_current_zone = 0
                                self.steps_fighting_current_boss = 0
                                self.boss_fight_start_time = 0.0
                                # Cancelar candidato si retrocede
                                self.new_zone_candidate = None
                                self.new_zone_first_seen = 0.0
                                self.zone_advance_rewarded = False
                                
                                # NUEVO: Si el juego nos tira para atrás automáticamente (por perder contra un jefe)
                                # Activamos el candado para obligar a la IA a farmear 5 mejoras antes de volver
                                self.advance_locked = True
                                self.needs_to_press_a = True
                                self.upgrades_purchased_since_retreat = 0
                                self.can_advance_to_new_zone = False
                                
                                self.last_event_msg = f"🛡️ [FARMEANDO] Retroceso forzado a Zona {detected_zone} (Candado Activado)"

                        self.current_zone = detected_zone

            # INVARIANTE: max_zone_reached solo se actualiza cuando se confirma con estabilidad (arriba)
            # Pero siempre debe ser >= current_zone en caso de que se establezca por primera vez
            if self.max_zone_reached < self.current_zone and self.zone_advance_rewarded:
                self.max_zone_reached = self.current_zone

        # 6. REWARD CAPPING (Grifo Cerrado): Si lleva > 60s (300 steps) en la misma zona, apagar recompensa de daño
        cap_damage_reward = self.steps_in_current_zone > self.max_zone_steps
        
        # NUEVO: Habilidades Activas (1-9) NUNCA dan recompensa de daño/oro directa.
        # Las habilidades siguen ejecutándose y matando monstruos, pero la IA solo se beneficia
        # indirectamente (avance de zona, que sí da +20 pts). Esto evita el exploit de spamear 1-9.
        if actual_action_executed == InputController.ACTION_GLOBAL_SKILLS_BAR:
            cap_damage_reward = True
        
        # Si el cerrojo está desbloqueado y la IA se niega a avanzar al jefe (pereza), apagar recompensa
        # Esto incluye estar en una zona anterior, o estar en la zona máxima con el modo de progresión apagado.
        is_progression_off = False
        if not self.mock_mode and self.last_raw_rgb is not None:
            is_progression_off = self.smart_vision.is_auto_progression_off(self.last_raw_rgb)
            
        if not self.advance_locked and (self.current_zone < self.max_zone_reached or is_progression_off):
            cap_damage_reward = True
            
        zone_changed = (actual_action_executed == InputController.ACTION_NEXT_ZONE_PROGRESS and self.current_zone < self.max_zone_reached)

        # Evaluar recompensa a ALTA RESOLUCIÓN NATIVA (con CERO penalización por retroceso táctico de boss)
        reward, reward_info = self.reward_extractor.compute_reward(
            self.last_raw_rgb,
            actual_action_executed,
            zone_changed=zone_changed,
            cap_damage_reward=cap_damage_reward,
            is_retreat_from_boss=False,
            is_boss_zone=(self.current_zone % 5 == 0)
        )
        
        # Penalización por pereza (Bleeding): Si el candado está abierto y NO estamos avanzando,
        # la IA está perdiendo el tiempo farmeando. Penalizamos.
        bleeding_penalty = 0.0
        if not self.advance_locked and (self.current_zone < self.max_zone_reached or is_progression_off):
            bleeding_penalty = -0.2  # Pierde puntos por cada paso que pasa sin avanzar
            if self.steps_in_current_zone % 50 == 0:
                self.last_event_msg = f"⚠️ [PEREZA] ¡Pierdes puntos por no avanzar a la Zona {self.max_zone_reached + 1 if is_progression_off else self.max_zone_reached}!"

        reward += blocked_action_penalty + advance_bonus_reward + skill_bonus_reward + upgrade_bonus_reward + bleeding_penalty

        # Registrar métricas para el display en tiempo real
        ACTION_NAME_MAP = {
            InputController.ACTION_NOOP: "NoOp",
            InputController.ACTION_SMART_UPGRADE: "+NV Mejora",
            InputController.ACTION_SMART_HERO_SKILLS: "Hab Heroe",
            InputController.ACTION_GLOBAL_SKILLS_BAR: "Hab Global(H)",
            InputController.ACTION_NEXT_ZONE_PROGRESS: "Avanzar/Boss",
            InputController.ACTION_MONSTER_CLICK: "Click",
            InputController.ACTION_SCROLL_DOWN: "Scroll Abajo",
            InputController.ACTION_SCROLL_UP: "Scroll Arriba",
        }
        self.last_action_name = ACTION_NAME_MAP.get(actual_action_executed, str(actual_action_executed))
        self.has_boss_timer_active = has_boss_timer
        self.last_step_reward = float(reward)
        self.episode_total_reward += float(reward)

        if reward_info.get("stage_completed", False):
            self.can_advance_to_new_zone = True
            self.retreated_from_boss = False
            
            # Ejecutar avance automático para no quedarse farmeando
            if (self.current_zone % 5 == 0) or (self.steps_fighting_current_boss > 0):
                self.advance_locked = False
                
                # Si el sistema nativo necesita pulsar 'a' porque falló el jefe, se lo indicamos
                if self.needs_to_press_a:
                    self.input_controller.execute_action(InputController.ACTION_NEXT_ZONE_PROGRESS, frame_rgb=self.last_raw_rgb)
                    self.needs_to_press_a = False
                
                if self.mock_mode:
                    # Nuevo récord validado
                    self.current_zone = self.new_zone_candidate if self.new_zone_candidate is not None else max(self.current_zone + 1, self.max_zone_reached + 1)
                    self.max_zone_reached = max(self.max_zone_reached, self.current_zone)
                    self.global_steps_stuck = 0
                    advance_bonus_reward = 20.0  # Gran premio por avanzar
                    self.last_event_msg = f"🏰 [¡NUEVO RÉCORD!] Avanza a Zona {self.current_zone} (+20 pts)"
                    self.zone_advance_rewarded = True
                self.steps_fighting_current_boss = 0
                self.boss_fight_start_time = 0.0
                # En modo LIVE: la recompensa vendrá del OCR cuando confirme la zona nueva
                self.last_event_msg = f"🔄 [AVANCE AUTOMÁTICO] Intentando avanzar desde Zona {self.current_zone} (recompensa pendiente de confirmación OCR)"








        # 6. Increment step counter and check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "current_zone": self.current_zone,
            "max_zone_reached": self.max_zone_reached,
            "steps_in_current_zone": self.steps_in_current_zone,
            "can_advance": self.can_advance_to_new_zone,
            "retreated_from_boss": self.retreated_from_boss,
            "advance_locked": self.advance_locked,
            "reward_capped": cap_damage_reward,
            "blocked_action": actual_action_executed != action,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info





    def render(self) -> Optional[np.ndarray]:
        """
        Renders observation according to render_mode.
        """
        if self.last_observation is None:
            return None

        # Convert (C, H, W) -> (H, W, C) for visualization
        rgb_frame = np.transpose(self.last_observation, (1, 2, 0))

        if self.render_mode == "rgb_array":
            return rgb_frame
        return None

    def close(self) -> None:
        """
        Clean up resources.
        """
        self.screen_capture.close()
