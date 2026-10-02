"""
Virtual Input Controller Module for Game AI (Phase 2 & 3)
Simulates mouse clicks, scrolling, and keypresses using pydirectinput with fail-safe bounds and emergency stop checks.
"""
import time
from typing import Dict, Tuple, Optional
import numpy as np
import pydirectinput


from src.utils.safety import global_emergency_listener, EmergencyStopException
from src.utils.smart_vision import SmartVisionController

# Configure pydirectinput defaults
pydirectinput.FAILSAFE = True
pydirectinput.PAUSE = 0.005



class InputController:
    """
    Simulates input controls for Clicker Heroes using normalized coordinates.
    Calibrated for Clicker Heroes UI layout (Monster island, Hero upgrade panel, Zone arrows, Scroll).
    """

    # Defined discrete actions for Clicker Heroes / Idle Slayer (9 Smart Actions)
    ACTION_MONSTER_CLICK = 0       # Ráfaga de Clics en Monstruo (Combate / Ataque / Salto)
    ACTION_SMART_UPGRADE = 1       # Detección OpenCV y clic en Botones Azules Comprables (+NV)
    ACTION_SMART_HERO_SKILLS = 2   # Clic en Habilidades de Héroes (Pequeños iconos bajo cada héroe)
    ACTION_GLOBAL_SKILLS_BAR = 3   # Clic en Barra de Habilidades Activas Globales (Clickstorm, Powersurge, etc.)
    ACTION_NEXT_ZONE_PROGRESS = 4  # Flecha Derecha: Modo Progreso (Avanzar a siguiente zona / intentar Boss)
    ACTION_SCROLL_DOWN = 5         # Scroll Abajo en Panel de Héroes
    ACTION_SCROLL_UP = 6           # Scroll Arriba en Panel de Héroes
    ACTION_NOOP = 7                # No-op / Esperar

    # Calibrated relative coordinates (x_rel, y_rel) in range [0.0, 1.0] inside the game window
    DEFAULT_ACTION_MAP: Dict[int, Tuple[float, float]] = {
        ACTION_MONSTER_CLICK: (0.73, 0.55),       # Isla flotante / Zona de monstruos (center-right)
        ACTION_SMART_UPGRADE: (0.095, 0.46),      # Coordenada base para mejorías de héroes
        ACTION_SMART_HERO_SKILLS: (0.22, 0.50),   # Coordenada base para habilidades de héroes
        ACTION_GLOBAL_SKILLS_BAR: (0.521, 0.292),  # Columna vertical de habilidades activas globales (H1)

        ACTION_NEXT_ZONE_PROGRESS: (0.795, 0.055),# Imagen miniatura de Nivel Siguiente (Tile 4) -> Progreso / Boss
        ACTION_SCROLL_DOWN: (0.25, 0.65),         # Centro de panel de héroes para scroll down
        ACTION_SCROLL_UP: (0.25, 0.65),           # Centro de panel de héroes para scroll up
    }




    def __init__(
        self,
        region: Optional[Dict[str, int]] = None,
        action_map: Optional[Dict[int, Tuple[float, float]]] = None,
        action_delay: float = 0.02,
        turbo_clicks: int = 5,
        mock_mode: bool = False,
    ):
        """
        :param region: Dict with 'top', 'left', 'width', 'height'. Defaults to 1280x720 window at (0, 0).
        :param action_map: Custom relative coordinate mapping for actions.
        :param action_delay: Sleep time in seconds after performing an action.
        :param turbo_clicks: Number of rapid clicks to send for ACTION_MONSTER_CLICK.
        :param mock_mode: If True, logs actions without moving mouse hardware.
        """
        if region is None:
            self.region = {"top": 0, "left": 0, "width": 1280, "height": 720}
        else:
            self.region = region

        self.ACTION_MAP = action_map or dict(self.DEFAULT_ACTION_MAP)
        self.action_delay = action_delay
        self.turbo_clicks = turbo_clicks

        self.mock_mode = mock_mode
        self.last_action_time = 0.0
        self.smart_vision = SmartVisionController()

        if not self.mock_mode:
            try:
                pydirectinput.FAILSAFE = False
            except Exception:
                pass



    def set_region(self, top: int, left: int, width: int, height: int) -> None:
        """Updates window bounding box for action mapping."""
        self.region = {"top": top, "left": left, "width": width, "height": height}

    def relative_to_absolute(self, x_rel: float, y_rel: float) -> Tuple[int, int]:
        """
        Converts normalized coordinates (0.0 - 1.0) relative to game region into absolute screen pixels.
        """
        x_rel = max(0.0, min(1.0, x_rel))
        y_rel = max(0.0, min(1.0, y_rel))

        abs_x = int(self.region["left"] + x_rel * self.region["width"])
        abs_y = int(self.region["top"] + y_rel * self.region["height"])
        return abs_x, abs_y

    def click(self, x_rel: float, y_rel: float, clicks: int = 1, hide_tooltip: bool = False) -> Tuple[int, int]:
        """
        Performs left click(s) at the given relative coordinates.
        Checks EmergencyStopListener before clicking and raises EmergencyStopException if triggered.
        :param hide_tooltip: If True, moves the mouse to a safe zone after clicking to hide UI tooltips.
        :return: (abs_x, abs_y) targeted.
        """
        abs_x, abs_y = self.relative_to_absolute(x_rel, y_rel)

        # Safety Check: Instantly raise EmergencyStopException if stop is requested
        global_emergency_listener.raise_if_stopped()

        if not self.mock_mode:
            try:
                for _ in range(clicks):
                    global_emergency_listener.raise_if_stopped()
                    # Mover el ratón primero y hacer una pausa de 2 frames (60ms) evita clics "fantasma" que solo hacen hover
                    pydirectinput.moveTo(abs_x, abs_y)
                    time.sleep(0.06)
                    pydirectinput.mouseDown(abs_x, abs_y)
                    time.sleep(0.12) # Clic duro de 120ms para asegurar que el motor lo pille
                    pydirectinput.mouseUp(abs_x, abs_y)
                    time.sleep(0.05)
                
                if hide_tooltip:
                    safe_x, safe_y = self.relative_to_absolute(0.9, 0.1) # Zona segura sin tooltips
                    pydirectinput.moveTo(safe_x, safe_y)

                if self.action_delay > 0:
                    time.sleep(self.action_delay)
            except Exception as e:
                if isinstance(e, EmergencyStopException):
                    raise e
                print(f"⚠️ Alerta de InputController: {e}")

        # Check safety again right after click execution
        global_emergency_listener.raise_if_stopped()

        self.last_action_time = time.time()
        return abs_x, abs_y

    def scroll(self, x_rel: float, y_rel: float, clicks: int = -5, hide_tooltip: bool = True) -> Tuple[int, int]:
        """
        Moves mouse to (x_rel, y_rel) and scrolls mouse wheel using Win32 DirectX mouse_event.
        :param clicks: Negative for scroll down, positive for scroll up.
        :param hide_tooltip: If True, moves the mouse to a safe zone after scrolling to hide UI tooltips.
        """
        abs_x, abs_y = self.relative_to_absolute(x_rel, y_rel)
        global_emergency_listener.raise_if_stopped()

        if not self.mock_mode:
            try:
                pydirectinput.moveTo(abs_x, abs_y)
                time.sleep(0.05) # Pequeña pausa para asegurar que el foco del ratón está en el panel
                
                import ctypes
                # MOUSEEVENTF_WHEEL = 0x0800, WHEEL_DELTA = 120
                # Si el scroll es muy grande, lo dividimos en pequeños "tirones" para que el motor del juego no lo ignore
                steps = abs(clicks) // 5
                if steps == 0: steps = 1
                
                sign = 1 if clicks > 0 else -1
                wheel_delta_per_step = (5 * sign) * 120
                
                for _ in range(steps):
                    ctypes.windll.user32.mouse_event(0x0800, 0, 0, wheel_delta_per_step, 0)
                    time.sleep(0.03) # Pausa entre tirones para que el juego renderice el movimiento
                
                # Scroll residual si clicks no es múltiplo de 5
                remainder = abs(clicks) % 5
                if remainder > 0:
                    ctypes.windll.user32.mouse_event(0x0800, 0, 0, (remainder * sign) * 120, 0)
                
                time.sleep(0.2) # Pausa final para que la UI termine de deslizarse antes de capturar pantalla
                
                if hide_tooltip:
                    safe_x, safe_y = self.relative_to_absolute(0.9, 0.1)
                    pydirectinput.moveTo(safe_x, safe_y)

                if self.action_delay > 0:
                    time.sleep(self.action_delay)
            except Exception as e:
                if isinstance(e, EmergencyStopException):
                    raise e
                print(f"⚠️ Alerta de InputController: {e}")

        global_emergency_listener.raise_if_stopped()
        return abs_x, abs_y


    def soft_reset_game_save(self, save_data: str) -> None:
        """
        Ejecuta la secuencia para cargar una partida guardada desde un string (Soft Reset).
        Esto mantiene los desbloqueos globales (como la Progresión Automática).
        1. Clic en icono de Ajustes (Tuerca/Llave) en esquina superior derecha (0.965, 0.05)
        2. Clic en botón 'Importar' en la columna izquierda (0.30, 0.35)
        3. Clic en botón 'Iniciar importación' que lee el portapapeles (0.50, 0.60)
        4. Clic en botón 'Importar' para confirmar (0.40, 0.85)
        5. Clic en la 'X' roja para cerrar la ventana de bienvenida (0.79, 0.12)
        """
        global_emergency_listener.raise_if_stopped()

        if self.mock_mode:
            print("[MOCK] Simulando reinicio suave de partida (Soft Reset)...")
            return

        try:
            import pyperclip
        except ImportError:
            print("⚠️ [ADVERTENCIA] El paquete 'pyperclip' no está instalado. Instálalo con 'pip install pyperclip' para habilitar el Soft Reset.")
            return

        print("\n🔄 [SOFT RESET] Importando partida base para evaluación justa...")
        try:
            pyperclip.copy(save_data)
            time.sleep(0.1)

            # 1. Clic en icono de Llave Inglesa (Opciones) arriba a la derecha
            self.click(0.96, 0.05)
            time.sleep(0.5)

            # 2. Clic en "Importar"
            self.click(0.60, 0.88)
            time.sleep(0.5)

            # 3. Hacer clic en el cuadro de texto para enfocar
            self.click(0.50, 0.50)
            time.sleep(0.2)

            # 4. Pegar texto (Ctrl + V)
            pydirectinput.keyDown('ctrl')
            time.sleep(0.05)
            pydirectinput.press('v')
            time.sleep(0.05)
            pydirectinput.keyUp('ctrl')
            time.sleep(0.2)

            # 5. Clic en el botón "Importar" de la ventana modal
            self.click(0.48, 0.69)
            time.sleep(1.0)
            
            # 6. Clic en la X roja del popup de bienvenida
            self.click(0.79, 0.12)
            time.sleep(0.5)

            print("✔ [SOFT RESET COMPLETADO] Partida base cargada exitosamente.\n")
        except Exception as e:
            if isinstance(e, EmergencyStopException):
                raise e
            print(f"⚠️ Alerta durante soft reset de partida: {e}")

        global_emergency_listener.raise_if_stopped()

    def execute_action(self, action: int, frame_rgb: Optional[np.ndarray] = None) -> Optional[Tuple[int, int]]:
        """
        Executes a discrete action index (0..8) with intelligent vision support.
        :param action: Discrete action integer (0..8).
        :param frame_rgb: Optional RGB screen frame for OpenCV vision analysis.
        :return: Absolute coordinates clicked/scrolled, or None for NOOP.
        """
        global_emergency_listener.raise_if_stopped()

        if action == self.ACTION_NOOP:
            if not self.mock_mode and self.action_delay > 0:
                time.sleep(self.action_delay)
            return None

        if action == self.ACTION_SCROLL_DOWN:
            x_rel, y_rel = self.ACTION_MAP[action]
            # Usamos -15 para bajar un poco (un tercio de página aprox) sin saltar héroes del medio
            return self.scroll(x_rel, y_rel, clicks=-15)

        if action == self.ACTION_SCROLL_UP:
            x_rel, y_rel = self.ACTION_MAP[action]
            return self.scroll(x_rel, y_rel, clicks=15)

        if action == self.ACTION_NEXT_ZONE_PROGRESS:
            # Modo Progreso: Pulsamos simplemente la tecla 'a'
            if not self.mock_mode:
                try:
                    pydirectinput.press('a')
                    time.sleep(0.1)
                except Exception:
                    pass
            return self.relative_to_absolute(0.88, 0.28)




        if action == self.ACTION_GLOBAL_SKILLS_BAR:
            # 1. Enviar hotkeys directas (teclas 1..9 del teclado de Clicker Heroes para activar habilidades H)
            if not self.mock_mode:
                try:
                    for k in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                        pydirectinput.press(k)
                except Exception:
                    pass



            # 3. Fallback: Clic en la columna vertical de habilidades
            skill_y_positions = [0.292, 0.370, 0.449, 0.527, 0.605, 0.684, 0.762, 0.840, 0.919]
            y_target = skill_y_positions[int(time.time() * 2) % len(skill_y_positions)]
            return self.click(0.521, y_target, clicks=1)








        if action == self.ACTION_SMART_HERO_SKILLS:
            if frame_rgb is not None and hasattr(self, "smart_vision"):
                detected_skills = self.smart_vision.detect_hero_skill_icons(frame_rgb)
                if detected_skills:
                    x_rel, y_rel = detected_skills[int(time.time() * 3) % len(detected_skills)]
                    return self.click(x_rel, y_rel, clicks=1, hide_tooltip=True)
            return None

        if action == self.ACTION_SMART_UPGRADE:
            def _click_with_q(x_r, y_r):
                if not self.mock_mode:
                    try:
                        pydirectinput.keyDown('q')
                        time.sleep(0.05)
                        return self.click(x_r, y_r, clicks=1, hide_tooltip=True)
                    finally:
                        try:
                            pydirectinput.keyUp('q')
                        except Exception:
                            pass
                return self.click(x_r, y_r, clicks=1, hide_tooltip=True)

            # Detección dinámica OpenCV de botones azules (+NV) tras cualquier scroll
            if frame_rgb is not None and hasattr(self, "smart_vision"):
                buyable_coords = self.smart_vision.detect_buyable_upgrade_buttons(frame_rgb)
                if buyable_coords:
                    x_rel, y_rel = buyable_coords[int(time.time() * 4) % len(buyable_coords)]
                    return _click_with_q(x_rel, y_rel)
            
            return None

        if action == self.ACTION_MONSTER_CLICK:
            x_rel, y_rel = self.ACTION_MAP[action]
            return self.click(x_rel, y_rel, clicks=self.turbo_clicks)

        raise ValueError(f"Invalid action index: {action}. Valid actions are 0 to 7.")


    def force_ascension(self):
        """
        Macro directa para Ascender usando el botón de acceso rápido (remolino rojo).
        Se asume que Amenhotep ya ha sido subido de nivel de forma natural por la IA.
        """
        print("🚨 INICIANDO MACRO DE ASCENSIÓN (PORTAL) 🚨")
        
        # 1. Clic en el Portal de Ascensión (remolino rojo a la derecha de la pantalla)
        self.click(0.972, 0.355, hide_tooltip=False)
        time.sleep(1.0)
        
        # 2. Clic en el botón verde "Sí" en el popup de confirmación
        self.click(0.43, 0.69, hide_tooltip=False)
        time.sleep(2.0)
        
        print("✅ MACRO DE ASCENSIÓN TERMINADO")
