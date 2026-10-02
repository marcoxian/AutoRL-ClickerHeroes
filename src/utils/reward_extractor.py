"""
Visual Reward and State Extractor Module for Game AI (Phase 2 & 3)
Uses OpenCV HSV color filtering and ROI analysis to extract HP bars, kill signals, and rewards.
"""
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np


class RewardExtractor:
    """
    Analyzes visual game screen frames to compute rewards for Clicker Heroes.
    """

    def __init__(
        self,
        hp_bar_roi: Tuple[float, float, float, float] = (0.83, 0.68, 0.88, 0.88),
        damage_multiplier: float = 10.0,
        kill_bonus: float = 5.0,
        step_penalty: float = 0.01,
        mock_mode: bool = False,
    ):
        """
        :param hp_bar_roi: (y_min_rel, x_min_rel, y_max_rel, x_max_rel) relative bounding box for HP bar.
        :param damage_multiplier: Scale for damage dealt.
        :param kill_bonus: Reward bonus when a monster is defeated.
        :param step_penalty: Time penalty per step to encourage active clicking.
        :param mock_mode: Synthetic testing mode.
        """
        self.hp_bar_roi = hp_bar_roi
        self.damage_multiplier = damage_multiplier
        self.kill_bonus = kill_bonus
        self.step_penalty = step_penalty
        self.mock_mode = mock_mode

        self.last_hp: float = 1.0
        self.mock_hp: float = 1.0
        self.consecutive_kills: int = 0
        self.last_actions: List[int] = []
        self.skill_cooldown: int = 0


    def extract_hp(self, frame_rgb: np.ndarray) -> float:
        """
        Extracts current HP percentage (0.0 to 1.0) from screen frame.
        Expects frame in (H, W, 3) RGB format or (3, H, W).
        """
        if self.mock_mode:
            return self.mock_hp

        # Ensure frame is (H, W, C)
        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            # Transpose (C, H, W) to (H, W, C)
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        # ROI calibrada de la barra de vida del enemigo
        y1, x1, y2, x2 = int(0.835 * h), int(0.68 * w), int(0.885 * h), int(0.82 * w)

        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return 1.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # 1. Máscara de vida llena (Gradiente Naranja/Rojo/Verde del juego)
        mask_orange_red = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([25, 255, 255])) | \
                          cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        fill_mask = mask_orange_red | mask_green
        fill_pixels = cv2.countNonZero(fill_mask)

        # 2. Máscara de vida vacía (Fondo gris oscuro/negro de la barra)
        empty_mask = cv2.inRange(hsv, np.array([0, 0, 10]), np.array([180, 120, 85]))
        empty_pixels = cv2.countNonZero(empty_mask)

        bar_pixels = fill_pixels + empty_pixels
        if bar_pixels < 50:
            return 1.0  # Fallback si no se detecta la barra

        hp_ratio = fill_pixels / float(bar_pixels)
        return float(np.clip(hp_ratio, 0.0, 1.0))


    def detect_monster_dead_text(self, frame_rgb: np.ndarray) -> bool:
        """
        Detecta si en la barra de vida aparece la palabra 'Muerto' o 'Dead' mediante OCR.
        Ya NO usa el conteo de píxeles blancos (que causaba falsos positivos constantes).
        """
        if frame_rgb is None or frame_rgb.size == 0 or self.mock_mode:
            return False

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]
        crop = frame_rgb[int(0.175 * h):int(0.225 * h), int(0.60 * w):int(0.86 * w)]
        if crop.size == 0:
            return False

        # Solo comprobar texto 'Muerto' / 'Dead' mediante OCR - el único método fiable
        try:
            if not hasattr(self, "_ocr_reader"):
                import easyocr
                self._ocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
            res = self._ocr_reader.readtext(crop, detail=0)
            text = " ".join(res).lower()
            if "muert" in text or "dead" in text:
                return True
        except Exception:
            pass

        return False

    def compute_reward(
        self,
        frame_rgb: np.ndarray,
        action: int,
        zone_changed: bool = False,
        cap_damage_reward: bool = False,
        is_retreat_from_boss: bool = False,
        is_boss_zone: bool = False
    ) -> Tuple[float, Dict[str, Any]]:

        """
        Computes step reward based EXCLUSIVELY on result-oriented metrics (OpenCV HP reduction, genuine kills, 10/10 stage clear).
        Includes Reward Capping (grifo cerrado tras 60s en la misma zona) and anti-exploit oscillation locks.
        """
        current_hp = self.extract_hp(frame_rgb)
        hp_change = self.last_hp - current_hp

        # 1. Penalización fija por tiempo (-0.01 por step)
        reward = -self.step_penalty
        monster_killed = False
        stage_completed = False

        # Registrar historial de las últimas 8 acciones para detección de bucles
        self.last_actions.append(action)
        if len(self.last_actions) > 8:
            self.last_actions.pop(0)

        # 2. CANDADO ANTI-OSCILACIÓN: Penalización de -3.0 por bucles repetitivos entre flechas (4, 5) y otras acciones
        if len(self.last_actions) >= 4:
            pattern = self.last_actions[-4:]
            if pattern in ([4, 5, 4, 5], [5, 4, 5, 4], [4, 3, 5, 3], [5, 3, 4, 3], [3, 4, 3, 5]):
                reward -= 3.0  # Penalización severa por intentar explotar bucles de zona

        # 3. CAMBIO DE ZONA (Acciones 4 ó 5)
        if zone_changed:
            # Si es un retroceso táctico de jefe (is_retreat_from_boss), CERO penalización
            if not is_retreat_from_boss:
                reward -= 0.5  # Coste por cambio de zona estándar
            self.last_hp = current_hp
            info = {
                "current_hp": current_hp,
                "hp_change": 0.0,
                "monster_killed": False,
                "stage_completed": False,
                "consecutive_kills": self.consecutive_kills,
                "step_reward": reward,
            }
            return float(reward), info


        # NO HAY RECOMPENSA POR DAÑO INTERMEDIO (+ΔHP*10.0 ELIMINADO para evitar Reward Hacking)
        # NO HAY RECOMPENSA DIRECTA POR EJECUTAR ACCIONES DE INTERFAZ (1, 2, 3, 6, 7, 8)

        # 4. RECOMPENSAS ESTRICTAS POR EVENTOS FINALES (Muerte real vía 'Muerto' / sin números / salto de vida)
        is_dead_text = self.detect_monster_dead_text(frame_rgb)
        # Si estamos en un jefe, el HP se reinicia al 100% por timeout cada 30s. No contar eso como kill.
        hp_respawn_jump = (current_hp - self.last_hp > 0.40) and not is_boss_zone
        hp_reached_zero = (current_hp < 0.05 and self.last_hp > 0.12)

        if is_dead_text or hp_respawn_jump or hp_reached_zero:
            # Recompensa por baja: 5.0 normal, o reducida a 0.2 si el agente se estanca en farmeo sin avanzar
            monster_killed = True
            self.consecutive_kills += 1
            kill_gain = 0.0 if cap_damage_reward else self.kill_bonus
            reward += kill_gain

            # Marcar stage_completed al alcanzar 10/10 o derrotar al Boss
            if self.consecutive_kills >= 10:
                stage_completed = True
                self.consecutive_kills = 0


        self.last_hp = current_hp


        info = {
            "current_hp": current_hp,
            "hp_change": hp_change,
            "monster_killed": monster_killed,
            "stage_completed": stage_completed,
            "consecutive_kills": self.consecutive_kills,
            "step_reward": reward,
        }
        return float(reward), info



    def reset(self, initial_hp: float = 1.0) -> None:
        """Reset internal state tracker."""
        self.last_hp = initial_hp
        self.mock_hp = initial_hp
        self.consecutive_kills = 0
        self.last_actions = []
        self.skill_cooldown = 0


    def simulate_mock_damage(self, damage: float = 0.2) -> None:
        """Simulates damage in mock testing mode."""
        self.mock_hp -= damage
        if self.mock_hp <= 0.0:
            self.mock_hp = 1.0  # Respawn new monster
