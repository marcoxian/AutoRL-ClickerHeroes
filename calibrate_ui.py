"""
Herramienta de Calibración Visual de UI para Clicker Heroes.

Este script busca la ventana de Clicker Heroes (o la última captura tomada),
captura un fotograma a resolución nativa y dibuja sobre la imagen las zonas de detección
de vida (HP Bar ROI) y los puntos de clic de cada acción (incluyendo Scroll Down y Scroll Up).

Ejecutar:
    python calibrate_ui.py
"""

import sys
import os
import cv2
import numpy as np

from src.utils.window_finder import find_game_window, WindowNotFoundError
from src.utils.screen_capture import ScreenCapture
from src.utils.input_controller import InputController
from src.utils.reward_extractor import RewardExtractor


def main():
    print("=" * 65)
    print("[CALIBRACION UI] Verificando Puntos de Clic y Zona de Vida...")
    print("=" * 65 + "\n")

    raw_frame_bgr = None
    save_path = "ui_calibration_preview.png"

    try:
        region = find_game_window(title_substring="Clicker Heroes", bring_to_front=True)
        w, h = region["width"], region["height"]
        print(f"[RESOLUCION] Resolucion detectada de la ventana: {w} x {h} px")

        cap = ScreenCapture(region=region, target_size=(w, h), channel_first=False, mock_mode=False)
        raw_frame_rgb = cap.capture_frame()
        cap.close()
        raw_frame_bgr = cv2.cvtColor(raw_frame_rgb, cv2.COLOR_RGB2BGR)
    except WindowNotFoundError:
        if os.path.exists(save_path):
            print(f"[AVISO] Usando captura de pantalla previa guardada ({save_path})...")
            raw_frame_bgr = cv2.imread(save_path)
            h, w = raw_frame_bgr.shape[0], raw_frame_bgr.shape[1]
            region = {"top": 0, "left": 0, "width": w, "height": h}
            print(f"[RESOLUCION] Resolucion de la captura: {w} x {h} px")
        else:
            print("[AVISO] No se encontro la ventana del juego ni captura previa. Usando MOCK...")
            region = {"top": 0, "left": 0, "width": 1280, "height": 720}
            w, h = 1280, 720
            cap = ScreenCapture(region=region, target_size=(w, h), channel_first=False, mock_mode=True)
            raw_frame_rgb = cap.capture_frame()
            cap.close()
            raw_frame_bgr = cv2.cvtColor(raw_frame_rgb, cv2.COLOR_RGB2BGR)

    canvas = raw_frame_bgr.copy()

    # 1. Dibujar Zona ROI de la Barra de Vida (HP Bar)
    extractor = RewardExtractor(mock_mode=True)
    y1, x1, y2, x2 = (
        int(extractor.hp_bar_roi[0] * h),
        int(extractor.hp_bar_roi[1] * w),
        int(extractor.hp_bar_roi[2] * h),
        int(extractor.hp_bar_roi[3] * w),
    )
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(canvas, "BARRA DE VIDA (HP Bar ROI)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 2. Dibujar Puntos de Clic para las Acciones Inteligentes
    action_colors = {
        InputController.ACTION_MONSTER_CLICK: ((0, 0, 255), "Accion 0: Zona Combate (Monster Click)"),
        InputController.ACTION_SMART_UPGRADE: ((0, 255, 255), "Accion 1: Mejorias Inteligentes Héroes (+NV/CONT)"),
        InputController.ACTION_SMART_HERO_SKILLS: ((255, 255, 0), "Accion 2: Habilidades de Héroes (Iconos)"),
        InputController.ACTION_GLOBAL_SKILLS_BAR: ((0, 255, 128), "Accion 3: Barra Habilidades Activas Globales"),
        InputController.ACTION_PREV_ZONE_FARM: ((255, 128, 0), "Accion 4: Flecha Izq (Modo Farmeo de Oro)"),
        InputController.ACTION_NEXT_ZONE_PROGRESS: ((255, 0, 255), "Accion 5: Flecha Der (Modo Progreso / Boss)"),
        InputController.ACTION_SCROLL_DOWN: ((0, 165, 255), "Accion 6: Scroll Down Panel Héroes"),
        InputController.ACTION_SCROLL_UP: ((128, 0, 128), "Accion 7: Scroll Up Panel Héroes"),
    }



    ctrl = InputController(region=region, mock_mode=True)

    print("\n[PUNTOS DE CLIC] Puntos de Clic Calculados (Coordenadas Absolutas en Pantalla):")
    for action_id, (x_rel, y_rel) in ctrl.ACTION_MAP.items():
        abs_x, abs_y = ctrl.relative_to_absolute(x_rel, y_rel)
        rel_x_px, rel_y_px = int(x_rel * w), int(y_rel * h)
        color, label = action_colors[action_id]

        # Dibujar círculo objetivo
        cv2.circle(canvas, (rel_x_px, rel_y_px), 20, color, -1)
        cv2.circle(canvas, (rel_x_px, rel_y_px), 30, (255, 255, 255), 3)
        cv2.putText(canvas, f"{label} ({rel_x_px}, {rel_y_px})", (rel_x_px + 35, rel_y_px + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print(f"  - {label}: Relativo ({x_rel:.2f}, {y_rel:.2f}) -> Pantalla (X={abs_x}, Y={abs_y})")

    # Guardar vista previa
    cv2.imwrite(save_path, canvas)

    print("\n" + "=" * 65)
    print(f"[EXITO] CALIBRACION COMPLETADA! Vista previa guardada en:")
    print(f"   file:///{os.path.abspath(save_path).replace('\\', '/')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
