"""
Script de Prueba de Detección de Ventana para Clicker Heroes.

Ejecutar:
    python test_window_detection.py
"""

import sys
import os
import cv2

from src.utils.window_finder import find_game_window, WindowNotFoundError, list_visible_windows
from src.utils.screen_capture import ScreenCapture


def main():
    print("=" * 60)
    print("[TEST] Probando Deteccion de Ventana de Clicker Heroes...")
    print("=" * 60 + "\n")

    try:
        region = find_game_window(title_substring="Clicker Heroes", bring_to_front=True)
        print("\n[CAPTURA] Probando captura de fotograma sobre la ventana encontrada...")



        cap = ScreenCapture(region=region, target_size=(128, 128), channel_first=False, mock_mode=False)
        frame_rgb = cap.capture_frame()
        cap.close()

        # Convert to BGR for OpenCV save
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        save_path = "test_clicker_window_preview.png"
        cv2.imwrite(save_path, frame_bgr)

        print(f"[EXITO] PRUEBA EXITOSA! Ventana capturada correctamente.")
        print(f"[GUARDADO] Vista previa guardada en: {os.path.abspath(save_path)}")

    except WindowNotFoundError as e:
        print(e)
        print("\n[SUGERENCIA] Abre el juego 'Clicker Heroes' en tu ordenador e intenta ejecutar este comando de nuevo.")
    except Exception as e:
        print(f"\n[ERROR] Error inesperado durante la prueba: {e}")


if __name__ == "__main__":
    main()
