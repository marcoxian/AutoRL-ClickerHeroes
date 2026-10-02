"""
Script de Depuración Visual en Tiempo Real (debug_vision.py)
Renderiza una ventana interactiva nativa con todas las detecciones de OpenCV:
- Barra de vida del enemigo (HP %)
- Barra y reloj de tiempo del Jefe (detect_boss_timer)
- Estado de los 9 slots de habilidades globales (H1..H9)
- Botones de subida de nivel de héroes (+NV comprables)
- Habilidades de héroes (✔ Compradas vs ★ Comprables vs 🔒 Bloqueadas)
- Región y texto de lectura OCR de Zona

Uso:
    python debug_vision.py
    (Pulsa 'q' o 'ESC' para salir)
"""
import sys
import time
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Configurar encoding UTF-8 seguro para Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.utils.screen_capture import ScreenCapture
from src.utils.smart_vision import SmartVisionController
from src.utils.reward_extractor import RewardExtractor
from src.utils.window_finder import find_game_window, WindowNotFoundError


class VisionDebuggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoRL Vision Debugger - Clicker Heroes")
        self.root.geometry("1280x720")
        self.root.configure(bg="#111111")

        try:
            region = find_game_window(title_substring="Clicker Heroes", bring_to_front=True)
            self.cap = ScreenCapture(region=region, target_size=None, channel_first=False, mock_mode=False)
        except WindowNotFoundError as e:
            print(e)
            print("\n[MODO MOCK] Abriendo en modo simulado para pruebas locales.")
            self.cap = ScreenCapture(target_size=(720, 1280), channel_first=False, mock_mode=True)

        self.vision = SmartVisionController()
        self.reward_ext = RewardExtractor(mock_mode=False)

        self.lbl_image = tk.Label(self.root, bg="#111111")
        self.lbl_image.pack(fill=tk.BOTH, expand=True)

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_display = 0.0
        self.running = True

        # Atajos de teclado para salir
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("q", lambda e: self.close())
        self.root.bind("Q", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        frame_rgb = self.cap.capture_frame()
        if frame_rgb is not None and frame_rgb.size > 0:
            h, w = frame_rgb.shape[:2]
            sc = max(1.0, w / 1280.0)

            canvas = frame_rgb.copy()

            # 1. Barra de Vida del Enemigo (HP Bar ROI)
            hp_val = self.reward_ext.extract_hp(frame_rgb)
            hp_y1, hp_x1 = int(0.835 * h), int(0.68 * w)
            hp_y2, hp_x2 = int(0.885 * h), int(0.82 * w)
            hp_color = (0, 255, 0) if hp_val > 0.2 else (255, 0, 0)
            cv2.rectangle(canvas, (hp_x1, hp_y1), (hp_x2, hp_y2), hp_color, int(2 * sc))
            cv2.putText(canvas, f"HP: {hp_val * 100:.1f}%", (hp_x1, hp_y1 - int(10 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc, hp_color, int(2 * sc))

            # 2. Barra y Reloj de Tiempo del Jefe (Boss Timer ROI)
            boss_active = self.vision.detect_boss_timer(frame_rgb)
            boss_y1, boss_x1 = int(0.215 * h), int(0.705 * w)
            boss_y2, boss_x2 = int(0.265 * h), int(0.795 * w)
            boss_color = (255, 255, 0) if boss_active else (120, 120, 120)
            boss_text = "BOSS: TEMPORIZADOR ACTIVO" if boss_active else "BOSS: INACTIVO (Normal)"
            cv2.rectangle(canvas, (boss_x1, boss_y1), (boss_x2, boss_y2), boss_color, int(2 * sc))
            cv2.putText(canvas, boss_text, (boss_x1, boss_y1 - int(10 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, boss_color, int(2 * sc))

            # 3. Columna de Habilidades Globales (Activas vs Cooldown)
            sidebar_x_rel = 0.527
            sidebar_y_positions = [0.292, 0.370, 0.449, 0.528, 0.608, 0.687, 0.766, 0.845, 0.924]
            ready_skills = self.vision.detect_ready_global_skills(frame_rgb)
            for idx, y_pos in enumerate(sidebar_y_positions):
                cx, cy = int(sidebar_x_rel * w), int(y_pos * h)
                is_ready = any(abs(gx - sidebar_x_rel) < 0.015 and abs(gy - y_pos) < 0.015 for gx, gy in ready_skills)
                s_color = (0, 255, 0) if is_ready else (255, 40, 40)
                s_label = f"H{idx+1}: LISTA" if is_ready else f"H{idx+1}: CD/LOCK"
                cv2.rectangle(canvas, (cx - int(15 * sc), cy - int(15 * sc)),
                              (cx + int(15 * sc), cy + int(15 * sc)), s_color, int(2 * sc))
                cv2.putText(canvas, s_label, (cx + int(20 * sc), cy + int(5 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42 * sc, s_color, int(1.5 * sc))


            # 4. Botones Azules de Subida de Nivel (+NV Comprables)
            buyable_buttons = self.vision.detect_buyable_upgrade_buttons(frame_rgb)
            for b_x, b_y in buyable_buttons:
                px, py = int(b_x * w), int(b_y * h)
                cv2.rectangle(canvas, (px - int(45 * sc), py - int(18 * sc)),
                              (px + int(45 * sc), py + int(18 * sc)), (0, 255, 255), int(2 * sc))
                cv2.putText(canvas, "+NV DISPONIBLE", (px - int(45 * sc), py - int(22 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48 * sc, (0, 255, 255), int(2 * sc))

            # 5. Habilidades de Héroes (✔ Compradas vs ★ Comprables vs 🔒 Bloqueadas)
            hero_skills_det = self.vision.detect_hero_skills_detailed(frame_rgb)
            buyable_hero_skills_count = 0
            for item in hero_skills_det:
                s_x, s_y = item["coords"]
                status = item["status"]
                hx, hy = int(s_x * w), int(s_y * h)
                if status == "COMPRABLE":
                    buyable_hero_skills_count += 1
                    box_color = (255, 215, 0)  # Amarillo dorado
                    label = "COMPRABLE"
                elif status == "COMPRADA":
                    box_color = (0, 255, 0)  # Verde
                    label = "COMPRADA"
                else:
                    box_color = (200, 50, 50)  # Rojo
                    label = "BLOQUEADA"

                cv2.rectangle(canvas, (hx - int(14 * sc), hy - int(14 * sc)),
                              (hx + int(14 * sc), hy + int(14 * sc)), box_color, int(2 * sc))
                cv2.putText(canvas, label, (hx - int(25 * sc), hy - int(18 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38 * sc, box_color, 1)

            # 6. Reconocimiento OCR de Zona
            zone_detected = self.vision.detect_zone_number_from_screen(frame_rgb)
            z_y1, z_x1 = int(0.115 * h), int(0.58 * w)
            z_y2, z_x2 = int(0.175 * h), int(0.82 * w)
            cv2.rectangle(canvas, (z_x1, z_y1), (z_x2, z_y2), (255, 0, 255), int(2 * sc))
            cv2.putText(canvas, f"OCR ZONA: {zone_detected}", (z_x1, z_y1 - int(8 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (255, 0, 255), int(2 * sc))

            # 7. Zonas de Interacción (Puntos de Clic Estáticos)
            def draw_crosshair(canvas, rx, ry, label, color):
                cx, cy = int(rx * w), int(ry * h)
                cv2.drawMarker(canvas, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=int(12 * sc), thickness=int(2 * sc))
                cv2.putText(canvas, label, (cx + int(8 * sc), cy - int(8 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * sc, color, 1)

            # Zona Monstruo
            draw_crosshair(canvas, 0.75, 0.55, "Click Ataque", (0, 0, 255))
            # Flechas de Zona
            draw_crosshair(canvas, 0.800, 0.055, "Siguiente", (255, 128, 0))
            draw_crosshair(canvas, 0.585, 0.055, "Anterior", (255, 128, 0))
            # Ancla de Scroll
            draw_crosshair(canvas, 0.120, 0.500, "Scroll Area", (128, 128, 255))
            # Zona Segura (Ocultar Tooltips)
            draw_crosshair(canvas, 0.900, 0.100, "Safe Zone (Raton)", (255, 255, 255))

            # 8. Panel HUD Superior
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 0.5:
                self.fps_display = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()

            bought_count = self.vision.count_bought_hero_skills(frame_rgb)

            hud_h = int(46 * sc)
            cv2.rectangle(canvas, (0, 0), (w, hud_h), (20, 20, 20), -1)
            hud_text = (f"AutoRL Vision (Resolucion: {w}x{h}) | FPS: {self.fps_display:.1f} | "
                        f"Zona: {zone_detected} | +NV: {len(buyable_buttons)} | "
                        f"Hab. Heroe: (✔ {bought_count} Compradas | ★ {buyable_hero_skills_count} Comprables) | "
                        f"Hab. H: {len(ready_skills)} | Salir: [Q / ESC]")
            cv2.putText(canvas, hud_text, (int(15 * sc), int(30 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48 * sc, (255, 255, 0), int(2 * sc))

            # Redimensionar al tamaño actual de la ventana de Tkinter
            win_w = max(640, self.root.winfo_width())
            win_h = max(360, self.root.winfo_height())
            resized = cv2.resize(canvas, (win_w, win_h), interpolation=cv2.INTER_AREA)

            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_image.imgtk = img_tk
            self.lbl_image.configure(image=img_tk)

        self.root.after(16, self.update_frame)

    def close(self):
        self.running = False
        self.cap.close()
        self.root.destroy()
        print("\n✔ [DEBUGGER FINALIZADO] Ventana de depuración cerrada correctamente.")


def main():
    print("=" * 70)
    print("🔍 [AUTORL VISION DEBUGGER] Iniciando auditoría visual en tiempo real...")
    print("💡 Pulsa la tecla 'q' o 'ESC' sobre la ventana para salir.")
    print("=" * 70)
    root = tk.Tk()
    app = VisionDebuggerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
