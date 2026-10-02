import cv2
import numpy as np
from src.utils.smart_vision import SmartVisionController
sv = SmartVisionController()

# Cargar la captura actual del usuario
img = cv2.imread(r'C:\Users\marco\.gemini\antigravity-ide\brain\17f0ffef-b2cd-471d-8ad6-582303ff5e88\.user_uploaded\media_1787512013274.png')
if img is None:
    print('ERROR: No se pudo cargar la imagen')
    exit()
frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = frame_rgb.shape[:2]
print(f'Frame: {w}x{h}')

# 1. Detectar botones +NV primero
btns = sv.detect_buyable_upgrade_buttons(frame_rgb)
print(f'\nBotones +NV detectados: {btns}')

# 2. Detectar habilidades detalladas
skills = sv.detect_hero_skills_detailed(frame_rgb)
for s in skills:
    x_rel, y_rel = s['coords']
    status = s['status']
    
    # Analizar el contenido real del slot
    y1, y2 = int((y_rel - 0.016) * h), int((y_rel + 0.016) * h)
    x1, x2 = int((x_rel - 0.012) * w), int((x_rel + 0.012) * w)
    slot = frame_rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(slot, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(slot, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    mask_green = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
    green_px = cv2.countNonZero(mask_green)
    
    mask_padlock = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 45, 170]))
    padlock_px = cv2.countNonZero(mask_padlock)
    
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))
    edge_px = cv2.countNonZero(edges)
    std_val = float(np.std(gray))
    
    mean_rgb = frame_rgb[y1:y2, x1:x2].mean(axis=(0,1))
    
    print(f'  Slot ({x_rel:.3f}, {y_rel:.3f}) -> {status}')
    print(f'    green_px={green_px}, padlock_px={padlock_px}, mean_S={mean_s:.1f}, mean_V={mean_v:.1f}')
    print(f'    edges={edge_px}, std={std_val:.1f}, mean_RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})')

bought = sum(1 for s in skills if s["status"] == "COMPRADA")
blocked = sum(1 for s in skills if s["status"] == "BLOQUEADA")
buyable = sum(1 for s in skills if s["status"] == "COMPRABLE")
print(f'\nTotal COMPRADA: {bought}')
print(f'Total BLOQUEADA: {blocked}')
print(f'Total COMPRABLE: {buyable}')
