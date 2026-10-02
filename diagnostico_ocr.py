import cv2
import easyocr
import re
import numpy as np
from src.utils.screen_capture import ScreenCapture

def test():
    print("Capturando pantalla...")
    cap = ScreenCapture()
    img = cap.capture_raw_rgb()
    if img is None or img.size == 0:
        print("Error: Imagen vacía o negra.")
        return

    h, w = img.shape[:2]
    # ESTAS SON LAS COORDENADAS NUEVAS ARREGLADAS (0.55 a 0.92)
    crop = img[int(0.115 * h):int(0.175 * h), int(0.55 * w):int(0.92 * w)]
    
    cv2.imwrite("debug_crop_user.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"Recorte guardado en debug_crop_user.png. Mean pixel RGB: {img.mean(axis=(0,1))}")

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
    
    # Prueba 1: Gris normal
    res1 = reader.readtext(gray, detail=0)
    print("OCR (Gris Normal):", " ".join(res1))
    
    # Prueba 2: Gris x2 (El que pusimos en el código real)
    res2 = reader.readtext(gray_2x, detail=0)
    print("OCR (Gris x2):", " ".join(res2))
    
    # Prueba 3: Umbral adaptativo
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("debug_thresh_user.png", thresh)
    res3 = reader.readtext(thresh, detail=0)
    print("OCR (Umbral):", " ".join(res3))

if __name__ == "__main__":
    test()
