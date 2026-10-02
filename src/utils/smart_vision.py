"""
Módulo de Visión Inteligente (Smart Vision) con OpenCV para Clicker Heroes / Idle Slayer.
Detecta automáticamente mediante visión por computador:
1. Botones de mejoría de héroes disponibles (color azul brillante).
2. Pequeñas habilidades de héroes y barra de habilidades globales.
3. Estado de Boss / Jefe (reloj de tiempo, calavera de vida y velocidad de daño).
4. Detección de atasco (Boss sin suficiente DPS) para retroceder y farmear oro.
"""

from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np


class SmartVisionController:
    """
    Controlador de Visión Computacional Avanzada.
    En lugar de coordenadas ciegas, analiza el fotograma de pantalla en tiempo real
    y detecta exactamente qué elementos son interactivos y utilizables.
    """

    def __init__(self):
        # Rango HSV para botones azules brillantes de mejoría (+ NV / CONT)
        self.LOWER_BLUE = np.array([90, 120, 140], dtype=np.uint8)
        self.UPPER_BLUE = np.array([125, 255, 255], dtype=np.uint8)

        # Rango HSV para la barra de vida de jefes (roja/verde)
        self.LOWER_HP_RED = np.array([0, 150, 120], dtype=np.uint8)
        self.UPPER_HP_RED = np.array([10, 255, 255], dtype=np.uint8)

        # Rango HSV para botones azules brillantes comprables (+ NV / CONT)
        # Excluye botones desaturados/oscuros no comprables (azul flojo o gris)
        self.LOWER_BLUE = np.array([90, 140, 160], dtype=np.uint8)
        self.UPPER_BLUE = np.array([125, 255, 255], dtype=np.uint8)

        # Rango HSV para icono de calavera / indicador de Boss en barra superior (dorado/rojo)
        self.LOWER_BOSS_GOLD = np.array([15, 150, 150], dtype=np.uint8)
        self.UPPER_BOSS_GOLD = np.array([35, 255, 255], dtype=np.uint8)

        self.last_hp = 1.0
        self.stuck_counter = 0



    def detect_buyable_upgrade_buttons(self, frame_rgb: np.ndarray) -> List[Tuple[float, float]]:

        """
        Analiza el panel izquierdo de héroes y devuelve las coordenadas relativas (x_rel, y_rel)
        de TODOS los botones azules brillantes (+NV) que están COMPRABLES en este instante.
        Restringido a la columna de botones de héroe (x=0.03 a 0.16, y=0.38 a 0.98) para ignorar pestañas superiores.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return []

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]

        # Enfocar estrictamente en la columna de botones +NV (x=0.03 a 0.16, y=0.38 a 0.98)
        y_min, y_max = int(h * 0.38), int(h * 0.98)
        x_min, x_max = int(w * 0.03), int(w * 0.16)
        hero_btn_col = frame_rgb[y_min:y_max, x_min:x_max]
        if hero_btn_col.size == 0:
            return []

        hsv = cv2.cvtColor(hero_btn_col, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buyable_centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Ignorar botones cortados por los bordes superior/inferior del scroll (altura relativa < 0.065)
                if float(bh) / h > 0.065:
                    center_x_rel = float(x_min + x + bw / 2.0) / w
                    center_y_rel = float(y_min + y + bh / 2.0) / h
                    buyable_centers.append((center_x_rel, center_y_rel))

        buyable_centers.sort(key=lambda item: item[1])
        return buyable_centers

    def detect_hero_skills_detailed(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analiza con OpenCV todos los slots de habilidades visibles de los héroes y clasifica su estado:
        - "COMPRADA": Ya comprada (tiene la marca de verificación verde ✔).
        - "COMPRABLE": Desbloqueada con icono a todo color, lista para comprar (sin tick verde).
        - "BLOQUEADA": Silueta oscura/gris, requiere subir más nivel de héroe.
        Ignora automáticamente los huecos vacíos de pergamino que aún no tienen habilidad.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return []

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]

        # Posiciones Y base de las 4 tarjetas de héroe visibles por defecto (sin scroll)
        hero_rows_y = [0.455, 0.595, 0.735, 0.875]
        ref_y = None

        # 1. Intentar encontrar botones +NV azules (comprables)
        buyable_btns = self.detect_buyable_upgrade_buttons(frame_rgb)
        if buyable_btns:
            ref_y = buyable_btns[0][1]
        else:
            # 2. Si no hay botones azules, buscar los grises (bloqueados) para usarlos como ancla
            h, w = frame_rgb.shape[:2]
            y_min, y_max = int(h * 0.38), int(h * 0.98)
            x_min, x_max = int(w * 0.03), int(w * 0.16)
            hero_btn_col = frame_rgb[y_min:y_max, x_min:x_max]
            
            if hero_btn_col.size > 0:
                hsv = cv2.cvtColor(hero_btn_col, cv2.COLOR_RGB2HSV)
                # Los botones grises tienen saturación baja y valor medio/bajo
                mask_grey = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 60, 160]))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
                mask_cleaned = cv2.morphologyEx(mask_grey, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) > 800:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        # Ignorar botones cortados por el borde del scroll
                        if float(bh) / h > 0.065:
                            ref_y = float(y_min + y + bh / 2.0) / h
                            break
        
        # 3. Si hemos encontrado un ancla (azul o gris), extrapolamos todas las filas perfectamente alineadas
        if ref_y is not None:
            inferred_rows = []
            
            # Hacia arriba
            y = ref_y
            while y > 0.40:
                inferred_rows.append(y)
                y -= 0.14
            
            # Hacia abajo
            y = ref_y + 0.14
            while y < 0.95:
                inferred_rows.append(y)
                y += 0.14
                
            inferred_rows.sort()
            hero_rows_y = inferred_rows

        # Las habilidades de los héroes están separadas horizontalmente por ~0.030
        # Expandido a 7 slots (algunos héroes como Cira tienen 7 habilidades)
        skill_x_offsets = [0.168, 0.198, 0.228, 0.258, 0.288, 0.318, 0.348]
        results = []

        for row_y in hero_rows_y:
            skill_y = row_y + 0.038
            for sx in skill_x_offsets:
                # Región del cuadro de la habilidad
                y1, y2 = int((skill_y - 0.016) * h), int((skill_y + 0.016) * h)
                x1, x2 = int((sx - 0.012) * w), int((sx + 0.012) * w)

                if y1 < 0 or y2 > h or x1 < 0 or x2 > w:
                    continue

                slot_roi = frame_rgb[y1:y2, x1:x2]
                if slot_roi.size == 0:
                    continue

                # 0. COMPROBACIÓN: ¿Existe realmente una caja de habilidad o es pergamino vacío?
                gray = cv2.cvtColor(slot_roi, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 30, 100)
                edge_pixels = cv2.countNonZero(edges)
                std_val = float(np.std(gray))

                # Si es pergamino liso beige uniforme sin bordes ni marco, no hay habilidad
                if edge_pixels < 18 and std_val < 18.0:
                    continue

                hsv = cv2.cvtColor(slot_roi, cv2.COLOR_RGB2HSV)
                mean_v = float(np.mean(hsv[:, :, 2]))

                # 1. Comprobar si tiene el Tick Verde ✔ (Comprada)
                # El tick verde siempre aparece en la esquina superior izquierda. 
                h_roi, w_roi = hsv.shape[:2]
                hsv_top_left = hsv[0:int(h_roi*0.4), 0:int(w_roi*0.4)]
                
                # Usamos una máscara equilibrada: [50, 150, 150] evita hojas apagadas pero detecta el tick brillante
                mask_green = cv2.inRange(hsv_top_left, np.array([50, 150, 150]), np.array([85, 255, 255]))
                green_pixels = cv2.countNonZero(mask_green)

                if green_pixels > 10:
                    results.append({"coords": (sx, skill_y), "status": "COMPRADA"})
                    continue

                # 2. Comprobar si tiene Marco Dorado Brillante (Comprable con oro)
                mask_gold_frame = cv2.inRange(hsv, np.array([10, 120, 150]), np.array([40, 255, 255]))
                gold_pixels = cv2.countNonZero(mask_gold_frame)

                # Eliminamos la restricción de `mean_v > 90` porque algunos iconos son oscuros de por sí (ej. escudo de León)
                # Si tiene suficientes píxeles dorados puros en el marco, es comprable 100%.
                if gold_pixels > 25:
                    results.append({"coords": (sx, skill_y), "status": "COMPRABLE"})
                    continue

                # 3. Si no está comprada y no es comprable, verificar si es un candado gris (Bloqueada)
                # 3. Si no está comprada y no es comprable, verificar si es un candado gris (Bloqueada)
                # 3. Revisar el fondo de la casilla (Descubrimiento del usuario)
                # El juego usa #0A0A0A (V~10) para el fondo si está bloqueada/inasequible.
                # Usa #333333 (V~51) para el fondo si es comprable pero no es la siguiente.
                # Extraemos el color más oscuro de la imagen (el percentil 10) que corresponde al fondo de la caja.
                v_channel = hsv[:, :, 2]
                p10 = np.percentile(v_channel, 10)
                
                if p10 < 25:
                    # Fondo negro puro #0A0A0A
                    results.append({"coords": (sx, skill_y), "status": "BLOQUEADA"})
                elif p10 < 80:
                    # Fondo gris oscuro #333333
                    # Es comprable, así que la devolvemos como COMPRABLE para que la IA la pinche.
                    results.append({"coords": (sx, skill_y), "status": "COMPRABLE"})

        return results

    def count_bought_hero_skills(self, frame_rgb: np.ndarray) -> int:
        """Cuenta con precisión el número de habilidades de héroe con status COMPRADA usando detección por slots exactos.
        NO busca contornos verdes aleatorios en el panel (que capturarían textos de oro, bordes de botones, etc.)."""
        all_skills = self.detect_hero_skills_detailed(frame_rgb)
        return sum(1 for s in all_skills if s["status"] == "COMPRADA")

    def detect_buyable_hero_skills(self, frame_rgb: np.ndarray) -> List[Tuple[float, float]]:
        """Devuelve únicamente las coordenadas de habilidades de héroe que están COMPRABLES."""
        all_skills = self.detect_hero_skills_detailed(frame_rgb)
        return [s["coords"] for s in all_skills if s["status"] == "COMPRABLE"]

    def detect_hero_skill_icons(self, frame_rgb: np.ndarray) -> List[Tuple[float, float]]:
        """Alias para compatibilidad."""
        return self.detect_buyable_hero_skills(frame_rgb)












    def detect_boss_timer(self, frame_rgb: np.ndarray) -> bool:
        """
        Detecta mediante OpenCV si el indicador de tiempo del Boss (reloj + texto "XX.X s")
        está activo en la pantalla en la región exacta del cuadro bajo la calavera.
        ROI Calibrado: Y rel [0.215, 0.265], X rel [0.705, 0.795].
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return False

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]
        # Región milimétrica justo debajo del contador de monstruos
        y1, x1, y2, x2 = int(0.215 * h), int(0.705 * w), int(0.265 * h), int(0.795 * w)
        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # 1. Filtrar el icono circular del reloj (rojo/azul del temporizador)
        lower_red1 = np.array([0, 100, 120], dtype=np.uint8)
        upper_red1 = np.array([15, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([165, 100, 120], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # 2. Filtrar el texto blanco brillante del temporizador ("XX.X s")
        lower_white = np.array([0, 0, 220], dtype=np.uint8)
        upper_white = np.array([180, 30, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        active_timer_pixels = cv2.countNonZero(mask_red) + cv2.countNonZero(mask_white)
        # Requiere al menos 90 píxeles activos combinados en la zona estricta del reloj
        return active_timer_pixels > 90

    def analyze_boss_and_dps_status(self, frame_rgb: np.ndarray, curr_hp: float) -> Dict[str, Any]:


        """
        Analiza si el jugador está atrapado en un Boss sin suficiente DPS.
        Si la vida del Boss no baja adecuadamente o el tiempo se agota,
        devuelve recomendación de retroceder a zona de farmeo de oro (Farm Mode).
        """
        hp_delta = self.last_hp - curr_hp

        # Si estamos en zona de Boss y la vida casi no baja
        if curr_hp > 0.3 and hp_delta < 0.002:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        self.last_hp = curr_hp

        is_stuck_on_boss = self.stuck_counter > 15  # Atrapado por más de 15 pasos sin bajar HP

        return {
            "curr_hp": curr_hp,
            "hp_delta": hp_delta,
            "stuck_counter": self.stuck_counter,
            "recommend_farm_mode": is_stuck_on_boss
        }

    def detect_zone_number_from_screen(self, frame_rgb: np.ndarray) -> Optional[int]:
        """
        Lee directamente de la pantalla mediante OCR y OpenCV el número exacto del nivel actual (e.g. 'Desierto Nv. 17' -> 17).
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return None

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]
        # ROI de la cabecera de nivel centrada (ampliado a la derecha para niveles altos): Y rel [0.115, 0.175], X rel [0.55, 0.92]
        crop = frame_rgb[int(0.115 * h):int(0.175 * h), int(0.55 * w):int(0.92 * w)]
        if crop.size == 0:
            return None

        try:
            if not hasattr(self, "_ocr_reader"):
                import easyocr
                self._ocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)

            # Escalar x2 mejora drásticamente la lectura de EasyOCR para fuentes pequeñas
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            results = self._ocr_reader.readtext(gray_2x, detail=0)
            text = " ".join(results)
            import re
            
            # Buscar cualquier número en el texto. En esta zona de la pantalla, los únicos números son los del nivel.
            matches = re.findall(r'\d+', text)
            if matches:
                # Tomar el PRIMER número encontrado, ya que la zona está en la parte superior izquierda.
                # Tomar el último provocaba que a veces leyera la vida del monstruo u otros números por debajo.
                val = int(matches[0]) 
                if 1 <= val <= 9999:
                    return val
        except Exception:
            pass



        return None

    def detect_hero_souls_to_ascend(self, frame_rgb: np.ndarray) -> Optional[int]:
        """
        Lee el texto del panel superior derecho de los héroes (en la cabecera marrón)
        para saber cuántas 'Almas de héroe' se obtendrán al ascender (ej: '+7 tras ascensión').
        Devuelve el número entero si lo encuentra, o None en caso contrario.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return None

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]
        
        # ROI ajustado a la caja marrón superior derecha del panel de héroes (según la captura del usuario)
        y1, y2 = int(0.23 * h), int(0.32 * h)
        x1, x2 = int(0.32 * w), int(0.44 * w)
        crop = frame_rgb[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None

        try:
            if not hasattr(self, "_ocr_reader"):
                import easyocr
                self._ocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)

            # Escalar x2 mejora drásticamente la lectura de EasyOCR
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            results = self._ocr_reader.readtext(gray_2x, detail=0)
            text = " ".join(results).lower()
            
            import re
            # Buscar el patrón de "+\d+ tras ascensión", tolerando errores del OCR
            match = re.search(r'\+(\d+)\s*(tras|ascens)', text)
            if match:
                return int(match.group(1))
        except Exception:
            pass

        return None

    def is_auto_progression_off(self, frame_rgb: np.ndarray) -> bool:
        """
        Detecta si el botón de Progresión Automática (Bota alada en la parte derecha) está APAGADO.
        Cuando está apagado, tiene un símbolo de prohibición rojo muy llamativo encima.
        ROI: Y rel [0.20, 0.35], X rel [0.83, 0.95].
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return False

        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        h, w = frame_rgb.shape[:2]
        y1, x1, y2, x2 = int(0.20 * h), int(0.83 * w), int(0.35 * h), int(0.95 * w)
        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Filtro para el símbolo de prohibición rojo (rojo puro/fuerte)
        lower_red1 = np.array([0, 150, 100], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 150, 100], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = cv2.countNonZero(mask_red)
        # Si vemos bastantes píxeles rojos en esa zona exacta, la bota está apagada
        return red_pixels > 30

    def detect_popup_close_button(self, frame_rgb: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detecta si hay un popup en la pantalla buscando la 'X' roja de cierre algorítmicamente.
        Busca un círculo rojo brillante que contenga píxeles blancos en forma de X.
        Es invulnerable a los cambios de tamaño de ventana o resolución (Scale-Invariant).
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return None
            
        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))
            
        h_frame, w_frame = frame_rgb.shape[:2]
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        # Filtro estricto para el rojo brillante/naranja del botón X (ampliado a naranjas)
        lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
        upper_red1 = np.array([30, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            # El botón de cerrar no es minúsculo ni gigante (entre 0.05% y 2% de la pantalla)
            if 150 < area < (h_frame * w_frame * 0.02):
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / max(1, h)
                
                # Debe ser más o menos cuadrado (la base de la X es un círculo)
                if 0.7 < aspect_ratio < 1.3:
                    # Las X de popup suelen estar en el tercio superior o medio de la pantalla
                    y_rel = y / h_frame
                    x_rel = x / w_frame
                    if y_rel < 0.6 and x_rel > 0.3:
                        
                        # Extraer la región interior para buscar la cruz blanca
                        roi_hsv = hsv[y:y+h, x:x+w]
                        lower_white = np.array([0, 0, 180], dtype=np.uint8)
                        upper_white = np.array([180, 80, 255], dtype=np.uint8)
                        mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)
                        
                        white_pixels = cv2.countNonZero(mask_white)
                        
                        # La X blanca ocupa entre el 5% y el 40% del área del botón rojo
                        if (w * h * 0.05) < white_pixels < (w * h * 0.45):
                            # Centro del botón en coordenadas relativas
                            center_x_rel = (x + w / 2.0) / w_frame
                            center_y_rel = (y + h / 2.0) / h_frame
                            
                            # IGNORAR FALSO POSITIVO: Reloj de Jefe (Boss Timer)
                            # El reloj de jefe es un círculo rojo con números blancos situado en [X: 0.70-0.81, Y: 0.20-0.28]
                            if 0.70 < center_x_rel < 0.81 and 0.20 < center_y_rel < 0.28:
                                continue # No es un popup, es el temporizador del jefe
                            
                            # IGNORAR FALSO POSITIVO: Barra de pestañas del panel izquierdo
                            # Las pestañas (espada, estrella, gráfico, etc.) están en [X: 0.0-0.30, Y: 0.0-0.08]
                            if center_x_rel < 0.30 and center_y_rel < 0.08:
                                continue # No es un popup, es un icono de pestaña
                            
                            # IGNORAR FALSO POSITIVO: Panel izquierdo completo (mercenarios, logros, etc.)
                            # Los popups reales siempre aparecen en la zona central/derecha (X > 0.30)
                            if center_x_rel < 0.30:
                                continue # No es un popup, es un elemento del panel lateral
                                
                            return (center_x_rel, center_y_rel)
        return None

    def is_on_wrong_tab(self, frame_rgb: np.ndarray) -> bool:
        """
        Detecta si el panel izquierdo NO muestra la pestaña de Héroes.
        Comprueba si hay botones azules '+NV' o texto de héroes en la zona habitual.
        Si no hay NADA azul/verde en el panel de héroes, probablemente estamos en otra pestaña
        (Mercenarios, Logros, etc.).
        ROI: Panel izquierdo Y [0.30, 0.95], X [0.0, 0.15]
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return False
            
        if frame_rgb.ndim == 3 and frame_rgb.shape[0] in (1, 3, 4):
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))
            
        h, w = frame_rgb.shape[:2]
        # Zona donde deberían estar los botones azules "+NV" de héroes
        y1, x1, y2, x2 = int(0.30 * h), int(0.0 * w), int(0.95 * h), int(0.15 * w)
        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return False
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Buscar azul brillante (botones +NV comprables)
        lower_blue = np.array([90, 120, 140], dtype=np.uint8)
        upper_blue = np.array([125, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = cv2.countNonZero(mask_blue)
        
        # Buscar verde brillante (botones +NV comprables verdes de bajo nivel)
        lower_green = np.array([35, 100, 140], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask_green)
        
        total_interactive_pixels = blue_pixels + green_pixels
        
        # Si hay muy pocos píxeles azules/verdes en la zona de botones, estamos en la pestaña incorrecta
        # Un solo botón +NV tiene al menos ~200 píxeles azules
        return total_interactive_pixels < 50


