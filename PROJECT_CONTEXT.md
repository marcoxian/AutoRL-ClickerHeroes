# Contexto de Proyecto: AutoRL Vision Computer (Clicker Heroes)

Este archivo sirve como la fuente única de verdad del estado actual de la IA, el entorno, el modelado de recompensas y la lógica de control. Debe ser utilizado para inyectar el contexto principal al reabrir nuevas sesiones de desarrollo y evitar degradación de memoria.

## 1. Resumen y Stack Tecnológico
**Propósito:** Agente de Aprendizaje por Refuerzo (PPO) que juega a Clicker Heroes utilizando visión artificial (OpenCV) para extraer el estado directamente de los píxeles y tomar decisiones a nivel humano sin acceder a la memoria RAM del juego.

**Librerías principales:**
- **Entorno:** `Gymnasium` (Implementación de entorno a medida).
- **Algoritmo RL:** `Stable-Baselines3` (PPO) con `CustomCNNFeatureExtractor` (Red Neuronal Convolucional que procesa matrices reducidas a 84x84).
- **Visión:** `OpenCV` (filtrado de color HSV, absdiff, máscaras, Template Matching) y `EasyOCR` (para lectura hiper-robusta de números de zona).
- **Captura de Pantalla:** `mss` (Módulo super-rápido basado en GDI32 de Windows).
- **Control Periféricos:** `PyWin32` (obtención de dimensiones de ventana) y `pydirectinput` (uso de DirectX `mouse_event` para evadir los sistemas anti-bots).
- **Optimización:** `Optuna` (búsqueda Bayesiana automatizada de hiperparámetros).

## 2. Espacio de Acciones (Discrete 9)
El agente PPO cuenta con una salida categórica de 9 acciones. Las coordenadas están mapeadas de forma normalizada `[0.0, 1.0]` en el módulo `input_controller.py`:

| ID | Acción | Descripción | Coordenadas Relativas (X, Y) |
|---|---|---|---|
| `0` | **ACTION_MONSTER_CLICK** | Ráfaga de clics sobre la isla de combate para golpear monstruos. | `(0.73, 0.55)` |
| `1` | **ACTION_SMART_UPGRADE** | Busca y pulsa botones azules "+NV" (Mejoras). | Variable / Base: `(0.095, 0.46)` |
| `2` | **ACTION_SMART_HERO_SKILLS**| Busca iconos pequeños cuadrados para comprar habilidades pasivas de héroes. | Variable / Base: `(0.22, 0.50)` |
| `3` | **ACTION_GLOBAL_SKILLS_BAR**| Clica en la barra vertical (Ej. Clickstorm). CD interno de 30s. | `(0.521, 0.292)` |
| `4` | **ACTION_PREV_ZONE_FARM** | Flecha Izquierda. Retroceder a zona de farmeo. | `(0.70, 0.055)` |
| `5` | **ACTION_NEXT_ZONE_PROGRESS**| Flecha Derecha. Progresar o reintentar el Boss. | `(0.795, 0.055)` |
| `6` | **ACTION_SCROLL_DOWN** | Simula girar la rueda del ratón hacia abajo. | `(0.25, 0.65)` -15 clics |
| `7` | **ACTION_SCROLL_UP** | Simula girar la rueda del ratón hacia arriba. | `(0.25, 0.65)` +15 clics |
| `8` | **ACTION_NOOP** | Dejar inactivo (útil para que DPS pasivo gane oro). | N/A |

**Modo "MAX" y Macro de Reset:** 
Actualmente, el juego no usa una acción de pulsar la letra "Z/Shift" para poner las compras en MAX.
Sin embargo, sí existe una Macro automatizada de Reinicio en `input_controller.py` > `reset(hard_reset=True)` que realiza la ascensión/reinicio manual: 
1. Clic Ajustes `(0.965, 0.05)`
2. Clic Reiniciar `(0.44, 0.725)`
3. Confirmar "Sí" `(0.41, 0.665)`

## 3. Lógica de Recompensa (Reward Shaping)
Totalmente enfocada en evitar *Reward Hacking* y fomentar la progresión pura.
- **Progresión (+Zona/Hito Boss):**
  - Avanzar a nivel récord de farmeo: **+20.0 pts**.
  - Derrotar un Jefe (Niveles múltiplos de 5): **+200.0 pts**.
  *(Nota: Se eliminó el retardo de confirmación de 7 segundos; ahora PPO recibe los 200 puntos en el instante exacto en que EasyOCR lee el cambio de pantalla).*
- **Desarrollo:**
  - Mejora Héroe `+NV` (Confirmada vía diff de píxeles): **+1.0 a +5.5 pts** (escala con la profundidad del scroll `scroll_depth * 1.5` para forzar a la IA a scrollear hacia los héroes más potentes).
  - Comprar Habilidad Pasiva (Tick verde `✔`): **+10.0 pts** multiplicable.
  - Usar Habilidad Global H: **+5.0 pts** (Máximo 1 vez cada 30 segundos, si hay listas).
- **Eliminaciones estrictas de puntos de farmeo:**
  - `ΔHP` de la barra roja: **0 puntos**. La IA estaba explotando golpear Jefes infinitamente sin matarlos para ganar puntos, ahora está eliminado.
  - Compras ciegamente ignoradas: **0 puntos**. Si hace clic en comprar pero no tiene oro (la imagen local con OpenCV no sufre cambios tras 0.25s), se detecta clic falso.
- **Penalizaciones:**
  - Scroll inútil (subir para nada o rascar el fondo): **-0.5 a -2.0 pts**.

## 4. Candados y Lógica Anti-Exploits
Situados en la jerarquía superior de `clicker_heroes_env.py` > `step()`. Las acciones se evalúan y sobrescriben a `NOOP` antes de enviar la orden física al ratón si rompen el cerrojo:

1. **Doble Cerrojo en Retroceso (Acción 4):**
   - Para pulsar Izquierda, `SmartVision` DEBE estar detectando el Cronómetro circular de Boss activo.
   - **Y** deben haber transcurrido obligatoriamente `>= 35.0 segundos` en el combate. (Huir antes da -0.5 de castigo y se bloquea el clic).
2. **Candado de Avance (`advance_locked`) en (Acción 5):**
   - Una vez la IA huye legítimamente a una zona de farmeo, el estado cambia a `advance_locked = True`.
   - Si intenta volver al Boss sin estar preparada, la Acción 5 es interceptada y castigada.
   - **Desbloqueo:** Solo se abren las puertas al Boss si compra al menos `5 mejoras de héroe` (`self.upgrades_purchased_since_retreat >= 5`).
3. **Reward Capping (Corte de Grifo):**
   - El atributo `self.max_zone_steps = 300` (aprox. 60 segundos). Si la IA pasa más de ese tiempo en el mismo nivel sin avanzar de zona, la recompensa por hacer upgrades o scroll se vuelve **0.0**, cortando la dopamina y forzándola a reintentar el Boss sí o sí.

## 5. Estado de Visión (OpenCV) y Optuna
- **Extracción de Texto (Zona):** `EasyOCR` escala a `x2`. Crop ampliado a X relativa `[0.55, 0.92]` para acomodar números altos (Ej: 115). Usa `re.findall(r'\d+', text)` extrayendo el último número.
- **Detección de Monstruo Muerto:** Usa `pytesseract` y filtrado para buscar las palabras clave "Muerto" o "Dead".
- **Plantillas (Template Matching):** Implementado `cv2.matchTemplate` con threshold ajustado para localizar el color de los iconos cuadrados de habilidades (`.tempmediaStorage/debug_verified_h1_green.png`, etc.).
- **Optuna Bayesiano:** Ajustes descubiertos por AutoRL para PPO actualmente cargados en `train_clicker_heroes_autorl.py`:
  - `Learning Rate`: 0.000354
  - `Batch Size`: 64
  - `Filters` en CNN Extractor: 32
