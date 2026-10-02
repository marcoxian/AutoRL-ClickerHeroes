# 📚 Base de Datos del Proyecto: AutoRL Clicker Heroes

Este documento es el registro oficial de todo el desarrollo, los errores que cometimos, las trampas que nos hizo la Inteligencia Artificial (Reward Hacking) y las soluciones definitivas que implementamos para lograr un agente PPO robusto.

---

## 🐛 1. El Problema del "Reward Hacking" (Farmeo Infinito)
**El Error:**
La IA descubrió que los primeros héroes de la lista (Cira, Foresto) eran muy baratos. Como le dábamos `+5.0 puntos` por comprar cualquier mejora, la IA decidió que era más rentable quedarse arriba del todo comprando las mejoras más baratas hasta el infinito. Se negó a hacer scroll hacia abajo porque alejarse de esos héroes baratos significaba dejar de ganar puntos gratis.

**La Solución:**
1. **Nerfeo de recompensas planas:** Redujimos el premio base por comprar una mejora de `+5.0` a `+1.5`.
2. **Sistema de Profundidad de Scroll:** Creamos la variable `scroll_depth`. Ahora, los puntos que la IA gana al comprar un héroe se multiplican por su profundidad. (Ej: Comprar arriba da `+1.0`, pero comprar tras hacer scroll 3 veces da `+5.5`). Esto obligó a la IA a entender que los héroes del fondo son más valiosos.

---

## 🐛 2. Alucinación de OCR (Retrocesos Injustificados)
**El Error:**
El código OCR de Tesseract a veces leía mal el nivel en el que estábamos (ej. leía 22 en lugar de 23). Como el juego te envía al nivel anterior cuando pierdes contra un Jefe, la IA interpretaba este fallo visual como que había perdido un combate y se bloqueaba en modo "Farmeo".

**La Solución:**
**Candado Matemático (Módulo 5):** En *Clicker Heroes*, los jefes solo existen en los niveles múltiplos de 5 (5, 10, 15...). Implementamos la regla `current_zone % 5 == 0`. Si la zona no es múltiplo de 5, el código ignora cualquier fallo de lectura porque es matemáticamente imposible retroceder.

---

## 🐛 3. Alucinación de Visión (Falsos Jefes)
**El Error:**
Incluso en zonas normales (ej. Zona 6), la IA a veces creía que estaba luchando contra un jefe y huía a la zona 5.
Esto pasaba porque la cámara buscaba píxeles **rojos y blancos** para detectar el reloj de tiempo del jefe. Cuando saltaba un texto de "Daño Crítico" (rojo y blanco) o volaba un pájaro rubí por esa zona, la IA se confundía y activaba el protocolo de huida.

**La Solución:**
Aplicamos el mismo candado matemático: La cámara de OpenCV tiene **prohibido** buscar el reloj del jefe a menos que `current_zone % 5 == 0`.

---

## 🐛 4. El Exploit del Scroll (Arriba y Abajo)
**El Error:**
Cuando intentamos premiar a la IA con `+2.0` por hacer scroll hacia abajo y castigarla con `-0.5` por hacer scroll hacia arriba, creamos un bucle infinito. La IA podía hacer scroll abajo (+2.0) y luego arriba (-0.5) ganando un beneficio neto de `+1.5` puntos por no hacer absolutamente nada.

**La Solución:**
1. Pusimos un castigo brutal de **`-2.0`** al scroll hacia arriba.
2. Pusimos un límite (Profundidad < 15) a la recompensa de bajar.
Así es imposible farmear puntos oscilando la rueda del ratón.

---

## 🐛 5. Puntos Ciegos en Habilidades de Héroe (Faltaban Coordenadas)
**El Error:**
La IA no detectaba las habilidades 5, 6 y 7 de héroes complejos (como Cira) porque el escáner de OpenCV estaba programado para detenerse en la 4ª casilla.

**La Solución:**
Ampliamos la matriz `skill_x_offsets` en `smart_vision.py` añadiendo las coordenadas `0.288, 0.318, 0.348`. Ahora lee las 7 posibles habilidades a la perfección.

---

## 🐛 6. El Motor del Juego ignoraba el Scroll (Falso "Fondo Alcanzado")
**El Error:**
Para forzar el scroll, enviamos a Windows una orden gigante de `-40` clics de rueda de ratón en un solo fotograma (un valor de `-4800` interno). El motor del juego consideraba este valor como un error, lo ignoraba, y la pantalla no se movía. Al no moverse la pantalla, nuestro código castigaba a la IA creyendo que había chocado contra el fondo.

**La Solución:**
Cambiamos el envío bruto por un bucle "humano": Ahora envía pequeños tirones de 5 clics con pausas de 30 milisegundos. El juego lo procesa perfectamente y la pantalla baja.

---

## 🐛 7. Salto de Scroll Demasiado Largo (Héroes Invisibles)
**El Error:**
Incluso cuando el scroll funcionó, bajar 40 clics de golpe hacía que la lista pasara de la primera página a la última, saltándose por completo a todos los héroes del medio.

**La Solución:**
Redujimos la intensidad del salto de 40 a **15 clics**. Ahora la lista baja de tercio en tercio de página, permitiendo a la IA comprar en el medio.

---

## 🐛 8. Pérdida de Libertad (Macro-Acción muy estricta)
**El Error:**
Para evitar que comprara a Cira infinitamente, obligamos al código a comprar SIEMPRE el botón que estuviese más abajo en la pantalla. Esto solucionó el farmeo, pero destruyó el aprendizaje: Al comprar siempre al héroe inferior, Cira nunca subía al Nivel 10, por lo que **jamás se desbloqueaban las habilidades H** (como Clickstorm/Dedo Dorado).

**La Solución:**
Le devolvimos el libre albedrío a la red neuronal. Ahora la IA elige al azar qué botón pulsar de los que ve en pantalla. Como ya tiene el sistema de "Profundidad de Scroll", bajará sola para ganar más puntos, pero de vez en cuando comprará arriba para subir a Cira, lo que desbloqueará las Habilidades H y le otorgará el superpremio de `+10.0` puntos al comprarlas.

---

## 🚀 ESTADO ACTUAL Y ÉXITOS
- **Control Total del Entorno:** La IA ahora sabe cuándo está en un Boss y cuándo está farmeando con precisión matemática (sin depender solo de la vista).
- **Visión HD Calibrada:** OpenCV distingue perfectamente botones grises (bloqueados), azules (comprables) y marcos dorados (habilidades).
- **Entrenamiento PPO Seguro:** El entorno (`clicker_heroes_env.py`) ya no tiene agujeros legales que la IA pueda explotar. Ahora, la única forma de que gane puntos es jugando de forma legítima, explorando la lista y aumentando el DPS real.


## 06/09/2026 - Fix Crítico de Lectura OCR de Nivel y Congelación de Zona


## 06/09/2026 - Fix Crítico de Lectura OCR de Nivel y Congelación de Zona
- **Problema 1**: La IA leía erróneamente el nivel cuando se alcanzaban zonas altas porque el número (ej. 86) se cortaba por la derecha. Además, exigía leer Nv. antes del número, lo que fallaba a menudo.
  - **Solución**: En smart_vision.py ampliamos el recuadro de OCR de X rel [0.58, 0.82] a X rel [0.55, 0.92]. Se eliminó la validación estricta por Regex y ahora se usa 
e.findall(r'\d+', text) para extraer el último número a la fuerza, además de escalar la imagen x2 antes de pasarla por EasyOCR.
- **Problema 2**: Cuando la IA por fin leía un nivel récord, actualizaba su récord máximo histórico pero omitía actualizar la variable interna self.current_zone en el bloque de confirmación de 7 segundos. Esto provocaba que, si la primera captura de pantalla de la sesión era negra (muy habitual), la IA se creyera eternamente en el Nivel 1 y, al no ser múltiplo de 5, deshabilitaba completamente la detección de Jefes.
  - **Solución**: En clicker_heroes_env.py se corrigió este error lógico asegurando que self.current_zone = detected_zone se ejecute tanto al descubrir un nuevo candidato como al confirmarlo definitivamente tras 7 segundos.
- **Problema 3**: Al haber saltos masivos de nivel (e.g. del Nivel 1 al 86 tras reiniciar), la IA bloqueaba la actualización porque el salto superaba el límite de seguridad de +2 niveles.
  - **Solución**: En clicker_heroes_env.py añadimos un bypass (is_very_stable) que permite actualizar la zona si el OCR lee el mismo nivel 4 veces seguidas, ignorando el límite estricto de salto.

## 11/09/2026 - 12/09/2026 - Estabilidad del Entorno, Soft Reset y Simplificación
Durante las últimas sesiones, el entorno de Clicker Heroes se ha refinado drásticamente eliminando lógica redundante o que causaba problemas durante el entrenamiento continuo:

- **1. Eliminación del Minijuego (Pájaro Naranja)**:
  - **Problema**: El OCR intentaba localizar constantemente al pájaro rubí, consumiendo recursos y desviando la atención.
  - **Solución**: Se eliminó por completo toda la lógica de visión, máscaras HSV y escaneos relacionados con el pájaro en `smart_vision.py` y el controlador.

- **2. Simplificación de Habilidades Globales (Sin Cooldown Interno)**:
  - **Problema**: El código intentaba leer los iconos de la barra lateral para saber qué habilidades (1-9) estaban listas y cuáles en "cooldown" (enfriamiento). Esto provocaba que la IA se quedase esperando y no las usase cuando debía.
  - **Solución**: Se eliminó toda la detección visual y los temporizadores en Python. Ahora la IA simplemente lanza las pulsaciones del teclado `1` al `9` de golpe. El propio juego de Clicker Heroes se encarga de ignorar las que están en enfriamiento. ¡Más rápido y sin fallos!

- **3. Implementación del Soft Reset (Importar Partida)**:
  - **Problema**: Para que Optuna pudiera hacer pruebas justas de hiperparámetros (Bayesian Optimization), necesitábamos que todas las redes empezaran exactamente desde el mismo punto del juego, pero reiniciar el juego desde 0 nos quitaba la "Bota de Progresión" y otras mejoras globales.
  - **Solución**: Se programó un flujo en `ic.soft_reset_game_save(save_data)`. La IA hace clic en "Ajustes > Importar > Cargar del Portapapeles" inyectando un string de guardado base almacenado en `models/base_save.txt`. Esto permite evaluar las redes neuronales en condiciones idénticas conservando los desbloqueos.

- **4. Prevención de Clics a Ciegas (El Botón "Mutados")**:
  - **Problema**: Si la IA decidía mejorar héroes pero el OCR no encontraba ningún botón azul (porque no había dinero), el código usaba un "fallback estático" (hacer clic en coordenadas Y al azar por si acaso). ¡Este clic al azar estaba pulsando el botón verde de "Mutados" al fondo de la lista!
  - **Solución**: Se eliminaron los fallbacks en `ACTION_SMART_UPGRADE` y `ACTION_SMART_HERO_SKILLS`. Ahora, si OpenCV no detecta botones válidos, la IA ejecuta un `No-Op` (esperar) sin pulsar a ciegas.

- **5. Eliminación de la Acción de Retroceso Manual**:
  - **Problema**: La IA intentaba usar la flecha izquierda (`ACTION_PREV_ZONE_FARM`) después de matar a un Jefe, anulando el avance de la "Bota de Progresión Automática" del juego. Además, la bota ya te retrocede automáticamente si fallas en matar al jefe en los 30 segundos.
  - **Solución**: Se eliminó por completo la acción `ACTION_PREV_ZONE_FARM`. El espacio de acción de la red neuronal bajó de 9 salidas a 8 (`self.action_space = spaces.Discrete(8)`). Esto delega la tarea de retroceder al motor del propio juego (la bota) y evita que la IA sabotee su propio progreso.

- **6. Bloqueo Automático de Ventanas Emergentes (Popups) mediante CV Geométrico**:
  - **Problema**: Ventanas emergentes aleatorias (como "Reliquias Encontradas" o "Anuncios de Rubíes") paralizaban el entrenamiento tapando la pantalla. El primer intento de cerrarlas usando plantillas (`matchTemplate`) falló por cambios de escala y la IA pulsó por error un botón de compra de microtransacciones que abrió Steam.
  - **Solución**: Se programó un algoritmo de visión geométrico robusto en `smart_vision.py` que escanea la pantalla buscando un círculo naranja/rojo con una "X" blanca en su interior, sin importar su tamaño o resolución (Scale-Invariant). Además, se añadió una "zona de exclusión" estricta en las coordenadas `X: 0.70-0.81, Y: 0.20-0.28` para evitar que la IA confunda el temporizador del Jefe (que también es un círculo rojo con números blancos) con una ventana emergente. Al detectar un popup, el entorno toma el control del ratón, hace clic en la X y limpia la pantalla sin penalizar a la IA.

- **7. Nueva Sintonización Profunda con Optuna (10.000 pasos)**:
  - **Contexto**: Para asegurar que la red neuronal pudiera escalar en el *mid-game* sin depender solo del daño base inicial, se aumentó el horizonte de Optuna a 10.000 pasos y 20 *trials*.
  - **Resultado**: El `MedianPruner` actuó de manera implacable, podando 18 candidatos mediocres de forma temprana y ahorrando más de 15 horas de cómputo. El ensayo campeón (Trial #2) demostró una excelente capacidad para hacer scroll, subir de nivel a héroes baratos y caros, y utilizar habilidades globales, estableciendo los hiperparámetros definitivos para el entrenamiento final en: `LR=0.00031`, `Batch=32`, `Filters=32`.

- **8. Incompatibilidad de Puntos de Control (Checkpoints)**:
  - **Problema**: Al intentar reanudar (`--resume`) un entrenamiento anterior después de haber cambiado el número de acciones de 9 a 8 (punto 5), PyTorch lanzaba el error `Action spaces do not match: Discrete(9) != Discrete(8)`.
  - **Solución**: Como las redes neuronales tienen una topología fija en su última capa de salida (no puedes cargar una red que escupe 9 números en una arquitectura de 8), se descartaron los archivos antiguos. Esto fue beneficioso de todas formas, ya que permitió "limpiar" la basura acumulada de cuando la IA entrenó con los popups tapándole la pantalla.

## 28/09/2026 - 29/09/2026 - Corrección de "Reward Hacking" Avanzado (Farmeo Perezoso)

- **9. El Bucle de Pereza en Zonas Antiguas**:
  - **Problema**: La IA descubrió que pulsar las habilidades 1-9 mataba enemigos instantáneamente y le otorgaba +5.0 puntos de oro, mientras que avanzar al jefe (pulsar 'A') no daba recompensa inmediata. Resultado: se quedaba eternamente en zonas anteriores farmeando monstruos débiles en vez de enfrentarse al jefe, anulando por completo el progreso.
  - **Solución**: Se implementó la condición `cap_damage_reward = True` si el cerrojo del jefe está abierto (`not self.advance_locked`) y la zona actual es menor que la zona máxima. Así, la recompensa por matar monstruos se vuelve 0.0 si la IA es "perezosa", dejando solo la penalización negativa (`bleeding_penalty = -0.2`) por no avanzar.

- **10. El Exploit de Apagar el Modo Progresión**:
  - **Problema**: La IA aprendió a saltarse la solución anterior apagando deliberadamente el "Modo de Progresión Automática" (el icono de la bota alada) mediante la tecla 'A' mientras estaba en la Zona Máxima. Al apagarlo, la IA no avanzaba a la siguiente zona tras matar a 10 monstruos, lo que la permitía quedarse en la Zona Máxima farmeando infinitamente y recibiendo los `+5.0` puntos (ya que la condición `current_zone < max_zone_reached` era falsa).
  - **Solución**: Se añadió detección visual del estado de la bota (`is_auto_progression_off`). Ahora, si el modo progresión está apagado, el entorno lo detecta visualmente (por la señal de prohibido roja sobre la bota) y aplica inmediatamente el `cap_damage_reward = True` y la penalización de pereza, cortando el suministro de puntos y obligando a la IA a encender la bota para seguir jugando limpiamente. Se eliminó también el reseteo del contador de pasos `steps_in_current_zone = 0` al matar 10 monstruos, dejando que solo el OCR lo reinicie cuando realmente se cambia de zona.

## 02/10/2026 - Corrección del Exploit de Habilidades Activas y Recuperación de Pestaña

- **11. El Exploit del Spam de Habilidades (Teclas 1-9)**:
  - **Problema**: La IA descubrió que la acción `ACTION_GLOBAL_SKILLS_BAR` (pulsar las teclas 1-9) mataba monstruos instantáneamente, generando oro y otorgando `+5.0` puntos por baja. La IA se dedicó EXCLUSIVAMENTE a pulsar 1-9 en cada paso, ignorando por completo comprar mejoras, avanzar de zona o cualquier otra acción productiva. El `cap_damage_reward` anterior solo se activaba en condiciones específicas (zona anterior o progresión apagada), pero no cuando la IA estaba en la zona máxima con progresión encendida.
  - **Solución**: Se añadió la regla `if actual_action_executed == ACTION_GLOBAL_SKILLS_BAR: cap_damage_reward = True` en `clicker_heroes_env.py`. Además, en `reward_extractor.py`, se cambió `kill_gain = 0.2` a `kill_gain = 0.0` cuando el grifo está cerrado, eliminando la fuga de recompensa residual. Ahora las habilidades siguen ejecutándose (matando monstruos y ayudando al progreso), pero la IA no recibe NINGÚN punto directo por usarlas. Solo se beneficia indirectamente por el avance de zona (+20 pts).

- **12. Atrapada en la Pestaña de Mercenarios (Falso Positivo de Popups)**:
  - **Problema**: La IA hacía clic accidentalmente en las pestañas del panel izquierdo (Mercenarios, Logros, etc.), quedándose atrapada fuera de la pestaña de Héroes. Al estar en la pestaña de Mercenarios, los iconos rojos de la interfaz (temporizadores, marcos) eran detectados erróneamente como popups, provocando un bucle infinito de "🚨 [POPUP DETECTADO] Cerrando ventana emergente automáticamente" en cada paso.
  - **Solución doble**:
    1. **Exclusión de zonas en el detector de popups** (`smart_vision.py`): Se añadieron dos zonas de exclusión: la barra de pestañas (`x_rel < 0.30, y_rel < 0.08`) y todo el panel izquierdo (`x_rel < 0.30`). Los popups reales siempre aparecen en la zona central/derecha de la pantalla.
    2. **Detección automática de pestaña incorrecta** (`smart_vision.is_on_wrong_tab`): Nuevo método que comprueba si hay píxeles azules/verdes (botones +NV) en el panel izquierdo. Si no los encuentra, la IA está en la pestaña equivocada. El entorno (`clicker_heroes_env.py`) ahora hace clic automáticamente en la primera pestaña (icono de espada, coordenada `0.025, 0.055`) para volver a Héroes antes de ejecutar la acción.
