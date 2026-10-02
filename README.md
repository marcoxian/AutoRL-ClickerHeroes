# 🎮 AutoRL Clicker Heroes

**Una Inteligencia Artificial que aprende a jugar a [Clicker Heroes](https://store.steampowered.com/app/363970/Clicker_Heroes/) completamente sola**, usando únicamente los píxeles de la pantalla y Aprendizaje por Refuerzo Profundo (Deep RL).

No accede a la memoria del juego. No usa trucos. Ve la pantalla como un humano y aprende a base de prueba y error.

> **🟢 Estado actual:** El agente ya es capaz de **farmear Almas de Héroe y realizar ascensiones de forma completamente desatendida**. Sube de zona, compra mejoras de héroes, usa habilidades, derrota jefes y, cuando se estanca, asciende automáticamente por el portal para acumular Almas de Héroe. Puedes dejarlo funcionando durante horas sin intervención.

---

## 🧠 ¿Cómo funciona?

```
Pantalla del Juego → OpenCV (Visión) → Red Neuronal CNN → Acción (clic/teclado) → Recompensa → Repetir
```

1. **Captura de pantalla** a alta velocidad con `mss` (capturas GDI32 de Windows)
2. **Visión por computador** con `OpenCV`: detecta botones comprables, nivel actual (OCR), barras de vida de jefes, popups, y estado de la interfaz
3. **Red Neuronal Convolucional** (CNN) procesa la imagen reducida a 84×84 píxeles
4. **PPO** (Proximal Policy Optimization) decide qué acción ejecutar entre 8 posibles
5. **Recompensa** basada en progresión real: avanzar zonas (+20 pts), derrotar jefes (+100 pts), comprar mejoras (+1-5.5 pts)
6. **Control de periféricos** con `pydirectinput` (DirectX) para evadir sistemas anti-bot

## 🏗️ Arquitectura del Proyecto

```
AutoRL-ClickerHeroes/
├── train_clicker_heroes_autorl.py  # 🚀 Script principal de entrenamiento
├── evaluate_clicker_heroes.py      # 📊 Evaluación del agente entrenado
├── src/
│   ├── environments/
│   │   └── clicker_heroes_env.py   # 🎮 Entorno Gymnasium personalizado
│   ├── models/
│   │   └── cnn_feature_extractor.py # 🧠 CNN Feature Extractor para SB3
│   ├── utils/
│   │   ├── smart_vision.py         # 👁️ Visión por computador (OpenCV + HSV)
│   │   ├── input_controller.py     # 🖱️ Control de ratón/teclado (DirectX)
│   │   ├── reward_extractor.py     # 🏆 Sistema de recompensas anti-exploit
│   │   ├── screen_capture.py       # 📸 Captura de pantalla (mss)
│   │   ├── window_finder.py        # 🔍 Detección de ventana del juego
│   │   ├── safety.py               # 🛑 Parada de emergencia (tecla ESC)
│   │   └── visualization.py        # 📈 Visualización en tiempo real
│   └── autorl/
│       └── hyperparameter_tuner.py # ⚙️ Optimización Bayesiana con Optuna
├── tests/                          # 🧪 Tests unitarios
├── models/
│   └── base_save.txt               # 💾 Partida base para soft-reset
├── configs/                        # 📋 Configuraciones YAML
├── autorl_training_history.md      # 📚 Historial de bugs y soluciones
├── PROJECT_CONTEXT.md              # 📖 Documentación técnica detallada
└── requirements.txt                # 📦 Dependencias
```

## 🎯 Espacio de Acciones

La IA puede ejecutar **8 acciones** en cada paso:

| Acción | Descripción |
|---|---|
| 🖱️ **Click Monstruo** | Ráfaga de clics sobre los monstruos |
| ⬆️ **Mejorar Héroe** | Compra mejoras +NV detectadas por OpenCV |
| ⚔️ **Habilidad de Héroe** | Desbloquea habilidades pasivas de héroes |
| 🔥 **Habilidades Globales** | Activa las teclas 1-9 (Clickstorm, etc.) |
| ➡️ **Avanzar Zona / Boss** | Activa la progresión automática |
| 🔽 **Scroll Abajo** | Baja en la lista de héroes |
| 🔼 **Scroll Arriba** | Sube en la lista de héroes |
| ⏸️ **No-Op** | Espera (DPS pasivo genera oro) |

## 🛡️ Sistema Anti-Exploit (Reward Hacking)

Una de las partes más fascinantes del proyecto es la **guerra constante contra la IA** que intenta hackear su propia recompensa. Algunos exploits que descubrió y que tuvimos que parchear:

| # | Exploit | Solución |
|---|---|---|
| 1 | Farmear mejoras baratas infinitamente | Sistema de profundidad de scroll |
| 2 | OCR alucinaba retrocesos de zona | Candado matemático (solo en zonas múltiplo de 5) |
| 3 | Explotar scroll arriba/abajo sin parar | Penalización de -2.0 por scroll up |
| 4 | Spamear habilidades 1-9 para oro gratis | `cap_damage_reward = True` para skills |
| 5 | Apagar el modo progresión adrede | Detección visual del icono de la bota |
| 6 | Quedarse en zonas anteriores farmeando | Bleeding penalty (-0.2/paso) |

> 📖 Historial completo de los 12 exploits documentados en [`autorl_training_history.md`](autorl_training_history.md)

## 🚀 Cómo Usar

### Requisitos Previos
- **Windows 10/11** (usa APIs de Windows para captura y control)
- **Python 3.10+**
- **Clicker Heroes** instalado en Steam (ventana visible, no minimizada)

### Instalación

```bash
git clone https://github.com/marcoxian/AutoRL-ClickerHeroes.git
cd AutoRL-ClickerHeroes
pip install -r requirements.txt
```

### Entrenar al Agente

```bash
# Entrenamiento desde cero (abre Clicker Heroes antes de ejecutar)
python train_clicker_heroes_autorl.py --mode train --algo PPO --timesteps 500000 --live --lr 0.00031 --batch-size 32 --filters 32

# Reanudar entrenamiento previo
python train_clicker_heroes_autorl.py --mode train --algo PPO --timesteps 500000 --live --lr 0.00031 --batch-size 32 --filters 32 --resume
```

### Controles Durante el Entrenamiento
- **`ESC`** → Parada de emergencia (guarda checkpoint automáticamente)
- **`P`** → Pausar/Reanudar (libera el ratón)

### Evaluar el Agente

```bash
python evaluate_clicker_heroes.py
```

## 🔧 Stack Tecnológico

| Componente | Librería |
|---|---|
| Aprendizaje por Refuerzo | `Stable-Baselines3` (PPO) |
| Entorno | `Gymnasium` (custom) |
| Visión por Computador | `OpenCV` + `EasyOCR` |
| Red Neuronal | `PyTorch` (CNN) |
| Optimización | `Optuna` (búsqueda Bayesiana) |
| Captura de Pantalla | `mss` |
| Control de Input | `pydirectinput` (DirectX) |

## 📊 Progreso Actual

- ✅ **Farmeo autónomo de Almas de Héroe** — Sube zonas, compra mejoras, mata jefes y asciende solo
- ✅ Entorno Gymnasium funcional con visión en tiempo real
- ✅ 12 exploits de Reward Hacking detectados y parcheados
- ✅ Ascensión automática vía portal cuando se estanca (>300 pasos, ≥10 almas)
- ✅ Detección y cierre automático de popups del juego
- ✅ Recuperación automática si cambia a pestaña incorrecta
- ✅ Optimización de hiperparámetros con Optuna (20 trials)
- 🔄 Entrenamiento en curso (~92K/500K pasos) — mejorando estrategia continuamente

## 📄 Licencia

Este proyecto es de uso educativo y personal. Clicker Heroes es propiedad de Playsaurus.
