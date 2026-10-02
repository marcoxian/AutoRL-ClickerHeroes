"""
Script de Evaluación y Despliegue Autónomo del Agente Entrenado de Clicker Heroes / Idle Slayer.

Carga el modelo final entrenado en la Fase 3 (models/clicker_heroes_ppo_model.zip) y ejecuta
el agente en vivo en el juego con parada de emergencia mediante la tecla 'ESC'.

Ejecutar:
    python evaluate_clicker_heroes.py
"""

import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environments.clicker_heroes_env import ClickerHeroesEnv
from src.utils.safety import global_emergency_listener, EmergencyStopException
from src.utils.window_finder import find_game_window, WindowNotFoundError
from src.utils.input_controller import InputController


def make_env(img_size: int = 84):
    def _thunk():
        return ClickerHeroesEnv(
            target_size=(img_size, img_size),
            max_steps=1000,
            action_delay=0.03,
            mock_mode=False,
        )
    return _thunk


def main():
    model_path = os.path.join("models", "clicker_heroes_ppo_model.zip")
    if not os.path.exists(model_path) and not os.path.exists(model_path.replace(".zip", "")):
        # Probar modelo de emergencia
        model_path = os.path.join("models", "clicker_heroes_emergency_save.zip")

    if not os.path.exists(model_path) and not os.path.exists(model_path.replace(".zip", "")):
        # Buscar cualquier archivo .zip en models/
        if os.path.exists("models"):
            zip_files = [f for f in os.listdir("models") if f.endswith(".zip")]
            if zip_files:
                model_path = os.path.join("models", zip_files[0])

    print("=" * 65)
    print("🤖 [DESPLIEGUE AUTÓNOMO] Evaluando Agente IA Entrenado en Clicker Heroes")
    print(f"📦 Cargar Modelo: {model_path}")
    print("=" * 65)


    try:
        reg = find_game_window(title_substring="Clicker Heroes", bring_to_front=True)
        print(f"[VENTANA ENCONTRADA] Ubicación: ({reg['left']}, {reg['top']}), Tamaño: {reg['width']}x{reg['height']} px")
    except WindowNotFoundError as e:
        print(e)
        return

    print("\n[CARGANDO MODELO] Cargando pesos entrenados...")
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    model = PPO.load(model_path, env=vec_env)
    print("✔ Modelo PPO cargado exitosamente.")
    print("\n🎮 [INICIANDO EJECUCIÓN AUTÓNOMA]")
    print("👉 Presiona la tecla 'ESC' (1 solo toque) en cualquier momento para detener la IA.\n")

    global_emergency_listener.reset()
    obs = vec_env.reset()
    total_reward = 0.0
    action_names = {
        0: "Ataque / Combate",
        1: "Mejorías Inteligentes (+NV)",
        2: "Habilidades de Héroes",
        3: "Habilidades Activas Globales",
        4: "Flecha Izq (Farmeo de Oro)",
        5: "Flecha Der (Progreso / Boss)",
        6: "Scroll Abajo",
        7: "Scroll Arriba",
        8: "Esperar (No-op)",
    }

    try:
        step = 0
        while True:
            global_emergency_listener.raise_if_stopped()

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            action_idx = int(action[0])
            act_name = action_names.get(action_idx, f"Acción {action_idx}")
            step_reward = float(rewards[0])
            total_reward += step_reward
            step += 1

            if step % 10 == 0:
                curr_hp = infos[0].get("current_hp", 1.0)
                kills = infos[0].get("consecutive_kills", 0)
                print(f"Paso {step:04d} | Acción: {act_name:<32} | Vida HP: {curr_hp:.2f} | Kills 10/10: {kills}/10 | Recompensa Total: {total_reward:.2f}")

            if dones[0]:
                obs = vec_env.reset()

            time.sleep(0.01)

    except (EmergencyStopException, KeyboardInterrupt):
        print("\n🚨 [PARADA DE EMERGENCIA] Ejecución del agente detenida por el usuario.")
    finally:
        vec_env.close()
        print(f"\n=========================================================")
        print(f"📊 [RESUMEN FINAL] Pasos ejecutados: {step} | Recompensa Total: {total_reward:.2f}")
        print(f"=========================================================")


if __name__ == "__main__":
    main()
