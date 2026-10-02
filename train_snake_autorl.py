"""
Script principal para la Fase 1: Entrenamiento y Optimización AutoRL en Snake (Entorno de Píxeles).

Uso:
  - Optimizar hiperparámetros con AutoRL:
      python train_snake_autorl.py --mode tune --algo PPO --trials 10
  - Entrenar un agente directamente:
      python train_snake_autorl.py --mode train --algo PPO --timesteps 50000
"""

import os
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from src.environments.snake_env import SnakePixelEnv
from src.models.cnn_feature_extractor import CustomCNNFeatureExtractor
from src.autorl.hyperparameter_tuner import AutoRLTuner, make_env
from src.utils.visualization import evaluate_and_record_agent


def build_stacked_env(grid_size: int = 10, img_size: int = 84, frame_stack: int = 4):
    """Construye un entorno vectorizado con apilamiento de fotogramas (Frame Stacking)."""
    vec_env = DummyVecEnv([make_env(grid_size=grid_size, img_size=img_size)])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def train_single_agent(algo: str, timesteps: int, save_path: str):
    """Entrena un agente PPO o DQN de extremo a extremo en el entorno de Snake."""
    print(f"\n--- Iniciando entrenamiento de agente individual ({algo}) ---")
    env = build_stacked_env()

    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_filters_base=32,
            depth=3,
        ),
    )

    if algo.upper() == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    elif algo.upper() == "DQN":
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=20000,
            batch_size=64,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    else:
        raise ValueError(f"Algoritmo no soportado: {algo}")

    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Modelo guardado exitosamente en: {save_path}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="AutoRL Snake Vision Trainer - Fase 1")
    parser.add_argument("--mode", type=str, choices=["train", "tune", "eval"], default="train", help="Modo de ejecución")
    parser.add_argument("--algo", type=str, choices=["PPO", "DQN"], default="PPO", help="Algoritmo RL")
    parser.add_argument("--timesteps", type=int, default=30000, help="Pasos de entrenamiento")
    parser.add_argument("--trials", type=int, default=10, help="Número de ensayos para AutoRL (mode=tune)")
    parser.add_argument("--model-path", type=str, default="models/snake_ppo_model", help="Ruta del modelo guardado")
    args = parser.parse_args()

    if args.mode == "tune":
        tuner = AutoRLTuner(
            algo=args.algo,
            n_trials=args.trials,
            n_timesteps_per_trial=args.timesteps,
        )
        tuner.optimize()

    elif args.mode == "train":
        train_single_agent(
            algo=args.algo,
            timesteps=args.timesteps,
            save_path=args.model_path,
        )

    elif args.mode == "eval":
        env = build_stacked_env()
        if args.algo == "PPO":
            model = PPO.load(args.model_path, env=env)
        else:
            model = DQN.load(args.model_path, env=env)

        evaluate_and_record_agent(model, env, num_episodes=5)
        env.close()


if __name__ == "__main__":
    main()
