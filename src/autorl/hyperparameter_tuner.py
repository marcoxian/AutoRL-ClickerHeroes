"""
Módulo de AutoRL para búsqueda y optimización automática de hiperparámetros usando Optuna.
Soporta PPO y DQN con arquitectura CNN y apilado de fotogramas (Frame Stacking).
"""

import os
from typing import Dict, Any, Callable
import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor
from src.environments.snake_env import SnakePixelEnv
from src.models.cnn_feature_extractor import CustomCNNFeatureExtractor


def make_env(grid_size: int = 10, img_size: int = 84) -> Callable:
    """Crea una función creadora de entorno para VecEnv."""
    def _init():
        env = SnakePixelEnv(grid_size=grid_size, img_size=img_size)
        return Monitor(env)
    return _init


class AutoRLTuner:
    """
    Optimizador de hiperparámetros AutoRL impulsado por Optuna.
    """

    def __init__(
        self,
        algo: str = "PPO",
        n_trials: int = 20,
        n_timesteps_per_trial: int = 30000,
        n_eval_episodes: int = 10,
        frame_stack: int = 4,
        grid_size: int = 10,
        img_size: int = 84,
        output_dir: str = "configs",
    ):
        self.algo = algo.upper()
        self.n_trials = n_trials
        self.n_timesteps_per_trial = n_timesteps_per_trial
        self.n_eval_episodes = n_eval_episodes
        self.frame_stack = frame_stack
        self.grid_size = grid_size
        self.img_size = img_size
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def _build_vec_env(self):
        """Construye el entorno vectorizado con Frame Stacking y TransposeImage."""
        vec_env = DummyVecEnv([make_env(grid_size=self.grid_size, img_size=self.img_size)])
        vec_env = VecFrameStack(vec_env, n_stack=self.frame_stack, channels_order="last")
        vec_env = VecTransposeImage(vec_env)
        return vec_env

    def objective(self, trial: optuna.Trial) -> float:
        """Función objetivo ejecutada por Optuna en cada ensayo (Trial)."""
        # 1. Sugerencia de hiperparámetros de modelo y optimizador
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # 2. Sugerencia de hiperparámetros de arquitectura CNN
        n_filters_base = trial.suggest_categorical("n_filters_base", [16, 32, 64])
        depth = trial.suggest_int("depth", 2, 3)
        features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])

        policy_kwargs = dict(
            features_extractor_class=CustomCNNFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                n_filters_base=n_filters_base,
                depth=depth,
            ),
        )

        train_env = self._build_vec_env()

        try:
            if self.algo == "PPO":
                n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
                ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True)

                model = PPO(
                    "CnnPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                )

            elif self.algo == "DQN":
                buffer_size = trial.suggest_categorical("buffer_size", [10000, 30000, 50000])
                target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 2000])

                model = DQN(
                    "CnnPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    gamma=gamma,
                    target_update_interval=target_update_interval,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                )
            else:
                raise ValueError(f"Algoritmo no soportado: {self.algo}")

            # Entrenamiento del modelo en el ensayo actual
            model.learn(total_timesteps=self.n_timesteps_per_trial)

            # Evaluación final de rendimiento
            eval_env = self._build_vec_env()
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            train_env.close()
            eval_env.close()

            return float(mean_reward)

        except Exception as e:
            train_env.close()
            print(f"[Optuna Trial {trial.number}] Falló con error: {e}")
            return -9999.0

    def optimize(self) -> Dict[str, Any]:
        """Ejecuta el proceso completo de optimización de AutoRL."""
        sampler = TPESampler(n_startup_trials=5)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10000)

        study = optuna.create_study(
            study_name=f"snake_autorl_{self.algo.lower()}",
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        print(f"\n--- Iniciando Búsqueda AutoRL con Optuna ({self.algo}) ---")
        print(f"Ensayo totales: {self.n_trials} | Pasos por ensayo: {self.n_timesteps_per_trial}")

        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        print("\n=== ¡Optimización AutoRL Completada! ===")
        print(f"Mejor Recompensa Media Obtenida: {study.best_value:.2f}")
        print("Mejores Hiperparámetros Encontrados:")
        for key, val in study.best_params.items():
            print(f"  - {key}: {val}")

        # Guardar la mejor configuración en YAML
        best_config_path = os.path.join(self.output_dir, f"best_snake_{self.algo.lower()}_config.yaml")
        config_data = {
            "algo": self.algo,
            "best_reward": float(study.best_value),
            "hyperparameters": study.best_params,
            "environment": {
                "grid_size": self.grid_size,
                "img_size": self.img_size,
                "frame_stack": self.frame_stack,
            },
        }

        with open(best_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        print(f"Configuración guardada exitosamente en: {best_config_path}")

        return study.best_params
