"""
Pruebas unitarias automatizadas para verificar el entorno de Snake y la extracción de características CNN.
Utiliza unittest nativo de Python.
"""

import os
import sys
import unittest

# Asegurar que el directorio raíz del proyecto esté en sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from src.environments.snake_env import SnakePixelEnv
from src.models.cnn_feature_extractor import CustomCNNFeatureExtractor


class TestSnakeVision(unittest.TestCase):

    def test_snake_env_initialization(self):
        """Verifica la correcta inicialización del entorno SnakePixelEnv."""
        env = SnakePixelEnv(grid_size=10, img_size=84)
        self.assertEqual(env.observation_space.shape, (84, 84, 3))
        self.assertEqual(env.observation_space.dtype, np.uint8)
        self.assertEqual(env.action_space.n, 4)
        env.close()

    def test_snake_env_reset_and_step(self):
        """Verifica reset() y step() del entorno."""
        env = SnakePixelEnv(grid_size=10, img_size=84)
        obs, info = env.reset()

        self.assertEqual(obs.shape, (84, 84, 3))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertIn("score", info)

        # Ejecutar 5 pasos aleatorios
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            self.assertEqual(obs.shape, (84, 84, 3))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)

            if terminated or truncated:
                break

        env.close()

    def test_frame_stacking_wrapper(self):
        """Verifica que el apilado de fotogramas (Frame Stacking) funcione correctamente."""
        def make_env():
            return SnakePixelEnv(grid_size=10, img_size=84)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="last")
        vec_env = VecTransposeImage(vec_env)

        obs = vec_env.reset()
        # Con 4 fotogramas RGB apilados, la forma debe ser (1, 12, 84, 84) en PyTorch (12 canales: 4 frames * 3 RGB)
        self.assertEqual(obs.shape, (1, 12, 84, 84))

        vec_env.close()

    def test_cnn_feature_extractor_forward(self):
        """Verifica el paso forward de la CNN en PyTorch."""
        def make_env():
            return SnakePixelEnv(grid_size=10, img_size=84)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="last")
        vec_env = VecTransposeImage(vec_env)

        feature_extractor = CustomCNNFeatureExtractor(
            observation_space=vec_env.observation_space,
            features_dim=256,
            n_filters_base=32,
            depth=3,
        )

        sample_obs = torch.as_tensor(vec_env.reset()).float()
        features = feature_extractor(sample_obs)

        self.assertEqual(features.shape, (1, 256))
        vec_env.close()


if __name__ == "__main__":
    unittest.main()
