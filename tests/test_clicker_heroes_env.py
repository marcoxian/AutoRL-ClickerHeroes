"""
Unit and Integration Tests for Phase 2: Clicker Heroes Environment & Utilities.
"""
import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.utils.screen_capture import ScreenCapture, benchmark_capture
from src.utils.input_controller import InputController
from src.utils.reward_extractor import RewardExtractor
from src.environments.clicker_heroes_env import ClickerHeroesEnv
from src.models.cnn_feature_extractor import CustomCNNFeatureExtractor


class TestClickerHeroesPhase2(unittest.TestCase):

    def test_screen_capture_mock(self):
        """Validates ScreenCapture output shape, dimensions, and channel ordering."""
        cap = ScreenCapture(target_size=(84, 84), channel_first=True, mock_mode=True)
        frame = cap.capture_frame()

        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (3, 84, 84))
        self.assertEqual(frame.dtype, np.uint8)
        cap.close()

    def test_screen_capture_benchmark(self):
        """Verifies high performance screen capture speed (>30 FPS in mock mode)."""
        fps = benchmark_capture(num_frames=50, mock=True)
        self.assertGreater(fps, 30.0)

    def test_input_controller_coordinates(self):
        """Validates relative to absolute coordinate mapping."""
        region = {"top": 100, "left": 200, "width": 1000, "height": 500}
        ctrl = InputController(region=region, mock_mode=True)

        # Center click (0.5, 0.5) -> (200 + 500, 100 + 250) = (700, 350)
        abs_x, abs_y = ctrl.relative_to_absolute(0.5, 0.5)
        self.assertEqual((abs_x, abs_y), (700, 350))

        # Action execution for monster click (0.73, 0.55) -> (200 + 730, 100 + 275) = (930, 375)
        coords = ctrl.execute_action(InputController.ACTION_MONSTER_CLICK)
        self.assertIsNotNone(coords)
        self.assertEqual(coords, (930, 375))

        # No-op execution
        noop_coords = ctrl.execute_action(InputController.ACTION_NOOP)
        self.assertIsNone(noop_coords)

    def test_reward_extractor(self):
        """Validates HP bar reading and strict event-based reward computation."""
        extractor = RewardExtractor(mock_mode=True)
        extractor.reset(initial_hp=1.0)

        # Generate dummy frame
        cap = ScreenCapture(target_size=(84, 84), channel_first=False, mock_mode=True)
        dummy_frame = cap.capture_frame()

        # Step reward without kill (solo penalización por tiempo)
        reward1, info1 = extractor.compute_reward(dummy_frame, action=6)
        self.assertLess(reward1, 0.0)  # Step penalty (-0.01)

        # Simular muerte de enemigo (HP pasa de 0.1 a 1.0 del nuevo enemigo)
        extractor.last_hp = 0.1
        extractor.mock_hp = 1.0
        reward2, info2 = extractor.compute_reward(dummy_frame, action=0)
        self.assertGreater(reward2, 4.0)  # Kill reward (+5.0 - 0.01 = +4.99)
        self.assertTrue(info2["monster_killed"])

        cap.close()


    def test_clicker_heroes_env_gym_api(self):
        """Validates ClickerHeroesEnv integration with Gymnasium API."""
        env = ClickerHeroesEnv(target_size=(84, 84), max_steps=10, mock_mode=True)

        self.assertEqual(env.observation_space.shape, (3, 84, 84))
        self.assertEqual(env.action_space.n, 9)


        obs, info = env.reset()
        self.assertEqual(obs.shape, (3, 84, 84))

        for step_idx in range(5):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)

            self.assertEqual(next_obs.shape, (3, 84, 84))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("step", step_info)

        env.close()

    def test_cnn_extractor_compatibility(self):
        """Validates PyTorch CustomCNNFeatureExtractor compatibility with ClickerHeroesEnv observations."""
        env = ClickerHeroesEnv(target_size=(84, 84), max_steps=5, mock_mode=True)
        obs, _ = env.reset()

        cnn = CustomCNNFeatureExtractor(observation_space=env.observation_space, features_dim=256)

        # Convert observation to PyTorch tensor batch: (1, 3, 84, 84)
        obs_tensor = torch.as_tensor(obs[None]).float()
        features = cnn(obs_tensor)

        self.assertEqual(features.shape, (1, 256))
        env.close()


if __name__ == "__main__":
    unittest.main()
