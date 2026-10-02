"""
Unit and Integration Tests for Phase 3: Emergency Stop & Clicker Heroes Training Script.
"""
import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from src.utils.safety import EmergencyStopListener, EmergencyStopCallback, EmergencyStopException
from train_clicker_heroes_autorl import build_stacked_env, train_single_agent


class TestPhase3SafetyAndTraining(unittest.TestCase):

    def test_emergency_stop_listener(self):
        """Verifies EmergencyStopListener flag request and reset functionality."""
        listener = EmergencyStopListener(check_corner=False)
        listener.reset()
        self.assertFalse(listener.check_stop_requested())

        # Request manual stop
        listener.request_stop()
        self.assertTrue(listener.check_stop_requested())

        # Reset
        listener.reset()
        self.assertFalse(listener.check_stop_requested())

    def test_emergency_stop_callback(self):
        """Verifies EmergencyStopCallback halts SB3 training and saves emergency checkpoint."""
        listener = EmergencyStopListener(check_corner=False)
        listener.reset()

        callback = EmergencyStopCallback(listener=listener, save_dir="models", verbose=0)
        env = build_stacked_env(mock_mode=True)

        model = PPO("CnnPolicy", env, n_steps=64, batch_size=32, verbose=0)
        callback.init_callback(model)

        # Before stop trigger: callback returns True
        self.assertTrue(callback._on_step())

        # Trigger emergency stop
        listener.request_stop()
        with self.assertRaises(EmergencyStopException):
            callback._on_step()

        self.assertTrue(callback.stop_triggered)

        # Verify emergency model file saved
        emergency_file = os.path.join("models", "clicker_heroes_emergency_save.zip")
        self.assertTrue(os.path.exists(emergency_file))

        env.close()
        listener.reset()

    def test_short_training_run(self):
        """Verifies end-to-end training script execution in mock mode."""
        test_model_path = os.path.join("models", "test_clicker_model")
        train_single_agent(
            algo="PPO",
            timesteps=256,
            save_path=test_model_path,
            mock_mode=True,
        )
        self.assertTrue(os.path.exists(f"{test_model_path}.zip"))


if __name__ == "__main__":
    unittest.main()
