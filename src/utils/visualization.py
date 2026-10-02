"""
Módulo de visualización y evaluación de agentes entrenados en entorns de píxeles.
"""

from typing import Optional
import numpy as np
import cv2
from stable_baselines3.common.base_class import BaseAlgorithm


def evaluate_and_record_agent(
    model: BaseAlgorithm,
    env,
    num_episodes: int = 3,
    save_video_path: Optional[str] = None,
    fps: int = 15,
):
    """
    Evalúa un agente RL entrenado y opcionalmente graba la ejecución visual en video MP4.
    """
    print(f"\n--- Evaluando Agente durante {num_episodes} episodios ---")

    video_writer = None
    frame_shape = None

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            steps += 1

            # Captura de fotograma para video
            frame = env.render()
            if frame is not None and save_video_path:
                h, w, _ = frame.shape
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (w, h))
                
                # Convertir RGB a BGR para OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

        print(f"Episodio {episode + 1}: Recompensa Total = {total_reward:.2f} | Pasos = {steps}")

    if video_writer:
        video_writer.release()
        print(f"Video guardado exitosamente en: {save_video_path}")
