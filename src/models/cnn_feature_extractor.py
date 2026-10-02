"""
Extractor de características Convolucional (CNN) en PyTorch para procesamiento de imágenes en RL.
Compatible con Stable-Baselines3 y configurable para auto-tuning (AutoRL).
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Red Neuronal Convolucional (CNN) PyTorch para extracción de características
    a partir de imágenes o marcos apilados (Frame Stacking).

    Parámetros:
        observation_space: Espacio de observaciones de Gymnasium.
        features_dim: Dimensión del vector de salida latente (representación de estado).
        n_filters_base: Número base de filtros convolucionales (ajustable con AutoRL).
        depth: Profundidad de la red (número de bloques convolucionales: 2 o 3).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_filters_base: int = 32,
        depth: int = 3,
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        layers = []
        in_channels = n_input_channels
        out_channels = n_filters_base

        # Capa 1: Convolución inicial para reducción espacial amplia
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ])
        in_channels = out_channels

        # Capa 2: Segunda convolución
        out_channels = n_filters_base * 2
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ])
        in_channels = out_channels

        # Capa 3 opcional (según profundidad)
        if depth >= 3:
            out_channels = n_filters_base * 2
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels

        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Calcular dimensión de salida de la CNN dinámicamente con un tensor de prueba
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            # Asegurar rango normalizado [0, 1] si los valores vienen en [0, 255]
            if sample_input.max() > 1.0:
                sample_input /= 255.0
            n_flatten = self.cnn(sample_input).shape[1]

        # Capa fully connected final para proyectar al espacio de características deseado
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante: Normaliza las imágenes de entrada [0, 255] -> [0.0, 1.0]
        y extrae las características profundas.
        """
        # Normalización de píxeles
        normalized_obs = observations.float()
        if normalized_obs.max() > 1.0:
            normalized_obs = normalized_obs / 255.0

        features = self.cnn(normalized_obs)
        return self.linear(features)
