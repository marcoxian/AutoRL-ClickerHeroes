"""
Entorno de Snake con Salida Visual en Píxeles compatible con Gymnasium.

Este entorno implementa el juego clásico de Snake convirtiendo el estado del tablero
en una matriz de píxeles RGB (H, W, 3) lista para ser procesada por redes convolucionales (CNN).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces


class SnakePixelEnv(gym.Env):
    """
    Entorno Gymnasium de Snake basado en observaciones de fotogramas RGB.

    Acciones:
        0: ARRIBA (UP)
        1: DERECHA (RIGHT)
        2: ABAJO (DOWN)
        3: IZQUIERDA (LEFT)

    Observaciones:
        Imagen RGB de dimensiones (screen_size, screen_size, 3) con valores en [0, 255].
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 15}

    def __init__(
        self,
        grid_size: int = 10,
        img_size: int = 84,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        reward_shaping: bool = True,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.img_size = img_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping

        # Espacio de acciones discretas (4 direcciones)
        self.action_space = spaces.Discrete(4)

        # Espacio de observaciones: Imagen RGB (H, W, 3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.img_size, self.img_size, 3),
            dtype=np.uint8,
        )

        # Direcciones vectoriales [dy, dx]
        self._action_to_direction = {
            0: np.array([-1, 0]),  # UP
            1: np.array([0, 1]),   # RIGHT
            2: np.array([1, 0]),   # DOWN
            3: np.array([0, -1]),  # LEFT
        }

        # Paleta de colores RGB
        self.COLOR_BG = (15, 15, 20)        # Fondo oscuro estilo cyberpunk
        self.COLOR_SNAKE_HEAD = (0, 255, 128) # Verde neón brillante
        self.COLOR_SNAKE_BODY = (0, 180, 90)  # Verde medio
        self.COLOR_FOOD = (255, 50, 50)       # Rojo brillante

        # Estado interno
        self.snake = []
        self.direction = np.array([0, 1])
        self.food = np.array([0, 0])
        self.steps_taken = 0
        self.score = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno a un estado inicial."""
        super().reset(seed=seed)

        self.steps_taken = 0
        self.score = 0

        # Posición inicial de la serpiente (longitud 3 en el centro)
        center_y = self.grid_size // 2
        center_x = self.grid_size // 2
        self.snake = [
            np.array([center_y, center_x]),
            np.array([center_y, center_x - 1]),
            np.array([center_y, center_x - 2]),
        ]
        self.direction = np.array([0, 1]) # Inicia moviéndose a la derecha

        # Generar comida en posición válida
        self._spawn_food()

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta una acción en el entorno."""
        self.steps_taken += 1

        # Evitar giro de 180 grados instantáneo sobre sí misma
        new_dir = self._action_to_direction[action]
        if not np.array_equal(new_dir, -self.direction):
            self.direction = new_dir

        # Calcular nueva posición de la cabeza
        new_head = self.snake[0] + self.direction

        # Distancia Manhattan previa a la comida (para reward shaping)
        prev_dist = np.abs(self.snake[0] - self.food).sum()

        # Comprobar colisión con paredes o consigo misma
        terminated = False
        reward = -0.01  # Penalización pequeña por paso para acelerar caminos cortos

        if self._check_collision(new_head):
            terminated = True
            reward = -10.0
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, terminated, False, info

        # Mover la serpiente
        self.snake.insert(0, new_head)

        # Comprobar si ha comido la manzana
        if np.array_equal(new_head, self.food):
            self.score += 1
            reward = 10.0
            self._spawn_food()
        else:
            self.snake.pop() # Quitar cola si no comió

            # Reward shaping opcional basado en si se acerca a la manzana
            if self.reward_shaping:
                curr_dist = np.abs(new_head - self.food).sum()
                if curr_dist < prev_dist:
                    reward += 0.1
                else:
                    reward -= 0.15

        # Límite de pasos alcanzado (truncation)
        truncated = self.steps_taken >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Verifica si la posición colisiona con límites o el cuerpo."""
        y, x = pos
        # Colisión con límites del mapa
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Colisión con su propio cuerpo
        for segment in self.snake[:-1]:
            if np.array_equal(pos, segment):
                return True
        return False

    def _spawn_food(self) -> None:
        """Genera una manzana en una casilla libre."""
        empty_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = np.array([r, c])
                if not any(np.array_equal(cell, seg) for seg in self.snake):
                    empty_cells.append(cell)

        if empty_cells:
            idx = self.np_random.integers(0, len(empty_cells))
            self.food = empty_cells[idx]
        else:
            # Tablero completo ganado
            self.food = np.array([-1, -1])

    def _get_observation(self) -> np.ndarray:
        """Renderiza el estado actual en una imagen RGB ndarray (H, W, 3)."""
        # Crear lienzo vacío en baja resolución (grid_size x grid_size)
        grid_img = np.full((self.grid_size, self.grid_size, 3), self.COLOR_BG, dtype=np.uint8)

        # Dibujar comida
        if self.food[0] >= 0:
            grid_img[self.food[0], self.food[1]] = self.COLOR_FOOD

        # Dibujar cuerpo de la serpiente
        for segment in self.snake[1:]:
            grid_img[segment[0], segment[1]] = self.COLOR_SNAKE_BODY

        # Dibujar cabeza de la serpiente
        head = self.snake[0]
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            grid_img[head[0], head[1]] = self.COLOR_SNAKE_HEAD

        # Escalado a resolución final de entrada (img_size x img_size) usando OpenCV Nearest Neighbor
        pixel_obs = cv2.resize(
            grid_img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return pixel_obs

    def _get_info(self) -> Dict[str, Any]:
        """Devuelve información adicional de depuración."""
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps_taken,
        }

    def render(self) -> Optional[np.ndarray]:
        """Renderiza el estado visual del entorno."""
        obs = self._get_observation()
        if self.render_mode == "human":
            # Convertir de RGB a BGR para OpenCV imshow
            bgr_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            # Redimensionar para ventana visible cómoda
            display_img = cv2.resize(bgr_img, (400, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("AutoRL Snake - Visual Feed", display_img)
            cv2.waitKey(1)
            return None
        return obs

    def close(self) -> None:
        """Cierra recursos y ventanas abiertas."""
        if self.render_mode == "human":
            cv2.destroyAllWindows()
