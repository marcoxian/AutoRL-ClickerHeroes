"""
High-Performance Screen Capture Module for Game AI (Phase 2 & 3)
Uses `mss` for high FPS window/region capture with OpenCV processing and automatic window detection.
"""
import time
from typing import Dict, Optional, Tuple, Union
import cv2
import mss
import numpy as np

from src.utils.window_finder import find_game_window


class ScreenCapture:
    """
    High-speed screen capture handler using mss.
    Automatically binds to Clicker Heroes window in live mode.
    """

    def __init__(
        self,
        region: Optional[Dict[str, int]] = None,
        target_size: Tuple[int, int] = (84, 84),
        channel_first: bool = True,
        mock_mode: bool = False,
        window_title: str = "Clicker Heroes",
    ):
        """
        :param region: Dict with keys 'top', 'left', 'width', 'height'. If None in live mode, auto-detects window_title.
        :param target_size: (width, height) to resize captured frames for CNN input.
        :param channel_first: If True, returns shape (C, H, W) for PyTorch. If False, returns (H, W, C).
        :param mock_mode: If True, generates synthetic frames for offline testing.
        :param window_title: Title substring to auto-detect if region is None.
        """
        self.target_size = target_size
        self.channel_first = channel_first
        self.mock_mode = mock_mode
        self.sct = None if mock_mode else mss.mss()

        if region is None:
            if not mock_mode:
                # Auto-detect target game window
                self.region = find_game_window(title_substring=window_title, bring_to_front=True)
            else:
                self.region = {"top": 0, "left": 0, "width": 1280, "height": 720}
        else:
            self.region = region

    def set_region(self, top: int, left: int, width: int, height: int) -> None:
        """Update capture region coordinates."""
        self.region = {"top": top, "left": left, "width": width, "height": height}

    def capture_raw(self) -> np.ndarray:
        """Captures raw frame in BGRA numpy format."""
        if self.mock_mode:
            return self._generate_mock_frame()

        sct_img = self.sct.grab(self.region)
        return np.array(sct_img)

    def capture_raw_rgb(self) -> np.ndarray:
        """Captura el fotograma a resolución nativa completa en formato RGB (sin redimensionar)."""
        raw = self.capture_raw()
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)


    def capture_frame(self) -> np.ndarray:
        """
        Captures screen region, converts BGRA -> RGB, resizes (if target_size set), and optionally transposes to (C, H, W).
        :return: np.ndarray uint8 image.
        """
        raw = self.capture_raw()
        # Convert BGRA to RGB
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

        # Resize to target size (width, height) if specified
        if self.target_size is not None:
            resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = rgb

        if self.channel_first:
            # (H, W, C) -> (C, H, W)
            return np.transpose(resized, (2, 0, 1))

        return resized


    def _generate_mock_frame(self) -> np.ndarray:
        """Generates a synthetic Clicker Heroes-like screen frame for testing."""
        w, h = self.region.get("width", 1280), self.region.get("height", 720)
        frame = np.zeros((h, w, 4), dtype=np.uint8)

        # Background gradient
        frame[:, :, 0] = 50  # Blue
        frame[:, :, 1] = 30  # Green
        frame[:, :, 2] = 20  # Red
        frame[:, :, 3] = 255

        # Draw a mock monster health bar (green/red bar at top-middle)
        bar_x1, bar_y1, bar_x2, bar_y2 = int(w * 0.35), int(h * 0.15), int(w * 0.65), int(h * 0.18)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 0, 255, 255), -1)  # Red background
        # 70% current health
        hp_x2 = int(bar_x1 + (bar_x2 - bar_x1) * 0.7)
        cv2.rectangle(frame, (bar_x1, bar_y1), (hp_x2, bar_y2), (0, 255, 0, 255), -1)  # Green HP

        # Draw a mock monster in the center
        cv2.circle(frame, (w // 2, h // 2), 60, (200, 100, 50, 255), -1)

        return frame

    def close(self) -> None:
        """Release screen capture resources."""
        if self.sct is not None:
            self.sct.close()


def benchmark_capture(num_frames: int = 100, mock: bool = True) -> float:
    """
    Benchmarks screen capture FPS.
    :return: FPS (frames per second).
    """
    cap = ScreenCapture(target_size=(84, 84), mock_mode=mock)
    start_time = time.time()
    for _ in range(num_frames):
        _ = cap.capture_frame()
    elapsed = time.time() - start_time
    fps = num_frames / elapsed if elapsed > 0 else 0
    print(f"Benchmark ({'Mock' if mock else 'Real'}): {num_frames} frames in {elapsed:.4f}s -> {fps:.2f} FPS")
    cap.close()
    return fps
