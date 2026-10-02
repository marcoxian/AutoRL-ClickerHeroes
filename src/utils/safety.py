"""
Emergency Stop & Safety Listener Module for Game AI (Phase 3 Fix)
Uses a background daemon thread to detect single-tap ESC keypresses asynchronously,
raising an EmergencyStopException to immediately halt mouse input and save model checkpoints.
"""
import ctypes
import os
import sys
import threading
import time
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

# Virtual-Key Codes for Windows API
VK_ESCAPE = 0x1B  # ESC Key
VK_P = 0x50       # P Key


class EmergencyStopException(BaseException):
    """Exception raised immediately when Emergency Stop (ESC key) is triggered."""
    pass


class EmergencyStopListener:
    """
    Asynchronous background thread listener for single-tap ESC key detection and P key for pausing.
    """

    def __init__(self, stop_key: int = VK_ESCAPE, pause_key: int = VK_P, check_corner: bool = True, start_daemon: bool = True):
        self.stop_key = stop_key
        self.pause_key = pause_key
        self.check_corner = check_corner
        self.stop_requested = False
        self.paused = False
        self._pause_key_held = False
        self.is_windows = (sys.platform == "win32")
        self._running = True

        if start_daemon and self.is_windows:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def _poll_loop(self) -> None:
        """Background loop polling GetAsyncKeyState every 10ms."""
        user32 = ctypes.windll.user32
        while self._running:
            try:
                # Check ESC key
                state = user32.GetAsyncKeyState(self.stop_key)
                if state & 0x8000:
                    self.stop_requested = True

                # Check P key for pause
                state_p = user32.GetAsyncKeyState(self.pause_key)
                if state_p & 0x8000:
                    if not self._pause_key_held:
                        self.paused = not self.paused
                        self._pause_key_held = True
                        if self.paused:
                            print("\n⏸️ [PAUSA] Entrenamiento pausado. El ratón está libre. Pulsa 'P' de nuevo para reanudar.")
                        else:
                            print("\n▶️ [REANUDAR] Entrenamiento reanudado.")
                else:
                    self._pause_key_held = False

                # Check safety corner if enabled
                if self.check_corner and not self.stop_requested:
                    class POINT(ctypes.Structure):
                        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                    pt = POINT()
                    user32.GetCursorPos(ctypes.byref(pt))
                    if pt.x <= 10 and pt.y <= 10:
                        self.stop_requested = True
            except Exception:
                pass
            time.sleep(0.01)

    def request_stop(self) -> None:
        """Manually trigger emergency stop flag."""
        self.stop_requested = True

    def reset(self) -> None:
        """Reset emergency stop flag."""
        self.stop_requested = False

    def check_stop_requested(self) -> bool:
        """Returns True if stop has been triggered."""
        return self.stop_requested

    def raise_if_stopped(self) -> None:
        """Raises EmergencyStopException if stop requested. Blocks if paused."""
        if self.stop_requested:
            raise EmergencyStopException("[PARADA DE EMERGENCIA] Tecla ESC detectada. Deteniendo ejecucion.")
            
        # Si está pausado, bloqueamos la ejecución aquí de forma segura
        while getattr(self, 'paused', False) and not self.stop_requested:
            time.sleep(0.1)
            
        if self.stop_requested:
            raise EmergencyStopException("[PARADA DE EMERGENCIA] Tecla ESC detectada. Deteniendo ejecucion.")

    def close(self) -> None:
        """Stop background thread."""
        self._running = False


# Shared global instance
global_emergency_listener = EmergencyStopListener(check_corner=False)


class EmergencyStopCallback(BaseCallback):
    """
    Stable-Baselines3 Callback that halts training immediately if emergency stop is triggered.
    """

    def __init__(
        self,
        listener: Optional[EmergencyStopListener] = None,
        save_dir: str = "models",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.listener = listener or global_emergency_listener
        self.save_dir = save_dir
        self.stop_triggered = False

    def _on_step(self) -> bool:
        """
        Called after every step in the environment.
        Raises EmergencyStopException or returns False to interrupt SB3.
        """
        if self.listener.check_stop_requested():
            self.stop_triggered = True
            if self.verbose > 0:
                print("\n" + "=" * 60)
                print("[PARADA DE EMERGENCIA DETECTADA] (Tecla ESC presionada)")
                print("Deteniendo entrenamiento y congelando acciones de raton...")
                print("=" * 60 + "\n")

            if self.model is not None:
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, "clicker_heroes_emergency_save.zip")
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"[GUARDADO] Punto de control guardado en: {save_path}")

            # Raise exception to break out instantly from nested loops
            self.listener.raise_if_stopped()
            return False

        return True
