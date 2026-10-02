"""
Automatic Game Window Finder Module for Windows OS (Phase 3 Fix)
Locates target game window (e.g., Clicker Heroes), extracts bounding box coordinates,
and brings the window to the foreground safely before input interaction.
"""
import ctypes
from ctypes import wintypes
import sys
from typing import Dict, Optional, List, Tuple


class WindowNotFoundError(RuntimeError):
    """Raised when the target game window is not found."""
    pass


def list_visible_windows() -> List[Tuple[int, str]]:
    """Returns a list of tuples (hwnd, title) for all visible top-level windows."""
    if sys.platform != "win32":
        return []

    user32 = ctypes.windll.user32
    windows = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def enum_proc(hwnd, lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            if title and title.strip():
                windows.append((hwnd, title))
        return True


    user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
    return windows


def find_game_window(
    title_substring: str = "Idle Slayer",
    bring_to_front: bool = True,
) -> Dict[str, int]:
    """
    Locates a visible window matching title_substring (case-insensitive).
    :param title_substring: Part of the window title to search for.
    :param bring_to_front: If True, focuses and restores the window.
    :return: Region dict {'top': y, 'left': x, 'width': w, 'height': h}.
    :raises WindowNotFoundError: If no matching window is found.
    """
    if sys.platform != "win32":
        # Non-windows fallback
        return {"top": 0, "left": 0, "width": 1280, "height": 720}

    user32 = ctypes.windll.user32
    visible_windows = list_visible_windows()

    target_hwnd = None
    matched_title = ""

    # Lista de términos a buscar (título especificado + fallbacks habituales)
    search_terms = [title_substring.lower()]
    if "idle slayer" not in search_terms:
        search_terms.append("idle slayer")
    if "clicker heroes" not in search_terms:
        search_terms.append("clicker heroes")

    for term in search_terms:
        for hwnd, title in visible_windows:
            if term in title.lower():
                target_hwnd = hwnd
                matched_title = title
                break
        if target_hwnd is not None:
            break

    if target_hwnd is None:
        # Fallback: Comprobar si el proceso de Clicker Heroes está en ejecución
        import subprocess
        try:
            out = subprocess.check_output('tasklist /FI "IMAGENAME eq Clicker Heroes.exe"', shell=True).decode('latin1')
            if "Clicker Heroes" in out:
                w = user32.GetSystemMetrics(0) or 2560
                h = user32.GetSystemMetrics(1) or 1440
                print(f"\n[VENTANA DETECTADA] Proceso: 'Clicker Heroes.exe' (Pantalla Completa: {w}x{h} px)")
                return {"top": 0, "left": 0, "width": int(w), "height": int(h)}
        except Exception:
            pass

        titles_list = "\n".join([f"  - {t}" for _, t in visible_windows[:10]])
        raise WindowNotFoundError(
            f"\n[ERROR CRITICO] No se encontro ninguna ventana visible que contenga '{title_substring}'.\n"
            f"Asegurate de que 'Idle Slayer' o 'Clicker Heroes' este abierto en tu ordenador antes de ejecutar en vivo.\n"
            f"Algunas ventanas abiertas detectadas:\n{titles_list}"
        )



    if bring_to_front:
        # Restore if minimized (SW_RESTORE = 9)
        user32.ShowWindow(target_hwnd, 9)
        user32.SetForegroundWindow(target_hwnd)

    # Get window rect
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    user32.GetWindowRect(target_hwnd, ctypes.byref(rect))

    width = rect.right - rect.left
    height = rect.bottom - rect.top

    region = {
        "top": int(rect.top),
        "left": int(rect.left),
        "width": int(width),
        "height": int(height),
    }

    print(f"\n[VENTANA DETECTADA] Titulo: '{matched_title}'")
    print(f"Ubicacion: (X={region['left']}, Y={region['top']}), Tamano: {region['width']}x{region['height']} px")

    return region
