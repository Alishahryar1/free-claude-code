import subprocess
import sys

import pytest


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_platforms_do_not_load_windows_ui(monkeypatch, platform):
    from free_claude_code.core import windows_dpi

    def load_windows(*_args, **_kwargs):
        pytest.fail("Windows UI library loaded on another platform")

    monkeypatch.setattr(windows_dpi.sys, "platform", platform)
    monkeypatch.setattr(windows_dpi.ctypes, "WinDLL", load_windows, raising=False)
    windows_dpi.enable_dpi_awareness()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPI APIs")
@pytest.mark.parametrize("preset", ["none", "system", "per-monitor"])
def test_native_windows_use_scaling_and_respect_existing_awareness(preset):
    # DPI awareness is process-wide and can only be set once. Each case needs
    # its own process so other tests and the test runner keep their own mode.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes
import sys
from ctypes import wintypes

user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
user32.GetWindowDpiAwarenessContext.argtypes = [wintypes.HWND]
user32.GetWindowDpiAwarenessContext.restype = ctypes.c_void_p
user32.AreDpiAwarenessContextsEqual.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.AreDpiAwarenessContextsEqual.restype = wintypes.BOOL
expected = -2 if sys.argv[1] == 'system' else -4
if sys.argv[1] != 'none':
    assert user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(expected))

from free_claude_code.core.windows_dpi import enable_dpi_awareness
enable_dpi_awareness()
enable_dpi_awareness()

import tkinter
root = tkinter.Tk()
root.withdraw()
try:
    context = user32.GetWindowDpiAwarenessContext(root.winfo_id())
    assert user32.AreDpiAwarenessContextsEqual(context, ctypes.c_void_p(expected))
finally:
    root.destroy()
""",
            preset,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
