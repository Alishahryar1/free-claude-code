"""Windows scaling setup shared by native UI entrypoints."""

import ctypes
import sys

_PER_MONITOR_AWARE_V2 = -4
_ERROR_ACCESS_DENIED = 5


def enable_dpi_awareness() -> None:
    """Use native monitor scaling before creating windows in this process."""
    if sys.platform != "win32":
        return

    set_awareness = ctypes.WinDLL(
        "user32", use_last_error=True
    ).SetProcessDpiAwarenessContext
    set_awareness.argtypes = (ctypes.c_void_p,)
    set_awareness.restype = ctypes.c_int
    if set_awareness(ctypes.c_void_p(_PER_MONITOR_AWARE_V2)):
        return

    error = ctypes.get_last_error()
    # Windows rejects subsequent changes, including a mode set by the host's
    # manifest. Respect that choice and allow repeated initialization.
    if error != _ERROR_ACCESS_DENIED:
        raise ctypes.WinError(error)
