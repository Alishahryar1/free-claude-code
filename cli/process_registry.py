"""Track and clean up spawned CLI subprocesses.

This is a safety net for cases where the server is interrupted (Ctrl+C) and the
FastAPI lifespan cleanup doesn't run to completion. We only track processes we
spawn so we don't accidentally kill unrelated system processes.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import threading
from typing import Set
from loguru import logger


_lock = threading.Lock()
_pids: Set[int] = set()
_atexit_registered = False


def ensure_atexit_registered() -> None:
    global _atexit_registered
    with _lock:
        if _atexit_registered:
            return
        atexit.register(kill_all_best_effort)
        _atexit_registered = True


def register_pid(pid: int) -> None:
    if not pid:
        return
    ensure_atexit_registered()
    with _lock:
        _pids.add(int(pid))


def unregister_pid(pid: int) -> None:
    if not pid:
        return
    with _lock:
        _pids.discard(int(pid))


def kill_all_best_effort() -> None:
    """Kill any still-running registered pids (best-effort)."""
    with _lock:
        pids = list(_pids)
        _pids.clear()

    if not pids:
        return

    if os.name == "nt":
        for pid in pids:
            try:
                # /T kills child processes, /F forces termination.
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception as e:
                logger.debug("process_registry: taskkill failed pid=%s: %s", pid, e)
        return

    # Best-effort fallback for non-Windows.
    for pid in pids:
        try:
            os.kill(pid, 9)
        except Exception as e:
            logger.debug("process_registry: kill failed pid=%s: %s", pid, e)
