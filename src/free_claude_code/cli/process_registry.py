"""Track and clean up spawned CLI subprocesses.

This is a safety net for cases where the server is interrupted (Ctrl+C) and the
FastAPI lifespan cleanup doesn't run to completion. We only track processes we
spawn so we don't accidentally kill unrelated system processes.
"""

import atexit
import os
import signal
import subprocess
import threading

from loguru import logger

_lock = threading.Lock()
_pids: set[int] = set()
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


def kill_pid_tree_best_effort(pid: int) -> None:
    """Kill a tracked process and its children where the platform supports it."""
    if not pid:
        return
    if os.name == "nt":
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
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        logger.debug("process_registry: terminate failed pid=%s: %s", pid, e)


def kill_all_best_effort() -> None:
    """Kill any still-running registered pids (best-effort)."""
    with _lock:
        pids = list(_pids)
        _pids.clear()

    for pid in pids:
        kill_pid_tree_best_effort(pid)
