"""Small cross-platform advisory file lock."""

import os
import time
from pathlib import Path
from typing import BinaryIO

if os.name == "nt":
    import msvcrt

    def _try_lock(handle: BinaryIO) -> bool:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: BinaryIO) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class InterprocessFileLock:
    """Advisory lock released automatically if its process exits."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: BinaryIO | None = None

    def acquire(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        poll_interval: float = 0.05,
    ) -> bool:
        """Acquire the lock, optionally waiting for a bounded interval."""

        if self._handle is not None:
            return True
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        deadline = None if timeout is None else time.monotonic() + timeout
        while not _try_lock(handle):
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                handle.close()
                return False
            time.sleep(poll_interval)
        self._handle = handle
        return True

    def release(self) -> None:
        """Release a held lock; repeated calls are harmless."""

        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> InterprocessFileLock:
        if not self.acquire(wait=True):
            raise TimeoutError(f"Could not acquire file lock: {self._path}")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
