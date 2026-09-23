"""Atomic JSON persistence for messaging session state."""

import contextlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class _PendingWrite:
    generation: int
    snapshot: dict[str, Any]


class DebouncedJsonPersistence:
    """Thread-safe debounced JSON writer with atomic replace semantics."""

    def __init__(
        self,
        storage_path: str,
        *,
        snapshot: Callable[[], dict[str, Any]],
        on_dirty: Callable[[bool], None],
        debounce_secs: float = 0.5,
    ) -> None:
        self.storage_path = storage_path
        self._snapshot = snapshot
        self._on_dirty = on_dirty
        self._debounce_secs = debounce_secs
        self._save_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()
        self._writer_lock = threading.Lock()
        self._save_generation = 0

    def load_json(self) -> dict[str, Any]:
        if not os.path.exists(self.storage_path):
            return {}
        with open(self.storage_path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def schedule_save(self) -> None:
        self._on_dirty(True)
        with self._timer_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
            self._save_generation += 1
            generation = self._save_generation
            timer = threading.Timer(
                self._debounce_secs,
                self._save_from_timer,
                args=(generation,),
            )
            timer.daemon = True
            self._save_timer = timer
        timer.start()

    def flush(self) -> None:
        self._on_dirty(True)
        pending = self._snapshot_for_write()
        if pending is None:
            return
        self._write_pending(pending)

    def _save_from_timer(self, generation: int) -> None:
        try:
            pending = self._snapshot_for_write(expected_generation=generation)
            if pending is None:
                return
            self._write_pending(pending)
        except Exception as e:
            self._on_dirty(True)
            logger.error(
                "Failed to save sessions: exc_type={}",
                type(e).__name__,
            )

    def _write_pending(self, pending: _PendingWrite) -> None:
        try:
            written = self._write_if_current(pending)
        except Exception:
            self._on_dirty(True)
            raise
        if written:
            self._mark_clean_if_current(pending.generation)

    def _write_if_current(self, pending: _PendingWrite) -> bool:
        """Serialize writers and reject a snapshot superseded before replace."""
        with self._writer_lock:
            with self._timer_lock:
                if pending.generation != self._save_generation:
                    return False
            self._write_file(pending.snapshot)
            return True

    def _snapshot_for_write(
        self, *, expected_generation: int | None = None
    ) -> _PendingWrite | None:
        generation = self._claim_timer(expected_generation)
        if generation is None:
            return None
        snapshot = self._snapshot()
        return _PendingWrite(generation=generation, snapshot=snapshot)

    def _claim_timer(self, expected_generation: int | None) -> int | None:
        with self._timer_lock:
            if expected_generation is not None and (
                expected_generation != self._save_generation or self._save_timer is None
            ):
                return None
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            return self._save_generation

    def _mark_clean_if_current(self, generation: int) -> None:
        with self._timer_lock:
            is_current = (
                self._save_timer is None and generation == self._save_generation
            )
        if is_current:
            self._on_dirty(False)

    def write_data(self, data: dict[str, Any]) -> None:
        """Write authoritative state after invalidating older pending snapshots."""
        self._on_dirty(True)
        with self._timer_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            self._save_generation += 1
            pending = _PendingWrite(
                generation=self._save_generation,
                snapshot=data,
            )
        self._write_pending(pending)

    def _write_file(self, data: dict[str, Any]) -> None:
        abs_target = os.path.abspath(self.storage_path)
        dir_name = os.path.dirname(abs_target) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name,
            prefix=".sessions.",
            suffix=".tmp.json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, abs_target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
