"""Request-local evidence of public stream delivery, independent of wire format."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Literal

from .history_replay import ReplayRecord, encode_replay
from .stream_recovery import RECOVERY_RETENTION_BYTES, DeliveredPrefix


class StreamDeliveryState:
    """Exchange attempt transitions and immutable public-delivery evidence."""

    def __init__(self) -> None:
        self._content_released = False
        self._attempt_content_released = False
        self._attempt_revision = 0
        self._prefix_reader: Callable[[], DeliveredPrefix] | None = None
        self.transition: Literal["continue", "handoff"] | None = None
        self._replay_source: Callable[[], tuple[ReplayRecord, ...]] | None = None
        self._committed_replay: tuple[ReplayRecord, ...] = ()
        self._replay_bytes = 0
        self.replay_exhausted = False

    @property
    def content_released(self) -> bool:
        return self._content_released

    @property
    def attempt_revision(self) -> int:
        return self._attempt_revision

    @property
    def attempt_content_released(self) -> bool:
        return self._attempt_content_released

    @property
    def prefix(self) -> DeliveredPrefix | None:
        prefix = self._prefix_reader() if self._prefix_reader is not None else None
        if prefix is None:
            return None
        current = self._visible_replay()
        size = self._replay_bytes + sum(
            len(encode_replay(record)) for record in current
        )
        limited = (
            self.replay_exhausted
            or size + prefix.retained_bytes > RECOVERY_RETENTION_BYTES
        )
        return replace(
            prefix,
            replay_records=() if limited else self._committed_replay + current,
            eligible=prefix.eligible and not limited,
            can_handoff=prefix.can_handoff and not limited,
            unsafe_reason="retention_limit" if limited else prefix.unsafe_reason,
        )

    @property
    def committed_replay(self) -> tuple[ReplayRecord, ...]:
        return self._committed_replay

    def bind_replay_source(
        self, reader: Callable[[], tuple[ReplayRecord, ...]]
    ) -> None:
        """Expose completed native records without publishing an invisible attempt."""
        self._replay_source = reader

    def _visible_replay(self) -> tuple[ReplayRecord, ...]:
        return (
            self._replay_source()
            if self._attempt_content_released and self._replay_source
            else ()
        )

    def bind_prefix(self, reader: Callable[[], DeliveredPrefix] | None) -> None:
        self._prefix_reader = reader
        if reader is None:
            self._replay_source = None
            self._committed_replay = ()
            self._replay_bytes = 0

    def release_content(self, *, synthetic: bool = False) -> None:
        self._content_released = True
        if not synthetic:
            self._attempt_content_released = True

    def begin_attempt(self) -> None:
        records = self._visible_replay()
        self._replay_bytes += sum(len(encode_replay(record)) for record in records)
        if self._replay_bytes > RECOVERY_RETENTION_BYTES:
            self.replay_exhausted = True
            self._committed_replay = ()
        elif not self.replay_exhausted:
            self._committed_replay += records
        self._replay_source = None
        self._attempt_revision += 1
        self._attempt_content_released = False

    def begin_continuation(self, *, handoff: bool = False) -> None:
        self.transition = "handoff" if handoff else "continue"


_delivery: ContextVar[StreamDeliveryState | None] = ContextVar(
    "stream_delivery", default=None
)


def current_stream_delivery() -> StreamDeliveryState | None:
    return _delivery.get()


@contextmanager
def bind_stream_delivery(state: StreamDeliveryState | None) -> Generator[None]:
    """Bind only for an iterator operation, or mask a private stream consumer."""
    token = _delivery.set(state)
    try:
        yield
    finally:
        _delivery.reset(token)
