"""Request-local evidence of public stream delivery, independent of wire format."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from .stream_recovery import DeliveredPrefix


class StreamDeliveryState:
    """Exchange attempt transitions and immutable public-delivery evidence."""

    def __init__(self) -> None:
        self._content_released = False
        self._attempt_content_released = False
        self._attempt_revision = 0
        self._prefix_reader: Callable[[], DeliveredPrefix] | None = None
        self.transition: Literal["continue", "handoff"] | None = None

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
        return self._prefix_reader() if self._prefix_reader is not None else None

    def bind_prefix(self, reader: Callable[[], DeliveredPrefix] | None) -> None:
        self._prefix_reader = reader

    def release_content(self, *, synthetic: bool = False) -> None:
        self._content_released = True
        if not synthetic:
            self._attempt_content_released = True

    def begin_attempt(self) -> None:
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
def bind_stream_delivery(state: StreamDeliveryState | None) -> Iterator[None]:
    """Bind only for an iterator operation, or mask a private stream consumer."""
    token = _delivery.set(state)
    try:
        yield
    finally:
        _delivery.reset(token)
