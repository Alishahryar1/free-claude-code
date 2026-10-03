"""Request-local evidence of public stream delivery, independent of wire format."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class StreamDeliveryState:
    """Retain public commitment while identifying invisible generation attempts."""

    def __init__(self) -> None:
        self._content_released = False
        self._attempt_revision = 0

    @property
    def content_released(self) -> bool:
        return self._content_released

    @property
    def attempt_revision(self) -> int:
        return self._attempt_revision

    def release_content(self) -> None:
        self._content_released = True

    def begin_attempt(self) -> None:
        if not self._content_released:
            self._attempt_revision += 1


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
