"""Adapt the OpenAI SDK stream to FCC's asynchronous cleanup contract."""

from collections.abc import AsyncIterator, Callable

from openai import AsyncStream

from .stream_read import read_stream_event


def _is_keepalive(event: object) -> bool:
    return getattr(event, "type", None) == "ping"


class OpenAIStreamAdapter[EventT](AsyncIterator[EventT]):
    """Own an SDK response stream without closing its reusable client."""

    def __init__(
        self,
        stream: AsyncStream[EventT],
        *,
        read_timeout_s: float,
        on_event: Callable[[EventT], None] | None = None,
    ) -> None:
        self._stream = stream
        self._read_timeout_s = read_timeout_s
        self._on_event = on_event

    def __aiter__(self) -> AsyncIterator[EventT]:
        return self

    async def __anext__(self) -> EventT:
        event = await read_stream_event(
            self._stream, timeout=self._read_timeout_s, is_keepalive=_is_keepalive
        )
        if self._on_event is not None:
            self._on_event(event)
        return event

    async def aclose(self) -> None:
        await self._stream.close()
