"""Adapt the OpenAI SDK stream to FCC's asynchronous cleanup contract."""

from collections.abc import AsyncIterator, Callable

from openai import AsyncStream


class OpenAIStreamAdapter[EventT](AsyncIterator[EventT]):
    """Own an SDK response stream without closing its reusable client."""

    def __init__(
        self,
        stream: AsyncStream[EventT],
        *,
        on_event: Callable[[EventT], None] | None = None,
    ) -> None:
        self._stream = stream
        self._on_event = on_event

    def __aiter__(self) -> AsyncIterator[EventT]:
        return self

    async def __anext__(self) -> EventT:
        event = await anext(self._stream)
        if self._on_event is not None:
            self._on_event(event)
        return event

    async def aclose(self) -> None:
        await self._stream.close()
