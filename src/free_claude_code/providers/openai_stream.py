"""Adapt the OpenAI SDK stream to FCC's asynchronous cleanup contract."""

from collections.abc import AsyncIterator

from openai import AsyncStream


class OpenAIStreamAdapter[EventT](AsyncIterator[EventT]):
    """Own an SDK response stream without closing its reusable client."""

    def __init__(self, stream: AsyncStream[EventT]) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[EventT]:
        return self

    async def __anext__(self) -> EventT:
        return await anext(self._stream)

    async def aclose(self) -> None:
        await self._stream.close()
