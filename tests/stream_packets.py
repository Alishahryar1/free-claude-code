"""Explicit provider packets for tests whose fixtures are serialized text frames."""

from collections.abc import AsyncIterator

from free_claude_code.core.async_iterators import try_close_async_iterator
from free_claude_code.core.stream_output import StreamOutput


class PacketFixtureStream(AsyncIterator[StreamOutput]):
    def __init__(self, source: AsyncIterator[str]) -> None:
        self.source = source

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamOutput:
        return StreamOutput((await anext(self.source),))

    async def aclose(self) -> None:
        error = await try_close_async_iterator(self.source)
        if error is not None:
            raise error


def packets(source: AsyncIterator[str]) -> PacketFixtureStream:
    return PacketFixtureStream(source)
