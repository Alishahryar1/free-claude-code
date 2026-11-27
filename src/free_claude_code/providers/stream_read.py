"""Bound one upstream event read without letting keep-alives renew the wait."""

import asyncio
from collections.abc import AsyncIterator, Callable


async def read_stream_event[EventT](
    source: AsyncIterator[EventT],
    *,
    timeout: float,
    is_keepalive: Callable[[EventT], bool],
) -> EventT:
    async with asyncio.timeout(timeout):
        while True:
            event = await anext(source)
            if not is_keepalive(event):
                return event
            # Buffered heartbeat floods must remain cancellable too.
            await asyncio.sleep(0)
