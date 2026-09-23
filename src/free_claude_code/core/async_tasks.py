"""A worker boundary that retains ownership through caller cancellation."""

import asyncio
from collections.abc import Callable

from anyio import to_thread


async def run_sync_owned[T](function: Callable[[], T]) -> T:
    """Drain a finite worker before propagating cancellation; never abandon it."""
    worker = asyncio.create_task(to_thread.run_sync(function))
    cancellation: asyncio.CancelledError | None = None
    while not worker.done():
        try:
            await asyncio.wait({worker})
        except asyncio.CancelledError as exc:
            cancellation = exc
    try:
        result = worker.result()
    except BaseException:
        if cancellation is not None:
            raise cancellation from None
        raise
    if cancellation is not None:
        raise cancellation
    return result
