"""Drain Copilot-owned work before propagating caller cancellation."""

import asyncio


async def drain_owned[T](task: asyncio.Task[T]) -> T:
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
