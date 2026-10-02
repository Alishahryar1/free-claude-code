"""Minimal lifecycle helpers for composed asynchronous iterators."""

import asyncio
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable


async def complete_cleanup[T](operation: Awaitable[T]) -> T:
    """Drain owned cleanup before propagating even repeated caller cancellation."""
    task = asyncio.ensure_future(operation)
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


@runtime_checkable
class AsyncCloseable(Protocol):
    """An object whose asynchronous iteration resources can be released."""

    async def aclose(self) -> None: ...


async def try_close_async_iterator(value: object) -> Exception | None:
    """Close ``value`` when supported, returning ordinary cleanup failures.

    Cancellation remains control flow and propagates to the caller. Returning
    ordinary exceptions lets an owner observe cleanup failure without replacing
    the stream outcome that was already established.
    """
    if not isinstance(value, AsyncCloseable):
        return None
    try:
        await value.aclose()
    except Exception as exc:
        return exc
    return None
