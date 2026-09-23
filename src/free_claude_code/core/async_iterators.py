"""Minimal lifecycle helpers for composed asynchronous iterators."""

from typing import Protocol, runtime_checkable


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
