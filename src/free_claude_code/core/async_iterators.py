"""Minimal lifecycle helpers for composed asynchronous iterators."""

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

_CLEANUP_GRACE_SECONDS = 5.0


class CleanupBudget:
    """Share one fresh cleanup allowance across a resource and its nested owners."""

    def __init__(self) -> None:
        self._deadline: float | None = None

    @contextmanager
    def bind(self) -> Iterator[None]:
        token = _cleanup_budget.set(self)
        try:
            yield
        finally:
            _cleanup_budget.reset(token)

    async def run[T](self, operation: Awaitable[T]) -> T:
        if self._deadline is None:
            self._deadline = asyncio.get_running_loop().time() + _CLEANUP_GRACE_SECONDS

        async def bounded() -> T:
            with self.bind():
                async with asyncio.timeout_at(self._deadline):
                    return await operation

        return await complete_cleanup(bounded())


_cleanup_budget: ContextVar[CleanupBudget | None] = ContextVar(
    "provider_cleanup_budget", default=None
)


def current_cleanup_budget() -> CleanupBudget:
    return _cleanup_budget.get() or CleanupBudget()


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
