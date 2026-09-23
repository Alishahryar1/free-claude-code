"""Async iterator lifecycle helper contracts."""

import asyncio

import pytest

from free_claude_code.core.async_iterators import (
    AsyncCloseable,
    try_close_async_iterator,
)


class _Closeable:
    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._error is not None:
            raise self._error


@pytest.mark.asyncio
async def test_try_close_async_iterator_closes_supported_value_once() -> None:
    value = _Closeable()

    assert isinstance(value, AsyncCloseable)
    assert await try_close_async_iterator(value) is None
    assert value.close_calls == 1


@pytest.mark.asyncio
async def test_try_close_async_iterator_ignores_non_closeable_value() -> None:
    assert await try_close_async_iterator(object()) is None


@pytest.mark.asyncio
async def test_try_close_async_iterator_returns_ordinary_close_failure() -> None:
    failure = RuntimeError("close failed")
    value = _Closeable(failure)

    assert await try_close_async_iterator(value) is failure
    assert value.close_calls == 1


@pytest.mark.asyncio
async def test_try_close_async_iterator_propagates_cancellation() -> None:
    value = _Closeable(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await try_close_async_iterator(value)

    assert value.close_calls == 1
