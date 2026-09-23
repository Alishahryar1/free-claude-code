"""Direct tests for :class:`core.rate_limit.StrictSlidingWindowLimiter`."""

import asyncio
import time

import pytest

import free_claude_code.core.rate_limit as rate_limit_module
from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter


@pytest.mark.asyncio
async def test_strict_window_allows_burst_then_blocks():
    lim = StrictSlidingWindowLimiter(rate_limit=2, rate_window=0.2)
    await lim.acquire()
    await lim.acquire()
    start = time.monotonic()
    await lim.acquire()
    assert time.monotonic() - start >= 0.15


@pytest.mark.asyncio
async def test_strict_window_async_context_manager():
    lim = StrictSlidingWindowLimiter(rate_limit=1, rate_window=0.15)

    async def run():
        async with lim:
            pass

    await run()
    start = time.monotonic()
    await run()
    assert time.monotonic() - start >= 0.1


@pytest.mark.asyncio
async def test_rejected_conditional_acquisition_does_not_consume_capacity():
    lim = StrictSlidingWindowLimiter(rate_limit=1, rate_window=60)

    assert await lim.acquire_if(lambda: False) is False

    await asyncio.wait_for(lim.acquire(), timeout=0.1)


@pytest.mark.asyncio
async def test_conditional_acquisition_records_predicate_commit_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    sleep_delays: list[float] = []
    lim = StrictSlidingWindowLimiter(rate_limit=1, rate_window=10)

    def advance_during_condition() -> bool:
        nonlocal now
        now = 100.0
        return True

    async def advance_during_sleep(delay: float) -> None:
        nonlocal now
        sleep_delays.append(delay)
        now += delay

    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(rate_limit_module.asyncio, "sleep", advance_during_sleep)

    assert await lim.acquire_if(advance_during_condition) is True
    await lim.acquire()

    assert sleep_delays == [10.0]


def test_strict_window_rejects_invalid_config():
    with pytest.raises(ValueError):
        StrictSlidingWindowLimiter(rate_limit=0, rate_window=1.0)
    with pytest.raises(ValueError):
        StrictSlidingWindowLimiter(rate_limit=1, rate_window=0.0)
