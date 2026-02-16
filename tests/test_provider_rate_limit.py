import pytest
import pytest_asyncio
import asyncio
import time

from providers.rate_limit import GlobalRateLimiter


class TestProviderRateLimiter:
    """Tests for providers.rate_limit.GlobalRateLimiter."""

    @pytest_asyncio.fixture(autouse=True)
    async def reset_limiter(self):
        """Reset singleton before each test."""
        GlobalRateLimiter.reset_instance()
        yield
        GlobalRateLimiter.reset_instance()

    @pytest.mark.asyncio
    async def test_proactive_throttling(self):
        """
        Test proactive throttling.
        Logic ported from verify_provider_limiter.py
        """
        # Re-init with tight limits: 1 request per 0.25 second
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=1, rate_window=0.25)

        start_time = time.time()

        async def call_limiter():
            await limiter.wait_if_blocked()
            return time.time()

        # 5 requests.
        # R0 -> 0s
        # R1 -> 0.25s
        # R2 -> 0.50s
        # R3 -> 0.75s
        # R4 -> 1.00s
        results = []
        for _ in range(5):
            results.append(await call_limiter())

        total_time = time.time() - start_time

        assert len(results) == 5
        # Should take at least ~1.0s
        assert total_time >= 0.9, f"Throttling failed, took too fast: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_reactive_blocking(self):
        """
        Test reactive blocking when set_blocked is called.
        Logic ported from verify_provider_limiter.py
        """
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance()

        start_time = time.time()

        # Manually block for 1.5s
        block_time = 1.5
        limiter.set_blocked(block_time)

        assert limiter.is_blocked()

        async def call_limiter():
            return await limiter.wait_if_blocked()

        # Run 2 calls concurrently
        # They should both wait for the block time
        results = await asyncio.gather(call_limiter(), call_limiter())

        total_time = time.time() - start_time

        # Both should report having waited reactively
        assert all(results) is True
        assert total_time >= block_time - 0.1, (
            f"Reactive block failed, took {total_time:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_set_blocked_zero_immediately_unblocks(self):
        """set_blocked(0) should not actually block."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        limiter.set_blocked(0)

        # Should not be blocked since 0 seconds from now is already past
        await asyncio.sleep(0.01)
        assert limiter.is_blocked() is False
        assert limiter.remaining_wait() == 0

    @pytest.mark.asyncio
    async def test_remaining_wait_when_not_blocked(self):
        """remaining_wait() should return 0 when not blocked."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        assert limiter.remaining_wait() == 0

    @pytest.mark.asyncio
    async def test_remaining_wait_decreases(self):
        """remaining_wait() should decrease over time."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        limiter.set_blocked(2.0)

        wait1 = limiter.remaining_wait()
        assert wait1 > 1.5

        await asyncio.sleep(0.5)
        wait2 = limiter.remaining_wait()
        assert wait2 < wait1

    @pytest.mark.asyncio
    async def test_is_blocked_false_initially(self):
        """is_blocked() should be False for a fresh limiter."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        assert limiter.is_blocked() is False

    @pytest.mark.asyncio
    async def test_high_rate_limit_no_throttling(self):
        """Very high rate limit should not cause throttling."""
        GlobalRateLimiter.reset_instance()
        limiter = GlobalRateLimiter.get_instance(rate_limit=10000, rate_window=60)

        start = time.time()
        for _ in range(20):
            await limiter.wait_if_blocked()
        duration = time.time() - start

        # 20 requests with 10000 limit should be near-instant
        assert duration < 1.0, f"High rate limit caused throttling: {duration:.2f}s"

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """get_instance should return the same object."""
        limiter1 = GlobalRateLimiter.get_instance(rate_limit=10, rate_window=1)
        limiter2 = GlobalRateLimiter.get_instance()
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_reset_instance(self):
        """reset_instance should allow creating a new instance."""
        limiter1 = GlobalRateLimiter.get_instance(rate_limit=10, rate_window=1)
        GlobalRateLimiter.reset_instance()
        limiter2 = GlobalRateLimiter.get_instance(rate_limit=20, rate_window=2)
        assert limiter1 is not limiter2

    @pytest.mark.asyncio
    async def test_wait_if_blocked_returns_false_when_not_blocked(self):
        """wait_if_blocked should return False when not reactively blocked."""
        limiter = GlobalRateLimiter.get_instance(rate_limit=100, rate_window=60)
        result = await limiter.wait_if_blocked()
        assert result is False

    @pytest.mark.asyncio
    async def test_proactive_strict_rolling_window(self):
        """
        Proactive limiter should enforce a strict rolling window:
        for any i, t[i+rate_limit] - t[i] >= rate_window (within tolerance).
        """
        GlobalRateLimiter.reset_instance()
        rate_limit = 2
        rate_window = 0.5
        limiter = GlobalRateLimiter.get_instance(
            rate_limit=rate_limit, rate_window=rate_window
        )

        acquired: list[float] = []

        async def acquire():
            await limiter.wait_if_blocked()
            acquired.append(time.monotonic())

        # Trigger concurrency; without strict rolling windows, this can burst.
        await asyncio.gather(*(acquire() for _ in range(5)))

        acquired.sort()
        assert len(acquired) == 5

        tolerance = 0.05
        for i in range(len(acquired) - rate_limit):
            assert acquired[i + rate_limit] - acquired[i] >= rate_window - tolerance, (
                f"Rolling window violated at i={i}: "
                f"dt={acquired[i + rate_limit] - acquired[i]:.3f}s"
            )
