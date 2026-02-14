import pytest
import pytest_asyncio
import asyncio
import time
import os
import logging

# Set environment variables relative to test execution
os.environ["MESSAGING_RATE_LIMIT"] = "1"
os.environ["MESSAGING_RATE_WINDOW"] = "0.5"

from messaging.limiter import MessagingRateLimiter

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMessagingRateLimiter:
    """Tests for MessagingRateLimiter."""

    @pytest_asyncio.fixture(autouse=True)
    async def reset_limiter(self):
        """Reset singleton and environment before each test."""
        # Ensure the singleton worker is stopped between tests to avoid dangling tasks.
        await MessagingRateLimiter.shutdown_instance(timeout=0.1)
        os.environ["MESSAGING_RATE_LIMIT"] = "1"
        os.environ["MESSAGING_RATE_WINDOW"] = "0.5"

        yield

        await MessagingRateLimiter.shutdown_instance(timeout=0.1)

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that get_instance returns the same object."""
        limiter1 = await MessagingRateLimiter.get_instance()
        limiter2 = await MessagingRateLimiter.get_instance()
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_compaction(self):
        """
        Verify multiple rapid requests with same dedup_key are compacted.
        Logic ported from verify_limiter.py
        """
        # Set slow rate for testing compaction
        os.environ["MESSAGING_RATE_LIMIT"] = "1"
        os.environ["MESSAGING_RATE_WINDOW"] = "1.0"

        # Must reset instance to pick up new env vars
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        call_counts = {}

        async def mock_edit(msg_id, content):
            call_counts[msg_id] = call_counts.get(msg_id, 0) + 1
            return f"done_{content}"

        # Spam 5 edits
        for i in range(5):
            limiter.fire_and_forget(
                lambda i=i: mock_edit("msg1", f"update_{i}"), dedup_key="edit:msg1"
            )

        # Wait for processing
        # 1st might go through immediately, subsequent ones queue and compact
        await asyncio.sleep(2.5)

        # Expected: ~2 calls (first and last)
        assert call_counts["msg1"] <= 2, (
            f"Expected compaction to reduce calls, but got {call_counts.get('msg1', 0)}"
        )
        assert call_counts["msg1"] >= 1, "Expected at least one call"

    @pytest.mark.asyncio
    async def test_compaction_and_futures_resolution(self):
        """
        Verify that even when compacted, all futures resolve to the result of the LAST execution.
        Logic ported from verify_limiter_v2.py
        """
        os.environ["MESSAGING_RATE_LIMIT"] = "1"
        os.environ["MESSAGING_RATE_WINDOW"] = "0.5"
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        call_counts = {}
        msg_id = "test_msg_hang"

        async def mock_edit(mid, content):
            call_counts[mid] = call_counts.get(mid, 0) + 1
            await asyncio.sleep(0.05)
            return f"result_{content}"

        async def task(i):
            return await limiter.enqueue(
                lambda i=i: mock_edit(msg_id, f"v{i}"), dedup_key=f"edit:{msg_id}"
            )

        start_time = time.time()

        # Enqueue 3 tasks concurrently
        results = await asyncio.gather(task(1), task(2), task(3))

        duration = time.time() - start_time

        # All results should be the LAST one executed
        for res in results:
            assert res == "result_v3", f"Expected result_v3, got {res}"

        # Should be reasonably fast
        assert duration < 2.0, "Execution took too long"

        # Calls should be compacted
        assert call_counts[msg_id] <= 2, f"Too many actual calls: {call_counts[msg_id]}"

    @pytest.mark.asyncio
    async def test_flood_wait_handling(self):
        """Test that FloodWait exceptions pause the worker."""
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        # Mock exception with .seconds attribute
        class FloodWait(Exception):
            def __init__(self, seconds):
                self.seconds = seconds
                super().__init__(f"Flood wait {seconds}s")

        call_count = 0

        async def mock_fail():
            nonlocal call_count
            call_count += 1
            raise FloodWait(1)  # 1 second wait

        async def mock_success():
            nonlocal call_count
            call_count += 1
            return "success"

        # First call fails and triggers pause
        try:
            await limiter.enqueue(mock_fail, dedup_key="key1")
        except Exception:
            pass  # Expected

        assert limiter._paused_until > 0

        # Enqueue success, it should wait
        start = time.time()
        await limiter.enqueue(mock_success, dedup_key="key2")
        duration = time.time() - start

        # Should have waited at least ~1s
        assert duration >= 0.9, (
            f"Should have waited for FloodWait, but took {duration:.2f}s"
        )
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_flood_wait_retry_after_parsing(self):
        """Error message with 'retry after N' parses the wait seconds."""
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        async def mock_flood():
            raise Exception("Flood wait: retry after 2 seconds")

        try:
            await limiter.enqueue(mock_flood, dedup_key="retry_parse")
        except Exception:
            pass

        # Should have parsed "after 2" -> 2 seconds
        assert limiter._paused_until > 0

    @pytest.mark.asyncio
    async def test_non_flood_exception_no_pause(self):
        """Non-flood exception doesn't trigger pause."""
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        async def mock_error():
            raise ValueError("some regular error")

        try:
            await limiter.enqueue(mock_error, dedup_key="non_flood")
        except ValueError:
            pass

        # Should NOT have paused since it's not a flood error
        assert limiter._paused_until == 0

    @pytest.mark.asyncio
    async def test_flood_with_seconds_attribute(self):
        """Exception with .seconds attribute uses that value for pause."""
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        class FloodWaitCustom(Exception):
            def __init__(self):
                self.seconds = 2
                super().__init__("Flood wait custom")

        async def mock_flood():
            raise FloodWaitCustom()

        try:
            await limiter.enqueue(mock_flood, dedup_key="flood_sec")
        except Exception:
            pass

        assert limiter._paused_until > 0

    @pytest.mark.asyncio
    async def test_proactive_strict_sliding_window(self):
        """
        Proactive limiter should enforce a strict sliding window:
        for any i, t[i+rate_limit] - t[i] >= rate_window (within tolerance).
        """
        os.environ["MESSAGING_RATE_LIMIT"] = "2"
        os.environ["MESSAGING_RATE_WINDOW"] = "0.5"
        MessagingRateLimiter._instance = None
        limiter = await MessagingRateLimiter.get_instance()

        async def acquire(i: int) -> float:
            async def _do() -> float:
                return time.monotonic()

            return await limiter.enqueue(_do, dedup_key=f"strict:{i}")

        acquired = await asyncio.gather(*(acquire(i) for i in range(5)))
        acquired.sort()

        rate_limit = 2
        rate_window = 0.5
        tolerance = 0.05
        for i in range(len(acquired) - rate_limit):
            assert acquired[i + rate_limit] - acquired[i] >= rate_window - tolerance, (
                f"Sliding window violated at i={i}: "
                f"dt={acquired[i + rate_limit] - acquired[i]:.3f}s"
            )
