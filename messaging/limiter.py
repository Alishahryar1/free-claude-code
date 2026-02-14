"""
Global Rate Limiter for Messaging Platforms.

Centralizes outgoing message requests and ensures compliance with rate limits
using a strict sliding window algorithm and a task queue.
"""

import asyncio
import logging
import os
from collections import deque
import time
from typing import Awaitable, Callable, Any, Optional, Dict

logger = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """Strict sliding window limiter.

    Guarantees: at most `rate_limit` acquisitions in any interval of length
    `rate_window` (seconds).

    Implemented as an async context manager so call sites can do:
        async with limiter:
            ...
    """

    def __init__(self, rate_limit: int, rate_window: float) -> None:
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")

        self._rate_limit = int(rate_limit)
        self._rate_window = float(rate_window)
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._rate_window

                while self._times and self._times[0] <= cutoff:
                    self._times.popleft()

                if len(self._times) < self._rate_limit:
                    self._times.append(now)
                    return

                oldest = self._times[0]
                wait_time = max(0.0, (oldest + self._rate_window) - now)

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    async def __aenter__(self) -> "SlidingWindowLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class MessagingRateLimiter:
    """
    A thread-safe global rate limiter for messaging.

    Uses a custom queue with task compaction (deduplication) to ensure
    only the latest version of a message update is processed.
    """

    _instance: Optional["MessagingRateLimiter"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            pass
        return super(MessagingRateLimiter, cls).__new__(cls)

    @classmethod
    async def get_instance(cls) -> "MessagingRateLimiter":
        """Get the singleton instance of the limiter."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                # Start the background worker (tracked for graceful shutdown).
                cls._instance._start_worker()
        return cls._instance

    def __init__(self):
        # Prevent double initialization in singleton
        if hasattr(self, "_initialized"):
            return

        rate_limit = int(os.getenv("MESSAGING_RATE_LIMIT", "1"))
        rate_window = float(os.getenv("MESSAGING_RATE_WINDOW", "2.0"))

        self.limiter = SlidingWindowLimiter(rate_limit, rate_window)
        # Custom queue state - using deque for O(1) popleft
        self._queue_list: deque[str] = deque()  # Deque of dedup_keys in order
        self._queue_map: Dict[
            str, tuple[Callable[[], Awaitable[Any]], list[asyncio.Future]]
        ] = {}
        self._condition = asyncio.Condition()
        self._shutdown = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

        self._initialized = True
        self._paused_until = 0

        logger.info(
            f"MessagingRateLimiter initialized ({rate_limit} req / {rate_window}s with Task Compaction)"
        )

    def _start_worker(self) -> None:
        """Ensure the worker task exists."""
        if self._worker_task and not self._worker_task.done():
            return
        # Named task helps debugging shutdown hangs.
        self._worker_task = asyncio.create_task(
            self._worker(), name="msg-limiter-worker"
        )

    async def _worker(self):
        """Background worker that processes queued messaging tasks."""
        logger.info("MessagingRateLimiter worker started")
        while not self._shutdown.is_set():
            try:
                # Get a task from the queue
                async with self._condition:
                    while not self._queue_list and not self._shutdown.is_set():
                        await self._condition.wait()

                    if self._shutdown.is_set():
                        break

                    dedup_key = self._queue_list.popleft()
                    func, futures = self._queue_map.pop(dedup_key)

                # Check for manual pause (FloodWait)
                now = asyncio.get_event_loop().time()
                if self._paused_until > now:
                    wait_time = self._paused_until - now
                    logger.warning(
                        f"Limiter worker paused, waiting {wait_time:.1f}s more..."
                    )
                    await asyncio.sleep(wait_time)

                # Wait for rate limit capacity
                async with self.limiter:
                    try:
                        result = await func()
                        for f in futures:
                            if not f.done():
                                f.set_result(result)
                    except Exception as e:
                        # Report error to all futures and log it
                        for f in futures:
                            if not f.done():
                                f.set_exception(e)

                        error_msg = str(e).lower()
                        if "flood" in error_msg or "wait" in error_msg:
                            seconds = 30
                            try:
                                if hasattr(e, "seconds"):
                                    seconds = e.seconds
                                elif "after " in error_msg:
                                    # Try to parse "retry after X"
                                    parts = error_msg.split("after ")
                                    if len(parts) > 1:
                                        seconds = int(parts[1].split()[0])
                            except Exception:
                                pass

                            logger.error(
                                f"FloodWait detected! Pausing worker for {seconds}s"
                            )
                            wait_secs = (
                                float(seconds)
                                if isinstance(seconds, (int, float, str))
                                else 30.0
                            )
                            self._paused_until = (
                                asyncio.get_event_loop().time() + wait_secs
                            )
                        else:
                            logger.error(
                                f"Error in limiter worker for key {dedup_key}: {type(e).__name__}: {e}"
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"MessagingRateLimiter worker critical error: {e}", exc_info=True
                )
                await asyncio.sleep(1)

    async def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the background worker so process shutdown doesn't hang."""
        self._shutdown.set()
        try:
            async with self._condition:
                self._condition.notify_all()
        except Exception:
            # Best-effort: condition may be bound to a closing loop.
            pass

        task = self._worker_task
        if not task or task.done():
            self._worker_task = None
            return

        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("MessagingRateLimiter worker did not stop before timeout")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"MessagingRateLimiter worker shutdown error: {e}")
        finally:
            self._worker_task = None

    @classmethod
    async def shutdown_instance(cls, timeout: float = 2.0) -> None:
        """Shutdown and clear the singleton instance (safe to call multiple times)."""
        inst = cls._instance
        if not inst:
            return
        try:
            await inst.shutdown(timeout=timeout)
        finally:
            cls._instance = None

    async def _enqueue_internal(self, func, future, dedup_key, front=False):
        await self._enqueue_internal_multi(func, [future], dedup_key, front)

    async def _enqueue_internal_multi(self, func, futures, dedup_key, front=False):
        async with self._condition:
            if dedup_key in self._queue_map:
                # Compaction: Update existing task with new func, append new futures
                old_func, old_futures = self._queue_map[dedup_key]
                old_futures.extend(futures)
                self._queue_map[dedup_key] = (func, old_futures)
                logger.debug(
                    f"Compacted task for key: {dedup_key} (now {len(old_futures)} futures)"
                )
            else:
                self._queue_map[dedup_key] = (func, futures)
                if front:
                    self._queue_list.appendleft(dedup_key)
                else:
                    self._queue_list.append(dedup_key)
                self._condition.notify_all()

    async def enqueue(
        self, func: Callable[[], Awaitable[Any]], dedup_key: Optional[str] = None
    ) -> Any:
        """
        Enqueue a messaging task and return its future result.
        If dedup_key is provided, subsequent tasks with the same key will replace this one.
        """
        if dedup_key is None:
            # Unique key to avoid deduplication
            dedup_key = f"task_{id(func)}_{asyncio.get_event_loop().time()}"

        future = asyncio.get_event_loop().create_future()
        await self._enqueue_internal(func, future, dedup_key)
        return await future

    def fire_and_forget(
        self, func: Callable[[], Awaitable[Any]], dedup_key: Optional[str] = None
    ):
        """Enqueue a task without waiting for the result."""
        if dedup_key is None:
            dedup_key = f"task_{id(func)}_{asyncio.get_event_loop().time()}"

        future = asyncio.get_event_loop().create_future()

        async def _wrapped():
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    return await self.enqueue(func, dedup_key)
                except Exception as e:
                    error_msg = str(e).lower()
                    # Only retry transient connectivity issues that might have slipped through
                    # or occurred between platform checks.
                    if attempt < max_retries and any(
                        x in error_msg for x in ["connect", "timeout", "broken"]
                    ):
                        wait = 2**attempt
                        logger.warning(
                            f"Limiter fire_and_forget transient error (attempt {attempt + 1}): {e}. Retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
                        continue

                    logger.error(
                        f"Final error in fire_and_forget for key {dedup_key}: {type(e).__name__}: {e}"
                    )
                    if not future.done():
                        future.set_exception(e)
                    break

        asyncio.create_task(_wrapped())
