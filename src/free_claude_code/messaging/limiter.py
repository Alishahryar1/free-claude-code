"""Runtime-owned queued delivery for one messaging platform."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from free_claude_code.core.rate_limit import (
    StrictSlidingWindowLimiter as SlidingWindowLimiter,
)

from .safe_diagnostics import format_exception_for_log


class MessagingRateLimiter:
    """
    Rate limiter and compacting work queue for one messaging runtime.

    Uses a custom queue with task compaction (deduplication) to ensure
    only the latest version of a message update is processed.
    """

    def __init__(
        self,
        *,
        rate_limit: int,
        rate_window: float,
        log_error_details: bool = False,
    ) -> None:
        self.limiter = SlidingWindowLimiter(rate_limit, rate_window)
        self._log_error_details = log_error_details
        # Custom queue state - using deque for O(1) popleft
        self._queue_list: deque[str] = deque()  # Deque of dedup_keys in order
        self._queue_map: dict[
            str, tuple[Callable[[], Awaitable[Any]], list[asyncio.Future]]
        ] = {}
        self._condition = asyncio.Condition()
        self._shutdown = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._active_futures: list[asyncio.Future[Any]] = []
        self._closed = False
        self._paused_until = 0

        logger.info(
            f"MessagingRateLimiter initialized ({rate_limit} req / {rate_window}s with Task Compaction)"
        )

    def start(self) -> None:
        """Start the owned worker on the current event loop."""
        if self._closed:
            raise RuntimeError("Messaging rate limiter is closed.")
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(
            self._worker(), name="msg-limiter-worker"
        )

    async def _worker(self) -> None:
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
                    self._active_futures = futures

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
                    except asyncio.CancelledError:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        worker = asyncio.current_task()
                        if self._shutdown.is_set() or (
                            worker is not None and worker.cancelling()
                        ):
                            raise
                        logger.debug(
                            "Messaging operation cancelled for key {}; worker remains active",
                            dedup_key,
                        )
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
                                "Error in limiter worker for key {}: {}",
                                dedup_key,
                                format_exception_for_log(
                                    e,
                                    log_full_message=self._log_error_details,
                                ),
                            )
                    finally:
                        self._active_futures = []
            except asyncio.CancelledError:
                for future in self._active_futures:
                    if not future.done():
                        future.cancel()
                self._active_futures = []
                if self._shutdown.is_set():
                    break
                raise
            except Exception as e:
                if self._log_error_details:
                    logger.error(
                        "MessagingRateLimiter worker critical error: {}",
                        e,
                        exc_info=True,
                    )
                else:
                    logger.error(
                        "MessagingRateLimiter worker critical error: exc_type={}",
                        type(e).__name__,
                    )
                await asyncio.sleep(1)

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cancel queued work and stop every task owned by this limiter."""
        self._closed = True
        self._shutdown.set()
        async with self._condition:
            queued_futures = [
                future
                for _func, futures in self._queue_map.values()
                for future in futures
            ]
            self._queue_list.clear()
            self._queue_map.clear()
            for future in queued_futures:
                if not future.done():
                    future.cancel()
            for future in self._active_futures:
                if not future.done():
                    future.cancel()
            self._condition.notify_all()

        cancellation: asyncio.CancelledError | None = None
        timeout_error: TimeoutError | None = None
        task = self._worker_task
        if task and not task.done():
            task.cancel()
            try:
                drain = asyncio.gather(task, return_exceptions=True)
                if timeout is None:
                    await drain
                else:
                    await asyncio.wait_for(drain, timeout=timeout)
            except TimeoutError as exc:
                timeout_error = exc
            except asyncio.CancelledError as exc:
                cancellation = exc
        if task is None or task.done():
            self._worker_task = None

        background_tasks = tuple(self._background_tasks)
        for background_task in background_tasks:
            background_task.cancel()
        if background_tasks:
            try:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            except asyncio.CancelledError as exc:
                cancellation = exc
        self._background_tasks.difference_update(
            task for task in background_tasks if task.done()
        )

        if cancellation is not None:
            raise cancellation
        if timeout_error is not None:
            raise TimeoutError(
                "MessagingRateLimiter worker did not stop before timeout"
            ) from timeout_error

    async def _enqueue_internal(
        self,
        func: Callable[[], Awaitable[Any]],
        future: asyncio.Future[Any],
        dedup_key: str,
        front: bool = False,
    ) -> None:
        await self._enqueue_internal_multi(func, [future], dedup_key, front)

    async def _enqueue_internal_multi(
        self,
        func: Callable[[], Awaitable[Any]],
        futures: list[asyncio.Future[Any]],
        dedup_key: str,
        front: bool = False,
    ) -> None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("Messaging rate limiter is closed.")
            if self._worker_task is None or self._worker_task.done():
                raise RuntimeError("Messaging rate limiter has not been started.")
            if dedup_key in self._queue_map:
                # Compaction: Update existing task with new func, append new futures
                _old_func, old_futures = self._queue_map[dedup_key]
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
        self, func: Callable[[], Awaitable[Any]], dedup_key: str | None = None
    ) -> Any:
        """
        Enqueue a messaging task and return its future result.
        If dedup_key is provided, subsequent tasks with the same key will replace this one.
        """
        self._require_running()
        if dedup_key is None:
            # Unique key to avoid deduplication
            dedup_key = f"task_{id(func)}_{asyncio.get_running_loop().time()}"

        future = asyncio.get_running_loop().create_future()
        try:
            await self._enqueue_internal(func, future, dedup_key)
        except BaseException:
            future.cancel()
            raise
        return await future

    def fire_and_forget(
        self, func: Callable[[], Awaitable[Any]], dedup_key: str | None = None
    ) -> None:
        """Enqueue a task without waiting for the result."""
        self._require_running()
        if dedup_key is None:
            dedup_key = f"task_{id(func)}_{asyncio.get_running_loop().time()}"

        async def _wrapped() -> None:
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    await self.enqueue(func, dedup_key)
                    return
                except Exception as e:
                    error_msg = str(e).lower()
                    # Only retry transient connectivity issues that might have slipped through
                    # or occurred between platform checks.
                    if attempt < max_retries and any(
                        x in error_msg for x in ["connect", "timeout", "broken"]
                    ):
                        wait = 2**attempt
                        if self._log_error_details:
                            logger.warning(
                                "Limiter fire_and_forget transient error (attempt {}): {}. Retrying in {}s...",
                                attempt + 1,
                                e,
                                wait,
                            )
                        else:
                            logger.warning(
                                "Limiter fire_and_forget transient error (attempt {}): exc_type={}. Retrying in {}s...",
                                attempt + 1,
                                type(e).__name__,
                                wait,
                            )
                        await asyncio.sleep(wait)
                        continue

                    logger.error(
                        "Final error in fire_and_forget for key {}: {}",
                        dedup_key,
                        format_exception_for_log(
                            e,
                            log_full_message=self._log_error_details,
                        ),
                    )
                    break

        task = asyncio.create_task(_wrapped(), name=f"msg-limiter:{dedup_key}")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _require_running(self) -> None:
        if self._closed:
            raise RuntimeError("Messaging rate limiter is closed.")
        if self._worker_task is None or self._worker_task.done():
            raise RuntimeError("Messaging rate limiter has not been started.")
