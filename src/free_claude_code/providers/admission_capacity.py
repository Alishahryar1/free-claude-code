"""Provider capacity shared by clients with independent credential recovery."""

import asyncio
import time
from collections import deque
from collections.abc import Callable

from free_claude_code.providers.admission_policy import ProviderAdmissionLimits


class ProviderCapacityLimiter:
    """Own one provider's rate history, active slots, and capacity waiters."""

    def __init__(self, limits: ProviderAdmissionLimits) -> None:
        self._limits = limits
        self._admission_times: deque[float] = deque()
        self._last_discarded_at: float | None = None
        self._active_attempts = 0
        self._changed = asyncio.Event()

    def reconfigure(self, limits: ProviderAdmissionLimits) -> None:
        """Update limits without resetting occupancy or admission history."""
        if limits != self._limits:
            self._limits = limits
            self.wake_waiters()

    def wake_waiters(self) -> None:
        changed = self._changed
        self._changed = asyncio.Event()
        changed.set()

    async def acquire(self, permit_is_current: Callable[[], bool]) -> bool:
        """Charge capacity only while the caller's recovery permit is valid."""
        while permit_is_current():
            changed = self._changed
            now = time.monotonic()
            limits = self._limits
            cutoff = now - limits.rate_window
            while self._admission_times and self._admission_times[0] <= cutoff:
                self._last_discarded_at = self._admission_times.popleft()

            ready_at = now
            if self._last_discarded_at is not None:
                ready_at = max(ready_at, self._last_discarded_at + limits.rate_window)
            if len(self._admission_times) >= limits.rate_limit:
                ready_at = max(
                    ready_at,
                    self._admission_times[-limits.rate_limit] + limits.rate_window,
                )
            if self._active_attempts < limits.max_concurrency and ready_at <= now:
                self._admission_times.append(now)
                self._active_attempts += 1
                return True

            # Occupancy changes wake waiters. Avoid an expired rate timer
            # spinning while all concurrency slots remain occupied.
            delay = (
                max(0.0, ready_at - now)
                if self._active_attempts < limits.max_concurrency
                else None
            )
            try:
                async with asyncio.timeout(delay):
                    await changed.wait()
            except TimeoutError:
                pass
        return False

    def release(self) -> None:
        self._active_attempts -= 1
        self.wake_waiters()
