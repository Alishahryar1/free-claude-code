"""Global rate limiter for API requests."""

import asyncio
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    Global singleton rate limiter that blocks all requests
    when a rate limit error is encountered.

    No proactive limits - only reactive when 429 is hit.
    No retry logic - just pauses all requests until cooldown expires.
    """

    _instance: Optional["GlobalRateLimiter"] = None

    def __init__(self):
        self._blocked_until: float = 0
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "GlobalRateLimiter":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def wait_if_blocked(self) -> bool:
        """
        Wait if currently rate limited.

        Returns:
            True if was blocked and waited, False if not blocked
        """
        now = time.time()
        if now < self._blocked_until:
            wait_time = self._blocked_until - now
            logger.warning(f"Global rate limit active, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            return True
        return False

    def set_blocked(self, seconds: float = 60) -> None:
        """
        Set global block for specified seconds.

        Args:
            seconds: How long to block (default 60s)
        """
        self._blocked_until = time.time() + seconds
        logger.warning(f"Global rate limit set for {seconds:.1f}s")

    def is_blocked(self) -> bool:
        """Check if currently blocked."""
        return time.time() < self._blocked_until

    def remaining_wait(self) -> float:
        """Get remaining wait time in seconds."""
        return max(0, self._blocked_until - time.time())
