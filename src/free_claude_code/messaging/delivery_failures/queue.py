"""How long queued deliveries wait after a failure."""


class PlatformRateLimited(RuntimeError):
    """A platform kept rate-limiting one call; queued work waits ``retry_after``."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"Platform rate limit: retry after {retry_after:g}s")
        self.retry_after = retry_after


def queue_pause(error: BaseException) -> float | None:
    """Seconds every queued delivery waits after ``error``, or None to go on."""
    if isinstance(error, PlatformRateLimited):
        return error.retry_after
    return None
