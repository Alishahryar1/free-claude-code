"""How long a model waits before its next recovery probe."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


@dataclass(frozen=True, slots=True)
class RecoveryTiming:
    """Exponential backoff bounds, plus the most jitter one delay may add."""

    base_delay: float
    max_delay: float
    jitter: float

    def __post_init__(self) -> None:
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.jitter < 0:
            raise ValueError("jitter must be >= 0")


def retry_delay(
    attempt: int,
    retry_after: float | None,
    timing: RecoveryTiming,
    *,
    noise: float,
) -> float:
    """Back off from attempt ``attempt``; a Retry-After hint can only lengthen it.

    ``noise`` is the caller's jitter draw, between 0 and ``timing.jitter``.
    """
    exponent = max(0, attempt - 1)
    backoff = min(timing.base_delay * (2.0**exponent), timing.max_delay)
    return max(backoff + noise, retry_after or 0.0)


def retry_after_seconds(value: object, *, now: datetime) -> float | None:
    """Read a Retry-After header as seconds from ``now``, or None if unusable.

    Accepts delta-seconds or an HTTP date; a date without a zone is UTC.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
        except TypeError, ValueError, OverflowError:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - now).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)
