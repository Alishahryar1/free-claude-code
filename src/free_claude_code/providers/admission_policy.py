"""Validated capacity and retry policy, independent of clients and lifetime."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from free_claude_code.config.settings import Settings

# Conservative library defaults for a directly constructed controller. The
# application raises these through ``Settings`` so a throttled free tier can
# ride out a per-minute quota window before the failure reaches the client.
# The delay grows exponentially from the base and is capped, so retries are
# always bounded; raise ``PROVIDER_MAX_ATTEMPTS`` / ``PROVIDER_RETRY_MAX_DELAY``
# to wait longer, keeping the total under ``PROVIDER_PROGRESS_TIMEOUT``.
UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS = 5
DEFAULT_UPSTREAM_BASE_DELAY = 2.0
DEFAULT_UPSTREAM_MAX_DELAY = 60.0
DEFAULT_UPSTREAM_JITTER = 1.0


@dataclass(frozen=True, slots=True)
class ProviderAdmissionLimits:
    rate_limit: int
    rate_window: float
    max_concurrency: int
    max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    base_delay: float = DEFAULT_UPSTREAM_BASE_DELAY
    max_delay: float = DEFAULT_UPSTREAM_MAX_DELAY
    jitter: float = DEFAULT_UPSTREAM_JITTER

    def __post_init__(self) -> None:
        for name in ("rate_limit", "max_concurrency", "max_attempts"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.rate_window) or self.rate_window <= 0:
            raise ValueError("rate_window must be finite and > 0")
        if not math.isfinite(self.base_delay) or self.base_delay < 0:
            raise ValueError("base_delay must be finite and >= 0")
        if not math.isfinite(self.max_delay) or self.max_delay < self.base_delay:
            raise ValueError("max_delay must be finite and >= base_delay")
        if not math.isfinite(self.jitter) or self.jitter < 0:
            raise ValueError("jitter must be finite and >= 0")

    @classmethod
    def from_settings(cls, settings: Settings) -> ProviderAdmissionLimits:
        return cls(
            settings.provider_rate_limit,
            settings.provider_rate_window,
            settings.provider_max_concurrency,
            settings.provider_max_attempts,
            settings.provider_retry_base_delay,
            settings.provider_retry_max_delay,
            settings.provider_retry_jitter,
        )
