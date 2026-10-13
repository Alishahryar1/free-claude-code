"""Validated capacity policy, independent of clients and controller lifetime."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from free_claude_code.config.settings import Settings


@dataclass(frozen=True, slots=True)
class ProviderAdmissionLimits:
    rate_limit: int
    rate_window: float
    max_concurrency: int

    def __post_init__(self) -> None:
        for name in ("rate_limit", "max_concurrency"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.rate_window) or self.rate_window <= 0:
            raise ValueError("rate_window must be finite and > 0")

    @classmethod
    def from_settings(cls, settings: Settings) -> ProviderAdmissionLimits:
        return cls(
            settings.provider_rate_limit,
            settings.provider_rate_window,
            settings.provider_max_concurrency,
        )
