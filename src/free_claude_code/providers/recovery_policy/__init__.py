"""Decide what a retryable provider failure does to its model's recovery.

Pure rules: given the failed attempt, the model's recovery episode, the time
and a jitter draw, return one transition. Admission reads the clock and the
episode, applies the transition, and traces it.
"""

from .decision import (
    EpisodeView,
    Exhaust,
    FailedAttempt,
    FailFromEpisode,
    GiveUp,
    OpenEpisode,
    RearmProbe,
    StartCooldown,
    Transition,
    Wait,
    on_failure,
)
from .timing import RecoveryTiming, retry_after_seconds, retry_delay

__all__ = [
    "EpisodeView",
    "Exhaust",
    "FailFromEpisode",
    "FailedAttempt",
    "GiveUp",
    "OpenEpisode",
    "RearmProbe",
    "RecoveryTiming",
    "StartCooldown",
    "Transition",
    "Wait",
    "on_failure",
    "retry_after_seconds",
    "retry_delay",
]
