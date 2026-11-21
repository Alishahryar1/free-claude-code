"""Decide what a failed messaging platform call does next.

Pure rules: given what went wrong and how far the call has got, retry, resend
without formatting, treat it as delivered, hand a rate limit to the queue, or
fail. Platform adapters read their library's errors into a ``DeliveryFailure``
and carry out the decision; the limiter asks ``queue_pause`` how long queued
work waits.
"""

from .decision import (
    MAX_ATTEMPTS,
    CallState,
    Decision,
    DeliveryFailure,
    FailureKind,
    Ignore,
    Raise,
    RateLimited,
    Retry,
    RetryPlain,
    decide,
)
from .queue import PlatformRateLimited, queue_pause

__all__ = [
    "MAX_ATTEMPTS",
    "CallState",
    "Decision",
    "DeliveryFailure",
    "FailureKind",
    "Ignore",
    "PlatformRateLimited",
    "Raise",
    "RateLimited",
    "Retry",
    "RetryPlain",
    "decide",
    "queue_pause",
]
