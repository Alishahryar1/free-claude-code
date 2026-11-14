"""Run one provider request's attempts until one succeeds, hands off, or fails.

The shell that applies ``stream_failure_resolution``'s rules: it owns every
admission, credential and delivery effect of a streaming provider request.
"""

from .attempt import AttemptContext, TransportAttempt
from .loop import AttemptPlan, FailureReporting, run_attempts

__all__ = [
    "AttemptContext",
    "AttemptPlan",
    "FailureReporting",
    "TransportAttempt",
    "run_attempts",
]
