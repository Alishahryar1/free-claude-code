"""Decide what follows one failed model candidate.

Pure: the caller supplies what the client has received and what the failed
candidate left behind, and acts on the step returned.
"""

from collections.abc import Callable
from dataclasses import dataclass

from free_claude_code.core.stream_recovery import (
    ContinuationSeed,
    DeliveredPrefix,
    RecoveryAction,
    StreamFailureContext,
    assess_recovery,
)


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    """What one failed candidate left behind."""

    context: StreamFailureContext | None
    retryable: bool
    produced_output: bool


@dataclass(frozen=True, slots=True)
class Fail:
    """Raise the candidate's failure to the client."""


@dataclass(frozen=True, slots=True)
class Handoff:
    """End the response so the client can act on its released tool calls."""


@dataclass(frozen=True, slots=True)
class Fallback:
    """Open the next candidate, continuing from ``seed`` when there is one."""

    seed: ContinuationSeed | None
    action: str
    reason: str | None


type FallbackStep = Fail | Handoff | Fallback


def next_step(
    failure: CandidateFailure,
    *,
    content_released: bool,
    read_prefix: Callable[[], DeliveredPrefix | None],
    seed: ContinuationSeed | None,
    has_next: bool,
) -> FallbackStep:
    """Return the step after ``failure``, given the delivery so far."""
    action = "continue" if seed is not None else "restart"
    reason: str | None = "no_candidate_output"
    if failure.context is None:
        # Without stream evidence, only a candidate that produced nothing
        # leaves the delivered prefix, and so the current seed, intact.
        if content_released and (failure.produced_output or seed is None):
            return Fail()
    else:
        assessment = assess_recovery(
            content_released=content_released,
            prefix=read_prefix(),
            source=failure.context.source,
            retryable=failure.retryable,
        )
        if assessment.action is RecoveryAction.STOP:
            return Fail()
        if assessment.action is RecoveryAction.HANDOFF:
            return Handoff()
        seed = assessment.seed
        action = assessment.action.value
        reason = assessment.reason
    if not has_next:
        return Fail()
    return Fallback(seed, action, reason)
