"""Portable recovery evidence shared by provider, routing, and public delivery."""

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from .json_types import JsonObject


@dataclass(frozen=True, slots=True)
class DeliveredPrefix:
    text: str
    thinking: str
    has_calls: bool
    eligible: bool
    can_handoff: bool
    unsafe_reason: str | None


@dataclass(frozen=True, slots=True)
class ContinuationSeed:
    text: str
    thinking: str


@dataclass(frozen=True, slots=True)
class SourceRecoverySnapshot:
    normal_stop_seen: bool = False
    structured_output: bool = False
    unsafe_reason: str | None = None
    can_handoff: bool = True


@dataclass(frozen=True, slots=True)
class StreamFailureContext:
    source: SourceRecoverySnapshot
    responses_failure_payload: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.responses_failure_payload is not None:
            object.__setattr__(
                self,
                "responses_failure_payload",
                deepcopy(self.responses_failure_payload),
            )


class RecoveryAction(StrEnum):
    RESTART = "restart"
    CONTINUE = "continue"
    HANDOFF = "handoff"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RecoveryAssessment:
    action: RecoveryAction
    seed: ContinuationSeed | None = None
    reason: str | None = None


def assess_recovery(
    *,
    content_released: bool,
    prefix: DeliveredPrefix | None,
    source: SourceRecoverySnapshot,
    retryable: bool,
) -> RecoveryAssessment:
    """Assess safe response transitions independently of any attempt budget."""
    if source.normal_stop_seen:
        return RecoveryAssessment(RecoveryAction.STOP, reason="normal_stop")
    if not content_released:
        return (
            RecoveryAssessment(RecoveryAction.STOP, reason=source.unsafe_reason)
            if source.unsafe_reason
            else RecoveryAssessment(
                RecoveryAction.RESTART, reason="no_delivered_content"
            )
        )
    if prefix is None:
        return RecoveryAssessment(RecoveryAction.STOP, reason="missing_prefix")
    if prefix.has_calls:
        if (
            retryable
            and not source.structured_output
            and source.can_handoff
            and prefix.can_handoff
        ):
            return RecoveryAssessment(
                RecoveryAction.HANDOFF, reason="completed_tool_calls"
            )
        return RecoveryAssessment(RecoveryAction.STOP, reason="released_tool_calls")
    reason = (
        "structured_output"
        if source.structured_output
        else source.unsafe_reason or prefix.unsafe_reason
    )
    if reason or not prefix.eligible:
        return RecoveryAssessment(RecoveryAction.STOP, reason=reason or "unsafe_prefix")
    if not (prefix.text or prefix.thinking):
        return RecoveryAssessment(RecoveryAction.STOP, reason="empty_prefix")
    return RecoveryAssessment(
        RecoveryAction.CONTINUE,
        ContinuationSeed(prefix.text, prefix.thinking),
        reason="portable_prefix",
    )
