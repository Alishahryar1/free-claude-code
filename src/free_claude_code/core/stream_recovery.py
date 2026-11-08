"""Portable recovery evidence shared by provider, routing, and public delivery."""

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .history_replay import readable_reasoning
from .json_types import JsonObject


@dataclass(frozen=True, slots=True)
class DeliveredPrefix:
    text: str
    thinking: str
    has_calls: bool
    eligible: bool
    can_handoff: bool
    unsafe_reason: str | None
    retained_bytes: int = 0
    continuation: ContinuationSeed | None = None


@dataclass(frozen=True, slots=True)
class ContinuationSeed:
    """A snapshot of delivered output in the client's own protocol."""

    wire_api: Literal["messages", "responses", "gemini", "chat"]
    items: tuple[JsonObject, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", deepcopy(self.items))

    def readable_text(self, channel: Literal["text", "thinking"]) -> str:
        """Read the ordinary text already delivered in the client protocol."""
        values: list[str] = []
        for item in self.items:
            if self.wire_api == "chat":
                if channel == "text":
                    values.append(str(item.get("content") or ""))
                else:
                    values.extend(text for text, _ in readable_reasoning(item))
            elif self.wire_api == "messages":
                if item.get("type") == channel:
                    values.append(str(item.get(channel) or ""))
            elif self.wire_api == "gemini":
                if bool(item.get("thought")) == (channel == "thinking"):
                    values.append(str(item.get("text") or ""))
            elif item.get("type") == ("message" if channel == "text" else "reasoning"):
                for field in (
                    ("content",) if channel == "text" else ("content", "summary")
                ):
                    parts = item.get(field)
                    if isinstance(parts, list):
                        values.extend(
                            str(part.get("text") or "")
                            for part in parts
                            if isinstance(part, dict)
                        )
        return "".join(values)


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
    if prefix.continuation is None:
        return RecoveryAssessment(RecoveryAction.STOP, reason="missing_native_prefix")
    return RecoveryAssessment(
        RecoveryAction.CONTINUE,
        prefix.continuation,
        reason="portable_prefix",
    )
