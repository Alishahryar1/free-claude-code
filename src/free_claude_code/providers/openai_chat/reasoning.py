"""Provider-owned reasoning translations for OpenAI-compatible APIs."""

from dataclasses import dataclass
from typing import Any, Protocol

from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)

EffortValues = tuple[tuple[ReasoningEffort, str], ...]


class ReasoningEncoder(Protocol):
    """Translate provider-neutral reasoning intent into one wire shape."""

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None: ...

    @property
    def off_field(self) -> tuple[str, ...] | None: ...


@dataclass(frozen=True, slots=True)
class NoReasoning:
    """Leave reasoning computation entirely to the upstream provider."""

    off_field = None

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        return


@dataclass(frozen=True, slots=True)
class NamedEffortReasoning:
    """Encode a provider's documented named-effort vocabulary."""

    efforts: EffortValues
    disabled_value: str | bool | None = None
    enabled_value: str | bool | None = None
    field: str = "reasoning_effort"
    budget_field: str | None = None
    use_extra_body: bool = False

    @property
    def off_field(self) -> tuple[str, ...] | None:
        if self.disabled_value is None:
            return None
        return ("extra_body", self.field) if self.use_extra_body else (self.field,)

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        target = _extra_body(body) if self.use_extra_body else body
        if policy.control is ReasoningControl.OFF:
            if self.disabled_value is not None:
                target[self.field] = self.disabled_value
            return

        if policy.budget_tokens is not None and self.budget_field is not None:
            target[self.budget_field] = policy.budget_tokens
            return

        effort = dict(self.efforts).get(policy.effort)
        if effort is not None:
            target[self.field] = effort
            return

        if policy.control is ReasoningControl.ON and self.enabled_value is not None:
            target[self.field] = self.enabled_value


@dataclass(frozen=True, slots=True)
class ReasoningObject:
    """Encode gateways that accept a top-level ``reasoning`` object."""

    efforts: EffortValues
    supports_budget: bool = True

    off_field = ("extra_body", "reasoning")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            _extra_body(body)["reasoning"] = {"enabled": False}
            return

        reasoning: dict[str, Any] = {}
        if policy.budget_tokens is not None and self.supports_budget:
            reasoning["max_tokens"] = policy.budget_tokens
        elif effort := dict(self.efforts).get(policy.effort):
            reasoning["effort"] = effort
        elif policy.control is ReasoningControl.ON:
            reasoning["enabled"] = True

        if reasoning:
            _extra_body(body)["reasoning"] = reasoning


@dataclass(frozen=True, slots=True)
class ThinkingObjectReasoning:
    """Encode providers with an enabled/disabled ``thinking`` object."""

    enabled: dict[str, Any]
    disabled: dict[str, Any]

    off_field = ("extra_body", "thinking")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            _extra_body(body)["thinking"] = dict(self.disabled)
        elif policy.requests_reasoning:
            _extra_body(body)["thinking"] = dict(self.enabled)


@dataclass(frozen=True, slots=True)
class ChatTemplateReasoning:
    """Encode a provider-wide chat-template boolean without model guessing."""

    field: str = "thinking"

    @property
    def off_field(self) -> tuple[str, ...]:
        return ("extra_body", "chat_template_kwargs", self.field)

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if not policy.requests_reasoning and policy.control is not ReasoningControl.OFF:
            return
        kwargs = _nested_dict(_extra_body(body), "chat_template_kwargs")
        kwargs[self.field] = policy.control is not ReasoningControl.OFF


@dataclass(frozen=True, slots=True)
class LlamaCppReasoning:
    """Encode llama.cpp's per-request numeric thinking budget."""

    off_field = ("extra_body", "thinking_budget_tokens")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            _extra_body(body)["thinking_budget_tokens"] = 0
        elif (budget := policy.numeric_budget_tokens) is not None:
            _extra_body(body)["thinking_budget_tokens"] = budget


@dataclass(frozen=True, slots=True)
class SplitReasoningOutput:
    """Request separate reasoning output where compute is not controllable."""

    off_field = None

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        _extra_body(body)["reasoning_split"] = True


def _extra_body(body: dict[str, Any]) -> dict[str, Any]:
    value = body.setdefault("extra_body", {})
    if not isinstance(value, dict):
        raise TypeError("OpenAI extra_body must be an object.")
    return value


def _nested_dict(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object.")
    return value


NO_REASONING = NoReasoning()
LLAMACPP_REASONING = LlamaCppReasoning()
SPLIT_REASONING_OUTPUT = SplitReasoningOutput()
