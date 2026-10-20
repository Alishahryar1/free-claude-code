"""Exclusive reasoning encoders for Google OpenAI-compatible endpoints."""

from dataclasses import dataclass
from typing import Any, cast

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    validate_extra_body_does_not_override_reasoning_fields,
)

_GEMINI_EFFORTS = {
    ReasoningEffort.MINIMAL: "minimal",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
    ReasoningEffort.XHIGH: "high",
    ReasoningEffort.MAX: "high",
}
_THINKING_CONFIG_CONFLICT = (
    "Google extra_body.google.thinking_config cannot be combined with FCC "
    "reasoning controls. Use either thinking/output_config or the provider-native "
    "thinking_config."
)


@dataclass(frozen=True, slots=True)
class GeminiReasoningEncoder:
    """Choose one Gemini API reasoning channel for each resolved policy."""

    off_field = ("reasoning_effort",)

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if _preserve_caller_thinking_config(body, policy):
            return

        if policy.control is ReasoningControl.OFF:
            body["reasoning_effort"] = "none"
            return

        if policy.budget_tokens is not None:
            thinking = _thinking_config(body)
            thinking["thinking_budget"] = policy.budget_tokens
            thinking["include_thoughts"] = True
            return

        if effort := _GEMINI_EFFORTS.get(policy.effort):
            body["reasoning_effort"] = effort
            return

        if policy.control is ReasoningControl.ON:
            _thinking_config(body)["include_thoughts"] = True


@dataclass(frozen=True, slots=True)
class VertexReasoningEncoder:
    """Encode Vertex reasoning through one Google thinking-config object."""

    off_field = ("extra_body", "extra_body", "google", "thinking_config")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if _preserve_caller_thinking_config(body, policy):
            return

        if policy.control is ReasoningControl.OFF:
            thinking = _thinking_config(body)
            thinking["thinking_budget"] = 0
            thinking["include_thoughts"] = False
            return

        budget = policy.numeric_budget_tokens
        if budget is not None:
            thinking = _thinking_config(body)
            thinking["thinking_budget"] = budget
            thinking["include_thoughts"] = True
            return

        if policy.control is ReasoningControl.ON:
            _thinking_config(body)["include_thoughts"] = True


def validate_google_extra_body(extra: dict[str, Any]) -> None:
    """Validate caller extensions before a Google encoder takes ownership."""

    validate_extra_body_does_not_override_reasoning_fields(extra)
    literal_extra = _optional_object(extra, "extra_body", "extra_body.extra_body")
    if literal_extra is None:
        return
    google = _optional_object(literal_extra, "google", "extra_body.google")
    if google is None:
        return
    _optional_object(
        google,
        "thinking_config",
        "extra_body.google.thinking_config",
    )


def _preserve_caller_thinking_config(
    body: dict[str, Any], policy: ReasoningPolicy
) -> bool:
    if _existing_thinking_config(body) is None:
        return False
    if policy != DEFAULT_REASONING_POLICY:
        raise InvalidRequestError(_THINKING_CONFIG_CONFLICT)
    return True


def _existing_thinking_config(body: dict[str, Any]) -> dict[str, Any] | None:
    sdk_extra = body.get("extra_body")
    if not isinstance(sdk_extra, dict):
        return None
    literal_extra = sdk_extra.get("extra_body")
    if not isinstance(literal_extra, dict):
        return None
    google = literal_extra.get("google")
    if not isinstance(google, dict):
        return None
    thinking = google.get("thinking_config")
    return cast(dict[str, Any], thinking) if isinstance(thinking, dict) else None


def _thinking_config(body: dict[str, Any]) -> dict[str, Any]:
    sdk_extra = _object(body, "extra_body")
    literal_extra = _object(sdk_extra, "extra_body")
    google = _object(literal_extra, "google")
    return _object(google, "thinking_config")


def _object(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object.")
    return cast(dict[str, Any], value)


def _optional_object(
    container: dict[str, Any], key: str, path: str
) -> dict[str, Any] | None:
    if key not in container:
        return None
    value = container[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    return cast(dict[str, Any], value)
