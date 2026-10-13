"""Reasoning and thinking conversion helpers for OpenAI Responses."""

from collections.abc import Mapping
from typing import Any

from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)

from .tools import optional_str


def encrypted_reasoning_from_item(item: Mapping[str, Any]) -> str | None:
    """Return opaque reasoning content without interpreting it."""

    return optional_str(item.get("encrypted_content"))


def combine_reasoning(existing: str | None, addition: str | None) -> str | None:
    if addition is None:
        return existing
    if existing is None:
        return addition
    if existing == "":
        return addition
    if addition == "":
        return existing
    return f"{existing}\n{addition}"


def responses_reasoning_to_output_config(value: Any) -> dict[str, Any] | None:
    """Preserve the client's named effort for application-level resolution."""
    if not isinstance(value, Mapping):
        return None
    effort = value.get("effort")
    if isinstance(effort, str) and effort.strip():
        return {"effort": effort.strip().lower()}
    return None


def responses_reasoning_policy(value: object) -> ReasoningPolicy:
    """Parse the supported reasoning intent from a Responses request field."""

    if not isinstance(value, Mapping):
        return ReasoningPolicy.provider_default()
    raw_effort = value.get("effort")
    if not isinstance(raw_effort, str):
        return ReasoningPolicy.provider_default()
    normalized = raw_effort.strip().lower()
    if normalized == "none":
        return ReasoningPolicy.off()
    try:
        effort = ReasoningEffort(normalized)
    except ValueError:
        return ReasoningPolicy.provider_default()
    return ReasoningPolicy(
        control=ReasoningControl.DEFAULT,
        effort=effort,
    )


def responses_reasoning_config(reasoning: ReasoningPolicy) -> dict[str, str]:
    """Encode resolved reasoning intent for an upstream Responses request."""

    if reasoning.control is ReasoningControl.OFF:
        return {"effort": "none"}
    if reasoning.effort is not None:
        return {"effort": reasoning.effort.value, "summary": "auto"}
    if reasoning.requests_reasoning:
        return {"summary": "auto"}
    return {}
