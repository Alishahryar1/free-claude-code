"""Resolve native Messages controls from explicit intent and model metadata."""

from dataclasses import dataclass
from typing import Literal

from free_claude_code.core.anthropic.models import ThinkingConfig
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    NativeMessagesOptions,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy

DEFAULT_MESSAGES_OUTPUT_TOKENS = 8192
_MIN_THINKING_BUDGET = 1024
_NATIVE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True, slots=True)
class MessagesModelCapabilities:
    """Unknown metadata stays unknown rather than being inferred from names."""

    max_output_tokens: int | None = None
    adaptive_thinking: Literal["unsupported", "optional", "required"] | None = None
    supports_output_effort: bool | None = None
    supported_efforts: tuple[str, ...] | None = None
    supports_vision: bool | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("Advertised output cap must be a positive integer.")


def resolve_messages_options(
    *,
    model: str,
    max_tokens: int | None,
    reasoning: ReasoningPolicy,
    capabilities: MessagesModelCapabilities = MessagesModelCapabilities(),
    thinking: ThinkingConfig | None = None,
    output_effort: object = None,
) -> NativeMessagesOptions:
    """Encode supported controls without conflating effort and thinking mode."""

    if output_effort is not None:
        if not isinstance(
            output_effort, str
        ) or output_effort.strip().lower() not in _NATIVE_EFFORTS | {"minimal", "none"}:
            raise NativeMessagesError("Unsupported client output effort.")
        output_effort = output_effort.strip().lower()
    limit = DEFAULT_MESSAGES_OUTPUT_TOKENS if max_tokens is None else max_tokens
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise NativeMessagesError("Messages max_tokens must be a positive integer.")
    if capabilities.max_output_tokens is not None:
        limit = min(limit, capabilities.max_output_tokens)
    native_mode = thinking.type if thinking is not None else None
    if thinking is not None:
        if thinking.model_extra:
            raise NativeMessagesError("Unsupported native thinking option.")
        if native_mode not in {None, "enabled", "disabled", "adaptive"}:
            raise NativeMessagesError(
                f"Unsupported Messages thinking type: {native_mode!r}"
            )
        if thinking.budget_tokens is not None and thinking.budget_tokens <= 0:
            raise NativeMessagesError("Thinking budget must be a positive integer.")
    budget = reasoning.budget_tokens
    # The application owns intent. A native mode is only a wire-format hint.
    if reasoning.control is ReasoningControl.OFF:
        mode = "disabled"
    elif not reasoning.requests_reasoning or native_mode in {"enabled", "adaptive"}:
        mode = native_mode
    elif capabilities.adaptive_thinking in {"optional", "required"}:
        mode = "enabled" if budget is not None else "adaptive"
    elif capabilities.adaptive_thinking == "unsupported":
        mode = "enabled"
    elif reasoning.control is ReasoningControl.DEFAULT and budget is None:
        mode = None
    else:
        raise NativeMessagesError(
            "This model does not advertise its thinking mode. Supply native "
            "thinking.type or use an advertised reasoning effort without forcing thinking."
        )
    if mode == "adaptive" and capabilities.adaptive_thinking == "unsupported":
        raise NativeMessagesError("This model does not support adaptive thinking.")
    if mode == "enabled" and capabilities.adaptive_thinking == "required":
        raise NativeMessagesError(
            "This model requires adaptive rather than manual thinking."
        )
    if mode == "adaptive" and (
        budget is not None
        or (thinking is not None and thinking.budget_tokens is not None)
    ):
        raise NativeMessagesError(
            "Adaptive thinking cannot represent an exact token budget."
        )

    wire_thinking: JsonObject | None = None
    if mode is not None:
        wire_thinking = {"type": mode}
        if mode == "enabled":
            exact = budget
            if (
                exact is None
                and thinking is not None
                and not reasoning.requests_reasoning
            ):
                exact = thinking.budget_tokens
            derived = max(_MIN_THINKING_BUDGET, reasoning.numeric_budget_tokens or 2048)
            effective = exact if exact is not None else min(derived, limit - 1)
            if effective < _MIN_THINKING_BUDGET or effective >= limit:
                raise NativeMessagesError(
                    "Manual thinking budget must be at least 1024 and below max_tokens."
                )
            wire_thinking["budget_tokens"] = effective
        if mode != "disabled" and thinking is not None and thinking.display is not None:
            wire_thinking["display"] = thinking.display

    effort = (
        reasoning.effort.value
        if reasoning.effort is not None
        else output_effort
        if reasoning.control is ReasoningControl.DEFAULT
        else None
    )
    if effort == "none" and reasoning.control is ReasoningControl.OFF:
        effort = None
    if (
        mode == "enabled"
        and capabilities.supports_output_effort is False
        and reasoning.effort is not None
    ):
        effort = None  # Already represented by the resolved manual token budget.
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise NativeMessagesError("Messages output effort must be a nonempty string.")
    if isinstance(effort, str):
        supported = capabilities.supported_efforts
        if effort == "minimal" and supported is not None and "low" in supported:
            effort = "low"
        if effort == "minimal" or (supported is None and effort not in _NATIVE_EFFORTS):
            raise NativeMessagesError(
                f"This model does not advertise a native mapping for effort {effort!r}."
            )
        if supported is not None and effort not in supported:
            raise NativeMessagesError(f"This model does not support effort {effort!r}.")
        if capabilities.supports_output_effort is False:
            raise NativeMessagesError("This model does not support output effort.")
        elif (
            capabilities.supports_output_effort is None
            and supported is None
            and native_mode is None
            and mode is None
        ):
            raise NativeMessagesError(
                "This model does not advertise output effort support."
            )
    return NativeMessagesOptions(
        model=model, max_tokens=limit, thinking=wire_thinking, output_effort=effort
    )
