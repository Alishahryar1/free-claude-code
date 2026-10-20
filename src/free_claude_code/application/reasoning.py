"""Resolve client reasoning input and FCC configuration exactly once."""

from collections.abc import Mapping

from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import MessagesRequest, ThinkingConfig
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    responses_reasoning_policy,
)
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)


def resolve_reasoning_policy(
    request: MessagesRequest,
    preference: ReasoningPreference,
) -> ReasoningPolicy:
    """Apply one resolved configuration preference to the client request."""

    return _apply_reasoning_preference(
        client_reasoning_policy(request),
        preference,
    )


def resolve_responses_reasoning_policy(
    request: OpenAIResponsesRequest,
    preference: ReasoningPreference,
) -> ReasoningPolicy:
    """Apply one resolved configuration preference to a Responses request."""

    return _apply_reasoning_preference(
        client_responses_reasoning_policy(request),
        preference,
    )


def resolve_fallback_reasoning(
    original: ReasoningPolicy,
    preference: ReasoningPreference,
    *,
    client: ReasoningPolicy,
) -> ReasoningPolicy:
    """Resolve a candidate from original intent, never a previous fallback."""

    # Classifier requests keep the handler's tolerant policy on every candidate.
    if original.control is ReasoningControl.PREFER_OFF:
        return original
    if preference is ReasoningPreference.INHERIT:
        return original
    if preference is ReasoningPreference.CLIENT:
        return client
    return _apply_reasoning_preference(original, preference)


def prepare_fallback_request[T: MessagesRequest | OpenAIResponsesRequest](
    original: T, *, model: str, preference: ReasoningPreference
) -> T:
    """Copy a candidate and remove computation controls replaced by its policy."""

    request = original.model_copy(update={"model": model}, deep=True)
    if preference in {ReasoningPreference.INHERIT, ReasoningPreference.CLIENT}:
        return request
    if isinstance(request, MessagesRequest):
        if request.thinking is not None:
            thinking = request.thinking.model_dump(exclude_unset=True)
            for key in ("type", "enabled", "budget_tokens"):
                thinking.pop(key, None)
            request.thinking = (
                ThinkingConfig.model_validate(thinking) if thinking else None
            )
        if request.output_config is not None:
            request.output_config.pop("effort", None)
    elif request.reasoning is not None:
        request.reasoning.pop("effort", None)
    return request


def _apply_reasoning_preference(
    client_policy: ReasoningPolicy,
    preference: ReasoningPreference,
) -> ReasoningPolicy:
    """Apply FCC's route preference to already-parsed client intent."""

    if preference is ReasoningPreference.INHERIT:
        raise ValueError("Reasoning preference must be resolved before application.")
    if preference is ReasoningPreference.OFF:
        return ReasoningPolicy.off()
    if preference is not ReasoningPreference.CLIENT:
        return ReasoningPolicy.on(effort=ReasoningEffort(preference.value))
    return client_policy


def client_reasoning_policy(request: MessagesRequest) -> ReasoningPolicy:
    """Return the lossless reasoning intent expressed by one client request."""

    budget_tokens = _positive_budget(request.thinking)
    thinking_control = _thinking_control(
        request.thinking,
        budget_tokens=budget_tokens,
    )
    effort, effort_disables = _output_effort(request.output_config)

    if effort_disables:
        return ReasoningPolicy.off()
    if thinking_control is ReasoningControl.OFF:
        return ReasoningPolicy(
            control=ReasoningControl.OFF,
            effort=effort,
        )
    if thinking_control is ReasoningControl.ON or budget_tokens is not None:
        return ReasoningPolicy.on(
            effort=effort,
            budget_tokens=budget_tokens,
        )
    return ReasoningPolicy(
        control=ReasoningControl.DEFAULT,
        effort=effort,
    )


def client_responses_reasoning_policy(
    request: OpenAIResponsesRequest,
) -> ReasoningPolicy:
    """Return the supported reasoning intent from one Responses request."""

    return responses_reasoning_policy(request.reasoning)


def _thinking_control(
    thinking: ThinkingConfig | None,
    *,
    budget_tokens: int | None,
) -> ReasoningControl:
    if thinking is None:
        return ReasoningControl.DEFAULT
    if thinking.type == "disabled" or (
        "enabled" in thinking.model_fields_set and thinking.enabled is False
    ):
        return ReasoningControl.OFF
    if (
        thinking.type in {"adaptive", "enabled"}
        or ("enabled" in thinking.model_fields_set and thinking.enabled is True)
        or budget_tokens is not None
    ):
        return ReasoningControl.ON
    return ReasoningControl.DEFAULT


def _output_effort(value: object) -> tuple[ReasoningEffort | None, bool]:
    if not isinstance(value, Mapping):
        return None, False
    raw = value.get("effort")
    if not isinstance(raw, str):
        return None, False
    normalized = raw.strip().lower()
    if normalized == "none":
        return None, True
    try:
        return ReasoningEffort(normalized), False
    except ValueError:
        return None, False


def _positive_budget(thinking: ThinkingConfig | None) -> int | None:
    if thinking is None:
        return None
    value = thinking.budget_tokens
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
