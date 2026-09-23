"""Copilot control encoding from advertised model capabilities."""

from dataclasses import replace

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.reasoning_compatibility import (
    prepare_messages_reasoning,
)

from .types import CopilotModel

PROVIDER_NAME = "GITHUB_COPILOT"
REPLAY_SCOPE = "github_copilot/anthropic_messages"


def chat_profile(model: CopilotModel) -> OpenAIChatProfile:
    advertised = model.supported_efforts or ()
    return OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name=PROVIDER_NAME,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message="Copilot Chat does not support extra_body overrides.",
        ),
        NamedEffortReasoning(
            efforts=tuple(
                (effort, effort.value)
                for effort in ReasoningEffort
                if effort.value in advertised
            ),
            disabled_value="none" if "none" in advertised else None,
            enabled_value=model.default_effort
            if model.default_effort in advertised and model.default_effort != "none"
            else None,
        ),
        reasoning_delta_fallback_field="reasoning",
    )


def non_messages_reasoning(
    request: MessagesRequest | OpenAIResponsesRequest,
    policy: ReasoningPolicy,
    model: CopilotModel,
) -> ReasoningPolicy:
    """Reject controls these egresses cannot encode before any inference."""
    if policy.control is ReasoningControl.PREFER_OFF and isinstance(
        request, MessagesRequest
    ):
        prepared, wire_policy = prepare_messages_reasoning(
            request,
            policy,
            model_info=model.info,
            can_disable="none" in (model.supported_efforts or ()),
            normal_max_tokens=None,
        )
        non_messages_reasoning(prepared, wire_policy, model)
        return policy
    raw_effort = (
        (request.output_config.get("effort") if request.output_config else None)
        if isinstance(request, MessagesRequest)
        else (request.reasoning.get("effort") if request.reasoning else None)
    )
    if raw_effort is not None and (
        not isinstance(raw_effort, str)
        or raw_effort.strip().lower()
        not in {"none", *(effort.value for effort in ReasoningEffort)}
    ):
        raise InvalidRequestError("Unsupported Copilot reasoning effort.")
    if policy.budget_tokens is not None or (
        isinstance(request, MessagesRequest)
        and request.thinking is not None
        and request.thinking.budget_tokens is not None
        and policy.control is not ReasoningControl.OFF
    ):
        raise InvalidRequestError(
            "This Copilot endpoint cannot represent an exact thinking token budget."
        )
    advertised = model.supported_efforts or ()
    if policy.control is ReasoningControl.OFF:
        if "none" not in advertised:
            raise InvalidRequestError(
                "This Copilot model does not advertise a way to disable reasoning."
            )
        return policy
    effort = policy.effort
    if effort is None and policy.control is ReasoningControl.ON:
        default = model.default_effort
        if default is None or default == "none" or default not in advertised:
            raise InvalidRequestError(
                "This Copilot model does not advertise a default reasoning effort. Choose a supported effort."
            )
        try:
            effort = ReasoningEffort(default)
        except ValueError:
            raise InvalidRequestError(
                "This Copilot model's default reasoning effort is unsupported."
            ) from None
    if effort is not None:
        if (
            effort.value not in advertised
            or model.messages.supports_output_effort is False
        ):
            raise InvalidRequestError(
                f"This Copilot model does not advertise support for reasoning effort {effort.value!r}."
            )
        return replace(policy, effort=effort)
    return policy
