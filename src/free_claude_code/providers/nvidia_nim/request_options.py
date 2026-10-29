"""NVIDIA NIM request adaptation."""

from copy import deepcopy
from typing import Any

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode, set_if_not_none
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

from .tool_schema import sanitize_nim_tool_schemas

_TEXT_ONLY_NIM_MODELS = frozenset({"z-ai/glm-5.3"})
_IMAGE_OMITTED_TEXT = "[Image omitted: this model does not accept image input.]"

NIM_REASONING = NamedEffortReasoning(
    tuple((effort, effort.value) for effort in ReasoningEffort),
    disabled_value="none",
    enabled_value="high",
)

NIM_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="NIM",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


def build_nim_request_body(
    request_data: MessagesRequest, *, reasoning: ReasoningPolicy
) -> dict[str, Any]:
    """Build an OpenAI-format request with NIM protocol adaptation."""
    return build_openai_chat_request_body(
        request_data,
        reasoning=reasoning,
        policy=NIM_REQUEST_POLICY,
        postprocessors=(_apply_nim_messages_request_options,),
    )


def _apply_nim_messages_request_options(
    body: dict[str, Any],
    request_data: MessagesRequest,
    reasoning: ReasoningPolicy,
) -> None:
    """Copy Messages-only fields, then apply the common NIM finalizer."""
    extra_body = deepcopy(request_data.extra_body or {})
    _set_extra(extra_body, "top_k", request_data.top_k, ignore_value=-1)
    if extra_body:
        body["extra_body"] = extra_body
    apply_nim_request_options(body, reasoning)


def apply_nim_request_options(
    body: dict[str, Any],
    reasoning: ReasoningPolicy,
) -> None:
    """Apply source-independent NIM policy to one Chat body."""
    _replace_images_for_text_only_model(body)
    sanitize_nim_tool_schemas(body)

    max_tokens = body.get("max_tokens")
    default_max_tokens = NIM_REQUEST_POLICY.default_max_tokens
    if max_tokens is None:
        max_tokens = default_max_tokens
    elif default_max_tokens is not None:
        max_tokens = min(max_tokens, default_max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    request_extra = body.get("extra_body")
    extra_body: dict[str, Any] = (
        deepcopy(request_extra) if isinstance(request_extra, dict) else {}
    )
    for key in (
        "reasoning",
        "reasoning_budget",
        "reasoning_effort",
        "reasoning_tokens",
        "thinking",
        "thinking_budget_tokens",
    ):
        extra_body.pop(key, None)
    request_template_kwargs = extra_body.get("chat_template_kwargs")
    if isinstance(request_template_kwargs, dict):
        for key in ("thinking", "enable_thinking", "reasoning_budget"):
            request_template_kwargs.pop(key, None)
        if not request_template_kwargs:
            extra_body.pop("chat_template_kwargs", None)

    NIM_REASONING.encode(body, reasoning)

    if extra_body:
        body["extra_body"] = extra_body
    else:
        body.pop("extra_body", None)


def _replace_images_for_text_only_model(body: dict[str, Any]) -> None:
    if body.get("model") not in _TEXT_ONLY_NIM_MODELS:
        return
    for message in body.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        message["content"] = [
            {"type": "text", "text": _IMAGE_OMITTED_TEXT}
            if isinstance(part, dict) and part.get("type") == "image_url"
            else part
            for part in content
        ]


def _set_extra(
    extra_body: dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value
