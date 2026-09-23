"""NVIDIA NIM request option injection."""

from copy import deepcopy
from typing import Any

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic import ReasoningReplayMode, set_if_not_none
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.providers.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

from .tool_schema import sanitize_nim_tool_schemas

NIM_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="NIM",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
)


def build_nim_request_body(
    request_data: MessagesRequest, nim: NimSettings, *, reasoning: ReasoningPolicy
) -> dict[str, Any]:
    """Build OpenAI-format request body from Anthropic request plus NIM settings."""
    return build_openai_chat_request_body(
        request_data,
        reasoning=reasoning,
        policy=NIM_REQUEST_POLICY,
        postprocessors=(
            lambda body, request, policy: _apply_nim_messages_request_options(
                body,
                request,
                policy,
                nim=nim,
            ),
        ),
    )


def _apply_nim_messages_request_options(
    body: dict[str, Any],
    request_data: MessagesRequest,
    reasoning: ReasoningPolicy,
    *,
    nim: NimSettings,
) -> None:
    """Copy Messages-only fields, then apply the common NIM finalizer."""
    extra_body = deepcopy(request_data.extra_body or {})
    _set_extra(extra_body, "top_k", request_data.top_k, ignore_value=-1)
    if extra_body:
        body["extra_body"] = extra_body
    apply_nim_request_options(body, reasoning, nim=nim)


def apply_nim_request_options(
    body: dict[str, Any],
    reasoning: ReasoningPolicy,
    *,
    nim: NimSettings,
) -> None:
    """Apply source-independent NIM policy to one Chat body."""
    sanitize_nim_tool_schemas(body)

    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    if body.get("temperature") is None and nim.temperature is not None:
        body["temperature"] = nim.temperature
    body["top_p"] = nim.top_p

    if "stop" not in body and nim.stop:
        body["stop"] = nim.stop

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    body["parallel_tool_calls"] = nim.parallel_tool_calls

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

    if reasoning.control is ReasoningControl.OFF or reasoning.requests_reasoning:
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if isinstance(chat_template_kwargs, dict):
            enabled = reasoning.control is not ReasoningControl.OFF
            chat_template_kwargs["thinking"] = enabled
            chat_template_kwargs["enable_thinking"] = enabled
            if enabled and (budget := reasoning.numeric_budget_tokens) is not None:
                chat_template_kwargs["reasoning_budget"] = budget

    _set_extra(extra_body, "top_k", nim.top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)

    if extra_body:
        body["extra_body"] = extra_body


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
