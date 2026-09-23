"""Request-body policy for OpenAI-compatible chat providers."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import (
    ReasoningReplayMode,
    build_base_request_body,
)
from free_claude_code.core.anthropic.conversion import OpenAIConversionError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_tool_names import (
    OpenAIToolNameCodec,
    encode_openai_chat_tool_names,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.history_replay import validate_history

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]
OpenAIChatPostprocessor = Callable[
    [dict[str, Any], MessagesRequest, ReasoningPolicy], None
]
ExtraBodyValidator = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class OpenAIChatRequestPolicy:
    """Provider policy for Anthropic-to-OpenAI chat request conversion."""

    provider_name: str
    reasoning_replay: ReasoningReplayMode
    include_extra_body: bool = False
    extra_body_validator: ExtraBodyValidator | None = None
    reject_extra_body_message: str | None = None
    default_max_tokens: int | None = None
    max_tokens_field: MaxTokensField = "max_tokens"
    strip_message_names: bool = False
    unsupported_body_keys: frozenset[str] = field(default_factory=frozenset)
    normalize_n_to_one: bool = False


def build_openai_chat_request_body(
    request_data: MessagesRequest,
    *,
    reasoning: ReasoningPolicy,
    policy: OpenAIChatRequestPolicy,
    postprocessors: Iterable[OpenAIChatPostprocessor] = (),
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat request body from an Anthropic request."""
    validate_history(request_data.model_dump(mode="json"))
    logger.debug(
        "{}_REQUEST: conversion start model={} msgs={}",
        policy.provider_name,
        request_data.model,
        len(request_data.messages),
    )
    try:
        body = build_base_request_body(
            request_data,
            default_max_tokens=policy.default_max_tokens,
            reasoning_replay=policy.reasoning_replay,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    request_extra = request_data.extra_body
    if isinstance(request_extra, dict) and request_extra:
        if policy.reject_extra_body_message:
            raise InvalidRequestError(policy.reject_extra_body_message)
        if policy.include_extra_body:
            extra_body = deepcopy(request_extra)
            if policy.extra_body_validator is not None:
                try:
                    policy.extra_body_validator(extra_body)
                except ValueError as exc:
                    raise InvalidRequestError(str(exc)) from exc
            body["extra_body"] = extra_body

    apply_openai_chat_body_policy(body, policy)

    for postprocess in postprocessors:
        postprocess(body, request_data, reasoning)

    encode_openai_chat_tool_names(body, OpenAIToolNameCodec.from_request(request_data))

    logger.debug(
        "{}_REQUEST: conversion done model={} msgs={} tools={}",
        policy.provider_name,
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def apply_openai_chat_body_policy(
    body: dict[str, Any], policy: OpenAIChatRequestPolicy
) -> None:
    """Apply source-independent Chat Completions body policy."""
    if policy.strip_message_names:
        _strip_message_names(body.get("messages"))

    for key in policy.unsupported_body_keys:
        body.pop(key, None)

    # unsupported_body_keys must also be stripped from a caller-supplied
    # extra_body: it becomes the SDK's own extra_body= kwarg and gets merged
    # into the wire-level JSON verbatim, so a key unsupported at the top
    # level is equally unsupported when it arrives this way (#1548).
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        for key in policy.unsupported_body_keys:
            extra_body.pop(key, None)

    if policy.max_tokens_field == "max_completion_tokens":
        _normalize_max_completion_tokens(body)

    if policy.normalize_n_to_one and body.get("n") is not None:
        body["n"] = 1


def _strip_message_names(messages: Any) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict):
            message.pop("name", None)


def _normalize_max_completion_tokens(body: dict[str, Any]) -> None:
    if "max_completion_tokens" in body:
        body.pop("max_tokens", None)
        return
    if "max_tokens" in body and body["max_tokens"] is not None:
        body["max_completion_tokens"] = body.pop("max_tokens")
