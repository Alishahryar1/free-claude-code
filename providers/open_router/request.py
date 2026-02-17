"""Request builder for OpenRouter provider."""

from typing import Any, Dict

from providers.common.message_converter import AnthropicToOpenAIConverter
from loguru import logger


OPENROUTER_DEFAULT_MAX_TOKENS = 81920


def _set_if_not_none(body: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        body[key] = value


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for OpenRouter."""
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model=%s msgs=%d",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages, include_reasoning_for_openrouter=True
    )

    # Add system prompt
    system = getattr(request_data, "system", None)
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: Dict[str, Any] = {
        "model": request_data.model,
        "messages": messages,
    }

    max_tokens = getattr(request_data, "max_tokens", None)
    _set_if_not_none(body, "max_tokens", max_tokens or OPENROUTER_DEFAULT_MAX_TOKENS)

    _set_if_not_none(body, "temperature", getattr(request_data, "temperature", None))
    _set_if_not_none(body, "top_p", getattr(request_data, "top_p", None))

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = tool_choice

    # OpenRouter reasoning: extra_body={"reasoning": {"enabled": True}}
    extra_body: Dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    thinking = getattr(request_data, "thinking", None)
    thinking_enabled = (
        thinking.enabled if thinking and hasattr(thinking, "enabled") else True
    )
    if thinking_enabled:
        extra_body.setdefault("reasoning", {"enabled": True})

    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "OPENROUTER_REQUEST: conversion done model=%s msgs=%d tools=%d",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
