"""Request builder for LM Studio provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import AnthropicToOpenAIConverter
from providers.common.utils import set_if_not_none

LMSTUDIO_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for LM Studio."""
    logger.debug(
        "LMSTUDIO_REQUEST: conversion start model=%s msgs=%d",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages, include_reasoning_for_openrouter=False
    )

    # Add system prompt
    system = getattr(request_data, "system", None)
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {
        "model": request_data.model,
        "messages": messages,
    }

    max_tokens = getattr(request_data, "max_tokens", None)
    set_if_not_none(body, "max_tokens", max_tokens or LMSTUDIO_DEFAULT_MAX_TOKENS)

    set_if_not_none(body, "temperature", getattr(request_data, "temperature", None))
    set_if_not_none(body, "top_p", getattr(request_data, "top_p", None))

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = tool_choice

    logger.debug(
        "LMSTUDIO_REQUEST: conversion done model=%s msgs=%d tools=%d",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
