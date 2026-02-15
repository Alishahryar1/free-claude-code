"""Request builder for OpenRouter provider.

OpenRouter uses OpenAI-compatible API. We use standard params only
(no NIM-specific extras like reasoning_effort, thinking, etc.).
"""

import logging
from typing import Any, Dict

from config.nim import NimSettings
from providers.nvidia_nim.utils.message_converter import AnthropicToOpenAIConverter

logger = logging.getLogger(__name__)


def _set_if_not_none(body: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        body[key] = value


def build_request_body(request_data: Any, nim: NimSettings) -> dict:
    """Build OpenAI-format request body from Anthropic request.

    Uses only standard OpenAI/OpenRouter params (no NIM-specific extras).
    """
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model=%s msgs=%d",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    messages = AnthropicToOpenAIConverter.convert_messages(request_data.messages)

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

    # Standard params supported by OpenRouter
    max_tokens = getattr(request_data, "max_tokens", None)
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    _set_if_not_none(body, "max_tokens", max_tokens)

    req_temperature = getattr(request_data, "temperature", None)
    temperature = req_temperature if req_temperature is not None else nim.temperature
    _set_if_not_none(body, "temperature", temperature)

    req_top_p = getattr(request_data, "top_p", None)
    top_p = req_top_p if req_top_p is not None else nim.top_p
    _set_if_not_none(body, "top_p", top_p)

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences
    elif nim.stop:
        body["stop"] = nim.stop

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = tool_choice

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    body["parallel_tool_calls"] = nim.parallel_tool_calls

    logger.debug(
        "OPENROUTER_REQUEST: conversion done model=%s msgs=%d tools=%d",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
