"""Request builder for NVIDIA NIM provider."""

from typing import Any, Dict

from config.nim import NimSettings
from providers.common.message_converter import AnthropicToOpenAIConverter
from loguru import logger


def _set_if_not_none(body: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        body[key] = value


def _set_extra(
    extra_body: Dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value


def build_request_body(request_data: Any, nim: NimSettings) -> dict:
    """Build OpenAI-format request body from Anthropic request."""
    logger.debug(
        "NIM_REQUEST: conversion start model=%s msgs=%d",
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

    # max_tokens with optional cap
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

    # Handle non-standard parameters via extra_body
    extra_body: Dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    # Handle thinking/reasoning mode
    extra_body.setdefault("thinking", {"type": "enabled"})
    extra_body.setdefault("reasoning_split", True)
    extra_body.setdefault(
        "chat_template_kwargs",
        {
            "thinking": True,
            "enable_thinking": True,
            "reasoning_split": True,
            "clear_thinking": False,
        },
    )

    req_top_k = getattr(request_data, "top_k", None)
    top_k = req_top_k if req_top_k is not None else nim.top_k
    _set_extra(extra_body, "top_k", top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "return_tokens_as_token_ids", nim.return_tokens_as_token_ids)
    _set_extra(extra_body, "include_stop_str_in_output", nim.include_stop_str_in_output)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)
    _set_extra(extra_body, "reasoning_effort", nim.reasoning_effort)
    _set_extra(extra_body, "include_reasoning", nim.include_reasoning)

    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "NIM_REQUEST: conversion done model=%s msgs=%d tools=%d",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
