"""Request builder for OpenRouter provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

OPENROUTER_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for OpenRouter."""
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model=%s msgs=%d",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        default_max_tokens=OPENROUTER_DEFAULT_MAX_TOKENS,
        include_reasoning_for_openrouter=True,
    )

    # OpenRouter reasoning: extra_body={"reasoning": {"enabled": True}}
    extra_body: dict[str, Any] = {}
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
