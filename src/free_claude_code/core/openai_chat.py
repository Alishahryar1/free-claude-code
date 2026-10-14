"""OpenAI Chat Completions wire-history helpers."""

import json
from collections.abc import Mapping
from typing import Any

from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.token_estimation import estimate_text_tokens

IMAGE_TOOL_RESULT_MARKER = "[Image-bearing tool output follows in user content.]"


class ChatToolResultImages(dict[str, JsonValue]):
    """Relocated tool output, retaining turn ownership until serialization."""

    __slots__ = ()


def image_tool_result_label(tool_call_id: str) -> str:
    """Label relocated visual output with its source function-call identity."""
    return (
        "Image-bearing output for tool call "
        f"{json.dumps(tool_call_id, ensure_ascii=False)}:"
    )


def computer_screenshot_label(call_id: str) -> str:
    """Label a Responses computer screenshot relocated into Chat user content."""
    return f"Computer screenshot for call {json.dumps(call_id, ensure_ascii=False)}:"


class _SyntheticChatToolTurnBoundary(dict[str, JsonValue]):
    """Identify an FCC-inserted assistant boundary until wire serialization."""

    __slots__ = ()


def is_synthetic_chat_tool_turn_boundary(message: object) -> bool:
    """Return whether FCC inserted this message to close a Chat tool turn."""
    return isinstance(message, _SyntheticChatToolTurnBoundary)


def close_chat_tool_result_turns(messages: list[JsonObject]) -> list[JsonObject]:
    """Close completed Chat tool rounds before subsequent user input."""
    result: list[JsonObject] = []
    for message in messages:
        if (
            message.get("role") == "user"
            and result
            and result[-1].get("role") == "tool"
        ):
            # Some OpenAI-compatible chat templates reject a user role directly
            # after tool output. Non-empty whitespace closes the assistant turn
            # without inventing model content.
            result.append(_SyntheticChatToolTurnBoundary(role="assistant", content=" "))
        result.append(message)
    return result


def estimate_chat_input_tokens(body: Mapping[str, Any]) -> int:
    """Estimate prompt tokens for an OpenAI Chat Completions request body.

    The chat body is the request the provider actually receives, so counting
    it covers every body-mutating path — continuation and recovery paths grow
    the message list after the executor's original count, and only the body
    reflects what is sent. Payloads count like the Anthropic estimator:
    text content and tool-call arguments as text, base64 images by payload
    size, with per-message overhead.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if isinstance(message, Mapping):
            total += _chat_message_tokens(message)
    tools = body.get("tools")
    if isinstance(tools, list):
        total += sum(_json_tokens(tool) for tool in tools)
    return max(1, total + 4 * len(messages))


def _chat_message_tokens(message: Mapping[str, Any]) -> int:
    total = 0
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_text_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, Mapping):
                total += _chat_part_tokens(part)
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        total += estimate_text_tokens(reasoning)
    for tool_call in message.get("tool_calls") or ():
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            total += estimate_text_tokens(str(function.get("name") or ""))
            total += estimate_text_tokens(str(function.get("arguments") or ""))
    return total


def _chat_part_tokens(part: Mapping[str, Any]) -> int:
    part_type = part.get("type")
    if part_type == "text":
        text = part.get("text")
        return estimate_text_tokens(text) if isinstance(text, str) else 0
    if part_type == "image_url":
        url = part.get("image_url")
        if isinstance(url, Mapping):
            value = url.get("url")
            if isinstance(value, str) and value.startswith("data:"):
                return max(85, len(value) // 3000)
        return 765
    # Unrecognized part types (audio, future modalities) count by
    # serialization, the same fallback the Anthropic estimator uses.
    return _json_tokens(part)


def _json_tokens(value: object) -> int:
    return estimate_text_tokens(json.dumps(value, default=str))
