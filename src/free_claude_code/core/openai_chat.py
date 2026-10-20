"""OpenAI Chat Completions wire-history helpers."""

import json

from free_claude_code.core.json_types import JsonObject, JsonValue

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
