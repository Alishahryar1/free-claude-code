"""Request and history fields shared by the two OpenAI protocols."""

import json
from collections.abc import Mapping

from free_claude_code.core.json_types import JsonObject, JsonValue

TOOL_RESULT_ATTACHMENT_MARKER = "[Tool output attachments follow in user content.]"


def shared_openai_request_options(body: Mapping[str, JsonValue]) -> JsonObject:
    options = {
        key: body[key]
        for key in ("prompt_cache_key", "prompt_cache_retention", "safety_identifier")
        if body.get(key) is not None
    }
    cache = body.get("prompt_cache_options")
    if isinstance(cache, Mapping):
        common: JsonObject = {}
        if cache.get("mode") in ("implicit", "explicit"):
            common["mode"] = cache["mode"]
        # Both protocols support longer lifetimes per part, but only Responses
        # supports 12h as the default for the entire request.
        if cache.get("ttl") == "30m":
            common["ttl"] = "30m"
        if common:
            options["prompt_cache_options"] = common
    return options


def openai_cache_breakpoint(part: Mapping[str, JsonValue]) -> JsonObject:
    marker = part.get("prompt_cache_breakpoint")
    if not isinstance(marker, Mapping):
        return {}
    return {
        "prompt_cache_breakpoint": {
            key: marker[key] for key in ("mode", "ttl") if key in marker
        }
    }


def inline_file_fields(
    value: JsonValue, *, pdf_only: bool = False
) -> JsonObject | None:
    """Copy portable OpenAI file fields without fetching or re-encoding the data."""
    if (
        not isinstance(value, Mapping)
        or value.get("file_id") is not None
        or value.get("file_url") is not None
    ):
        return None
    data, name = value.get("file_data"), value.get("filename")
    if (
        not isinstance(data, str)
        or not data
        or (name is not None and not isinstance(name, str))
    ):
        return None
    if pdf_only:
        if data[:5].lower() == "data:":
            comma = data.find(",")
            if comma < 0 or comma + 1 == len(data):
                return None
            header = data[:comma].lower()
            if header.split(";", 1)[0] != "data:application/pdf" or not header.endswith(
                ";base64"
            ):
                return None
        elif not (isinstance(name, str) and name.lower().endswith(".pdf")):
            return None
    fields: JsonObject = {"file_data": data}
    if name is not None:
        fields["filename"] = name
    return fields


class ChatToolResultAttachments(dict[str, JsonValue]):
    """Relocated tool output, retaining turn ownership until serialization."""

    __slots__ = ()


def tool_result_attachment_label(tool_call_id: str) -> str:
    """Label relocated attachments with their source tool call."""
    return f"Attachments from tool call {json.dumps(tool_call_id, ensure_ascii=False)}:"


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
