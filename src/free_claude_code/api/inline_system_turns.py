"""Explain or hoist system turns inside Messages ``messages``."""

import re
from collections.abc import Mapping

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.json_types import JsonObject, JsonValue

_CLAUDE_CODE_VERSION = re.compile(r"\bclaude-cli/(\d+(?:\.\d+)+)")
_SYSTEM_JOINER = "\n\n"


def hoist_inline_system_turns(body: JsonObject) -> JsonObject:
    """Move ``system`` turns from ``messages`` into the top-level ``system`` field.

    Claude Code can inject mid-conversation reminders as ``{"role": "system"}``
    entries. Strict Anthropic and OpenAI-compatible providers reject that role
    inside ``messages``, so the opt-in ``HOIST_INLINE_SYSTEM_TURNS`` setting
    lifts their text into ``system`` and drops the original turns.

    Only text is preserved: block metadata such as ``cache_control``, ``clear_at``,
    or tool changes cannot be expressed in the top-level field. The input body is
    not mutated, and the remaining messages keep their original order.
    """

    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    kept: list[JsonValue] = []
    texts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "system":
            texts.extend(_system_text_parts(message.get("content")))
            continue
        kept.append(message)
    if len(kept) == len(messages):
        return body
    hoisted: JsonObject = {**body, "messages": kept}
    joined = _SYSTEM_JOINER.join(part for part in texts if part.strip())
    if joined:
        _merge_system_text(hoisted, joined)
    return hoisted


def _system_text_parts(content: JsonValue) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return parts


def _merge_system_text(body: JsonObject, text: str) -> None:
    existing = body.get("system")
    if isinstance(existing, str):
        body["system"] = (
            f"{existing}{_SYSTEM_JOINER}{text}" if existing.strip() else text
        )
        return
    if isinstance(existing, list):
        blocks: list[JsonValue] = list(existing)
        blocks.append({"type": "text", "text": text})
        body["system"] = blocks
        return
    if existing is None:
        body["system"] = text


def reject_inline_system_turns(
    body: Mapping[str, object], headers: Mapping[str, str] | None
) -> None:
    """Raise an actionable error when ``messages`` holds a ``system`` turn.

    Older Claude Code builds send these turns and cannot recover from the
    rejection, so the message says how to fix it. Newer builds retry without
    them on any 400, so the status stays an ordinary invalid request.
    """

    messages = body.get("messages")
    if not isinstance(messages, list) or not any(
        isinstance(message, Mapping) and message.get("role") == "system"
        for message in messages
    ):
        return
    user_agent = headers.get("user-agent", "") if headers is not None else ""
    if match := _CLAUDE_CODE_VERSION.search(user_agent):
        raise InvalidRequestError(
            f"Claude Code {match.group(1)} sent a system message inside `messages`, "
            "which Free Claude Code can't accept. Run `claude update`, then "
            "restart Claude Code."
        )
    raise InvalidRequestError(
        "System messages belong in the top-level `system` field, not in "
        "`messages`. If you're using Claude Code, run `claude update`, then "
        "restart it."
    )
