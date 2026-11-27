"""Content block helpers for Anthropic-compatible payloads."""

import json


def get_block_attr(block: object, attr: str, default: object = None) -> object:
    """Get an attribute from a Pydantic model, lightweight object, or dict."""
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def get_block_type(block: object) -> str | None:
    """Return a content block type when present."""
    kind = get_block_attr(block, "type")
    return kind if isinstance(kind, str) else None


def extract_text_from_content(content: object) -> str:
    """Extract concatenated text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = get_block_attr(block, "text", "")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)
    return ""


def streamed_tool_arguments(joined: str, start_input: object) -> str:
    """Return streamed tool arguments, using the start input when none streamed.

    A zero-argument call streams one empty ``input_json_delta``. Like Anthropic's
    SDK, only non-empty joined deltas replace the ``content_block_start`` input.
    """
    if joined:
        return joined
    return json.dumps(start_input, ensure_ascii=False, separators=(",", ":"))
