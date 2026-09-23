"""Content block helpers for Anthropic-compatible payloads."""

from typing import Any


def get_block_attr(block: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute from a Pydantic model, lightweight object, or dict."""
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def get_block_type(block: Any) -> str | None:
    """Return a content block type when present."""
    return get_block_attr(block, "type")


def extract_text_from_content(content: Any) -> str:
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
