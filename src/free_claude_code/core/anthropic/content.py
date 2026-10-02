"""Content block helpers for Anthropic-compatible payloads."""

from collections.abc import Mapping
from typing import Any

from free_claude_code.core.request_preservation import require_supported_fields


def require_block_fields(block: Any, supported: set[str], context: str) -> None:
    value = (
        block
        if isinstance(block, Mapping)
        else block.model_dump(mode="json", exclude_unset=True)
    )
    require_supported_fields(value, supported, context)


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
