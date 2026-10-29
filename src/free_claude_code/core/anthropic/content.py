"""Content block helpers for Anthropic-compatible payloads."""

from typing import Any, Literal

TOOL_CHANGE_BLOCK_TYPES = frozenset({"tool_addition", "tool_removal"})


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


def tool_change_kind(block: Any) -> Literal["reference", "definition"] | None:
    """Classify a ``tool_addition`` or ``tool_removal`` block by its tool form.

    Claude Code announces deferred tools with these blocks. A ``tool_reference``
    names a tool already declared in ``tools``, so a converter whose upstream sees
    every declared tool can drop it. A ``tool_definition`` carries a tool inline,
    and dropping it would lose that tool.
    """
    if get_block_type(block) not in TOOL_CHANGE_BLOCK_TYPES:
        return None
    tool = get_block_attr(block, "tool", {})
    if get_block_attr(tool, "type") == "tool_reference":
        return "reference"
    return "definition"


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
