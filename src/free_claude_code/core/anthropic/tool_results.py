"""Structured Anthropic tool-result content."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from free_claude_code.core.anthropic.content import (
    get_block_attr,
    get_block_type,
    require_block_fields,
)
from free_claude_code.core.anthropic.image_sources import (
    portable_anthropic_image_url,
)
from free_claude_code.core.failures import UnsupportedRequestFeature


@dataclass(frozen=True, slots=True)
class ToolResultText:
    text: str


@dataclass(frozen=True, slots=True)
class ToolResultImage:
    url: str


@dataclass(frozen=True, slots=True)
class DecomposedToolResult:
    parts: tuple[ToolResultText | ToolResultImage, ...]
    has_images: bool


def decompose_tool_result_content(
    content: object, *, preserve_features: bool = False
) -> DecomposedToolResult:
    """Return ordered text/image values for one Anthropic tool result."""
    if content is None:
        blocks: Sequence[object] = ()
    elif isinstance(content, (str, Mapping)) or hasattr(content, "type"):
        blocks = (content,)
    elif isinstance(content, Sequence):
        blocks = content
    else:
        blocks = (content,)

    parts: list[ToolResultText | ToolResultImage] = []
    for block in blocks:
        block_type = get_block_type(block)
        if block_type == "text":
            if preserve_features:
                require_block_fields(
                    block, {"type", "text"}, "Converted tool result text"
                )
            text = get_block_attr(block, "text", "")
            if isinstance(text, str):
                parts.append(ToolResultText(text))
            else:
                parts.append(ToolResultText(str(text)))
        elif block_type == "image":
            if preserve_features:
                require_block_fields(
                    block, {"type", "source"}, "Converted tool result image"
                )
            source = get_block_attr(block, "source")
            if preserve_features:
                require_block_fields(
                    source,
                    {"type", "data", "media_type", "url"},
                    "Converted image source",
                )
            parts.append(ToolResultImage(portable_anthropic_image_url(source)))
        elif isinstance(block, str):
            parts.append(ToolResultText(block))
        elif isinstance(block, Mapping):
            if preserve_features and block_type is not None:
                raise UnsupportedRequestFeature(
                    f"Converted tool results cannot preserve {block_type!r} content."
                )
            parts.append(ToolResultText(json.dumps(dict(block), ensure_ascii=False)))
        else:
            parts.append(ToolResultText(str(block)))

    result = tuple(parts)
    return DecomposedToolResult(
        parts=result,
        has_images=any(isinstance(part, ToolResultImage) for part in result),
    )
