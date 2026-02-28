"""Request utility functions for API route handlers.

Contains token counting for API requests.
"""

import json
from typing import Any

import tiktoken
from loguru import logger

from providers.common import get_block_attr

ENCODER = tiktoken.get_encoding("cl100k_base")

__all__ = ["get_token_count"]


def get_token_count(
    messages: list,
    system: str | list | None = None,
    tools: list | None = None,
) -> int:
    """Estimate token count for a request.

    Uses tiktoken cl100k_base encoding to estimate token usage.
    Includes system prompt, messages, tools, and per-message overhead.
    """
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                text = get_block_attr(block, "text", "")
                if text:
                    total_tokens += len(ENCODER.encode(str(text)))
        total_tokens += 4  # System block formatting overhead

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                b_type = get_block_attr(block, "type") or None

                if b_type == "text":
                    text = get_block_attr(block, "text", "")
                    total_tokens += len(ENCODER.encode(str(text)))
                elif b_type == "thinking":
                    thinking = get_block_attr(block, "thinking", "")
                    total_tokens += len(ENCODER.encode(str(thinking)))
                elif b_type == "tool_use":
                    name = get_block_attr(block, "name", "")
                    inp = get_block_attr(block, "input", {})
                    block_id = get_block_attr(block, "id", "")
                    total_tokens += len(ENCODER.encode(str(name)))
                    total_tokens += len(ENCODER.encode(json.dumps(inp)))
                    total_tokens += len(ENCODER.encode(str(block_id)))
                    total_tokens += 15
                elif b_type == "image":
                    source = get_block_attr(block, "source")
                    if isinstance(source, dict):
                        data = source.get("data") or source.get("base64") or ""
                        if data:
                            total_tokens += max(85, len(data) // 3000)
                        else:
                            total_tokens += 765
                    else:
                        total_tokens += 765
                elif b_type == "tool_result":
                    content = get_block_attr(block, "content", "")
                    tool_use_id = get_block_attr(block, "tool_use_id", "")
                    if isinstance(content, str):
                        total_tokens += len(ENCODER.encode(content))
                    else:
                        total_tokens += len(ENCODER.encode(json.dumps(content)))
                    total_tokens += len(ENCODER.encode(str(tool_use_id)))
                    total_tokens += 8
                else:
                    logger.debug(
                        "Unexpected block type %r, falling back to json/str encoding",
                        b_type,
                    )
                    try:
                        total_tokens += len(ENCODER.encode(json.dumps(block)))
                    except TypeError, ValueError:
                        total_tokens += len(ENCODER.encode(str(block)))

    if tools:
        for tool in tools:
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(ENCODER.encode(tool_str))

    total_tokens += len(messages) * 4
    if tools:
        total_tokens += len(tools) * 5

    return max(1, total_tokens)
