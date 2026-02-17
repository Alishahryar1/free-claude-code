"""Shared provider utilities used by NIM, OpenRouter, and LM Studio."""

from .sse_builder import SSEBuilder, ContentBlockManager, map_stop_reason
from .think_parser import ThinkTagParser, ContentType, ContentChunk
from .heuristic_tool_parser import HeuristicToolParser
from .message_converter import (
    AnthropicToOpenAIConverter,
    get_block_attr,
    get_block_type,
)
from .error_mapping import map_error

__all__ = [
    "SSEBuilder",
    "ContentBlockManager",
    "map_stop_reason",
    "ThinkTagParser",
    "ContentType",
    "ContentChunk",
    "HeuristicToolParser",
    "AnthropicToOpenAIConverter",
    "get_block_attr",
    "get_block_type",
    "map_error",
]
