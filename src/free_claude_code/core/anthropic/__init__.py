"""Anthropic protocol helpers shared across API, providers, and integrations."""

from .content import extract_text_from_content, get_block_attr, get_block_type
from .conversion import (
    AnthropicToOpenAIConverter,
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)
from .errors import (
    anthropic_error_payload,
    anthropic_error_type_for_failure,
    anthropic_failure_payload,
    anthropic_status_for_error_type,
)
from .models import (
    ContentBlockDocument,
    ContentBlockImage,
    ContentBlockRedactedThinking,
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ContentBlockWebFetchToolResult,
    ContentBlockWebSearchToolResult,
    Message,
    MessagesRequest,
    MessagesResponse,
    SystemContent,
    ThinkingConfig,
    TokenCountRequest,
    TokenCountResponse,
    Tool,
    Usage,
)
from .request_serialization import dump_messages_request, serialize_tool_result_content
from .request_snapshot import anthropic_request_snapshot
from .sse_aggregation import aggregate_anthropic_sse_to_message
from .streaming import (
    AnthropicStreamLedger,
    StreamBlockLedger,
    ToolBlockState,
    format_sse_event,
    map_stop_reason,
)
from .thinking import ContentChunk, ContentType, ThinkTagParser
from .tokens import get_token_count
from .tools import FunctionTagToolParser, HeuristicToolParser
from .utils import set_if_not_none

__all__ = [
    "AnthropicStreamLedger",
    "AnthropicToOpenAIConverter",
    "ContentBlockDocument",
    "ContentBlockImage",
    "ContentBlockRedactedThinking",
    "ContentBlockServerToolUse",
    "ContentBlockText",
    "ContentBlockThinking",
    "ContentBlockToolResult",
    "ContentBlockToolUse",
    "ContentBlockWebFetchToolResult",
    "ContentBlockWebSearchToolResult",
    "ContentChunk",
    "ContentType",
    "FunctionTagToolParser",
    "HeuristicToolParser",
    "Message",
    "MessagesRequest",
    "MessagesResponse",
    "OpenAIConversionError",
    "ReasoningReplayMode",
    "StreamBlockLedger",
    "SystemContent",
    "ThinkTagParser",
    "ThinkingConfig",
    "TokenCountRequest",
    "TokenCountResponse",
    "Tool",
    "ToolBlockState",
    "Usage",
    "aggregate_anthropic_sse_to_message",
    "anthropic_error_payload",
    "anthropic_error_type_for_failure",
    "anthropic_failure_payload",
    "anthropic_request_snapshot",
    "anthropic_status_for_error_type",
    "build_base_request_body",
    "dump_messages_request",
    "extract_text_from_content",
    "format_sse_event",
    "get_block_attr",
    "get_block_type",
    "get_token_count",
    "map_stop_reason",
    "serialize_tool_result_content",
    "set_if_not_none",
]
