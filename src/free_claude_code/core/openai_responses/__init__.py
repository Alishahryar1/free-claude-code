"""Public OpenAI Responses protocol boundary."""

from .chat_request import ResponsesChatRequest, build_responses_chat_request
from .errors import (
    ResponsesConversionError,
    openai_error_from_failure,
    openai_error_payload,
    openai_error_type_for_failure,
    openai_failure_payload,
)
from .events import OPENAI_RESPONSES_SSE_HEADERS, committed_response_failure_frame
from .ids import (
    new_call_id,
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
)
from .messages_request import ResponsesMessagesRequest, build_responses_messages_request
from .messages_stream import AnthropicToResponsesStream
from .models import OpenAIResponsesRequest
from .native import NativeResponsesRelay, build_native_responses_request
from .provider_input import build_responses_provider_request
from .provider_stream import (
    ResponsesProviderStream,
    ResponsesStreamFailure,
    responses_stream_failure_from_event,
)
from .reasoning import responses_reasoning_config, responses_reasoning_policy
from .streaming.blocks import ReasoningBlockState, TextBlockState, ToolBlockState
from .streaming.completion import (
    ResponseBlockCompleter,
    reasoning_output_item,
    tool_item,
)
from .streaming.error_mapping import replay_unsafe_function_call_error
from .streaming.event_builders import ResponseEventBuilder
from .streaming.ledger import ResponsesOutputLedger
from .tokens import estimate_responses_input_tokens
from .tool_adaptation import (
    ResponsesToolAdapter,
    ResponsesToolEventAdapter,
    ResponsesToolPolicy,
)
from .tools import (
    ResponsesToolIdentity,
    flatten_responses_tool_name,
)

__all__ = [
    "OPENAI_RESPONSES_SSE_HEADERS",
    "AnthropicToResponsesStream",
    "NativeResponsesRelay",
    "OpenAIResponsesRequest",
    "ReasoningBlockState",
    "ResponseBlockCompleter",
    "ResponseEventBuilder",
    "ResponsesChatRequest",
    "ResponsesConversionError",
    "ResponsesMessagesRequest",
    "ResponsesOutputLedger",
    "ResponsesProviderStream",
    "ResponsesStreamFailure",
    "ResponsesToolAdapter",
    "ResponsesToolEventAdapter",
    "ResponsesToolIdentity",
    "ResponsesToolPolicy",
    "TextBlockState",
    "ToolBlockState",
    "build_native_responses_request",
    "build_responses_chat_request",
    "build_responses_messages_request",
    "build_responses_provider_request",
    "committed_response_failure_frame",
    "estimate_responses_input_tokens",
    "flatten_responses_tool_name",
    "new_call_id",
    "new_message_item_id",
    "new_reasoning_item_id",
    "new_response_id",
    "openai_error_from_failure",
    "openai_error_payload",
    "openai_error_type_for_failure",
    "openai_failure_payload",
    "reasoning_output_item",
    "replay_unsafe_function_call_error",
    "responses_reasoning_config",
    "responses_reasoning_policy",
    "responses_stream_failure_from_event",
    "tool_item",
]
