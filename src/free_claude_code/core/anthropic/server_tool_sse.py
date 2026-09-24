"""Anthropic representation of completed local web-tool operations."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.web_tools import WebFetchResult, WebSearchResult

from .server_tool_types import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_ERROR,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from .streaming import format_sse_event


@dataclass(frozen=True, slots=True)
class ServerToolResponseContext:
    message_id: str
    tool_id: str
    model: str
    tool_name: str
    tool_input: Mapping[str, str]
    input_tokens: int
    provider_usage: Mapping[str, object] | None = None


def server_tool_start_frames(context: ServerToolResponseContext) -> Iterator[str]:
    yield from server_tool_message_start_frames(context)
    yield from server_tool_use_frames(context, index=0)


def server_tool_message_start_frames(
    context: ServerToolResponseContext,
) -> Iterator[str]:
    yield format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": context.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": context.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _message_start_usage(
                    context.input_tokens, context.provider_usage
                ),
            },
        },
    )


def server_tool_use_frames(
    context: ServerToolResponseContext, *, index: int
) -> Iterator[str]:
    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": SERVER_TOOL_USE,
                "id": context.tool_id,
                "name": context.tool_name,
                "input": context.tool_input,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def web_search_result_block(
    context: ServerToolResponseContext, results: Sequence[WebSearchResult]
) -> JsonObject:
    return {
        "type": WEB_SEARCH_TOOL_RESULT,
        "tool_use_id": context.tool_id,
        "content": [
            {"type": "web_search_result", "title": result.title, "url": result.url}
            for result in results
        ],
    }


def web_fetch_result_block(
    context: ServerToolResponseContext,
    result: WebFetchResult,
    *,
    retrieved_at: str,
) -> JsonObject:
    return {
        "type": WEB_FETCH_TOOL_RESULT,
        "tool_use_id": context.tool_id,
        "content": {
            "type": "web_fetch_result",
            "url": result.url,
            "content": {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": result.media_type,
                    "data": result.data,
                },
                "title": result.title,
                "citations": {"enabled": True},
            },
            "retrieved_at": retrieved_at,
        },
    }


def web_tool_error_block(
    context: ServerToolResponseContext, *, error_code: str = "unavailable"
) -> JsonObject:
    search = context.tool_name == "web_search"
    return {
        "type": WEB_SEARCH_TOOL_RESULT if search else WEB_FETCH_TOOL_RESULT,
        "tool_use_id": context.tool_id,
        "content": {
            "type": WEB_SEARCH_TOOL_RESULT_ERROR if search else WEB_FETCH_TOOL_ERROR,
            "error_code": error_code,
        },
    }


def server_tool_completion_frames(
    context: ServerToolResponseContext,
    result_block: JsonObject,
    *,
    summary: str,
) -> Iterator[str]:
    yield from server_tool_result_frames(result_block, index=1)
    yield from server_tool_finish_frames(context, summary=summary, index=2)


def server_tool_result_frames(result_block: JsonObject, *, index: int) -> Iterator[str]:
    yield format_sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": result_block},
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def server_tool_finish_frames(
    context: ServerToolResponseContext,
    *,
    summary: str,
    index: int,
    request_count: int = 1,
) -> Iterator[str]:
    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": summary},
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )
    yield format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": _completion_usage(
                context.input_tokens,
                summary,
                context.provider_usage,
                usage_key=f"{context.tool_name}_requests",
                request_count=request_count,
            ),
        },
    )
    yield format_sse_event("message_stop", {"type": "message_stop"})


def _message_start_usage(
    input_tokens: int,
    provider_usage: Mapping[str, object] | None,
) -> dict[str, object]:
    if provider_usage is None:
        return {"input_tokens": input_tokens, "output_tokens": 1}
    usage = _integer_usage(provider_usage)
    usage.setdefault("input_tokens", input_tokens)
    usage["output_tokens"] = 1
    return usage


def _completion_usage(
    input_tokens: int,
    summary: str,
    provider_usage: Mapping[str, object] | None,
    *,
    usage_key: str,
    request_count: int,
) -> dict[str, object]:
    if provider_usage is None:
        usage: dict[str, object] = {
            "input_tokens": input_tokens,
            "output_tokens": max(1, len(summary) // 4),
        }
    else:
        usage = _integer_usage(provider_usage)
        usage.setdefault("input_tokens", input_tokens)
        usage.setdefault("output_tokens", max(1, len(summary) // 4))
    usage["server_tool_use"] = {usage_key: request_count}
    return usage


def _integer_usage(usage: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
