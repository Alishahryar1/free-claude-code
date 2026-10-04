"""SSE streaming and shared results for local web server tools."""

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.server_tool_sse import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_ERROR,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.json_types import JsonObject, JsonValue

from . import outbound
from .constants import _MAX_FETCH_CHARS
from .egress import WebFetchEgressPolicy, web_url_allowed_by_domains
from .parsers import extract_query, extract_url
from .request import (
    LocalWebTool,
    forced_server_tool_name,
    forced_tool_turn_text,
    local_web_tools,
)


@dataclass(frozen=True, slots=True)
class WebToolResult:
    content: JsonValue
    summary: str
    result_block_type: str
    is_error: bool = False


def _search_summary(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}\n{result['url']}")
    return "\n\n".join(lines)


async def execute_web_tool(
    tool: LocalWebTool,
    tool_input: dict[str, str],
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
) -> WebToolResult:
    """Run validated local input and return the same evidence to both execution paths."""
    tool_name = tool.name
    result_type = (
        WEB_SEARCH_TOOL_RESULT if tool_name == "web_search" else WEB_FETCH_TOOL_RESULT
    )
    try:
        if tool_name == "web_search":
            query = tool_input["query"]
            results = await outbound._run_web_search(query)
            results = [
                result
                for result in results
                if web_url_allowed_by_domains(
                    result["url"], tool.allowed_domains, tool.blocked_domains
                )
            ]
            content: JsonValue = [
                {
                    "type": "web_search_result",
                    "title": result["title"],
                    "url": result["url"],
                }
                for result in results
            ]
            return WebToolResult(content, _search_summary(query, results), result_type)
        scoped = replace(
            web_fetch_egress,
            allowed_domains=tool.allowed_domains,
            blocked_domains=tool.blocked_domains,
        )
        fetched = await outbound._run_web_fetch(tool_input["url"], scoped)
        content = {
            "type": "web_fetch_result",
            "url": fetched["url"],
            "content": {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": fetched["media_type"],
                    "data": fetched["data"],
                },
                "title": fetched["title"],
                "citations": {"enabled": True},
            },
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        return WebToolResult(content, fetched["data"][:_MAX_FETCH_CHARS], result_type)
    except Exception as error:
        outbound._log_web_tool_failure(
            tool_name, error, fetch_url=tool_input.get("url")
        )
        error_type = (
            WEB_SEARCH_TOOL_RESULT_ERROR
            if tool_name == "web_search"
            else WEB_FETCH_TOOL_ERROR
        )
        return WebToolResult(
            {"type": error_type, "error_code": "unavailable"},
            outbound._web_tool_client_error_summary(
                tool_name, error, verbose=verbose_client_errors
            ),
            result_type,
            is_error=True,
        )


async def stream_web_message(
    blocks: list[JsonObject],
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    server_tool_use: dict[str, int],
    stop_reason: str = "end_turn",
    stop_sequence: str | None = None,
) -> AsyncIterator[str]:
    """Serialize one public turn, never concatenate internal provider messages."""
    yield format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4()}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    for index, block in enumerate(blocks):
        text = block.get("text") if block.get("type") == "text" else None
        start: JsonObject = (
            {"type": "text", "text": ""} if isinstance(text, str) else dict(block)
        )
        if block.get("type") == SERVER_TOOL_USE:
            start["input"] = {}
        yield format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": start,
            },
        )
        if block.get("type") == SERVER_TOOL_USE:
            yield format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            )
        if isinstance(text, str) and text:
            yield format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        yield format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": index}
        )
    yield format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "server_tool_use": server_tool_use,
            },
        },
    )
    yield format_sse_event("message_stop", {"type": "message_stop"})


async def stream_web_server_tool_response(
    request: MessagesRequest,
    input_tokens: int,
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    response_model: str | None = None,
    verbose_client_errors: bool = False,
) -> AsyncIterator[str]:
    """Forced local fallback, not a hosted citation or encrypted-content pipeline."""
    tool_name = forced_server_tool_name(request)
    if tool_name is None:
        return
    tool = next(item for item in local_web_tools(request) if item.name == tool_name)
    text = forced_tool_turn_text(request)
    tool_input = (
        {"query": extract_query(text)}
        if tool_name == "web_search"
        else {"url": extract_url(text)}
    )
    result = await execute_web_tool(
        tool,
        tool_input,
        web_fetch_egress=web_fetch_egress,
        verbose_client_errors=verbose_client_errors,
    )
    tool_id = f"srvtoolu_{uuid.uuid4().hex}"
    blocks: list[JsonObject] = [
        {
            "type": SERVER_TOOL_USE,
            "id": tool_id,
            "name": tool_name,
            "input": tool_input,
        },
        {
            "type": result.result_block_type,
            "tool_use_id": tool_id,
            "content": result.content,
        },
        {"type": "text", "text": result.summary},
    ]
    async for event in stream_web_message(
        blocks,
        model=request.model if response_model is None else response_model,
        input_tokens=input_tokens,
        output_tokens=max(1, len(result.summary) // 4),
        server_tool_use={f"{tool_name}_requests": 1},
    ):
        yield event
