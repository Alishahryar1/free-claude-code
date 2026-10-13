"""Local tool wire contracts independent of application and network execution."""

import pytest

from free_claude_code.core.anthropic.server_tool_sse import (
    ServerToolResponseContext,
    server_tool_completion_frames,
    server_tool_start_frames,
    web_fetch_result_block,
    web_search_result_block,
    web_tool_error_block,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.web_tools import WebFetchResult, WebSearchResult


@pytest.mark.parametrize(
    ("usage", "expected_start", "expected_end"),
    [
        (
            None,
            {"input_tokens": 7, "output_tokens": 1},
            {"input_tokens": 7, "output_tokens": 2},
        ),
        (
            {},
            {"input_tokens": 7, "output_tokens": 1},
            {"input_tokens": 7, "output_tokens": 2},
        ),
        (
            {
                "input_tokens": 17,
                "output_tokens": 9,
                "cache_read_input_tokens": 5,
                "bool": True,
                "text": "ignore",
            },
            {"input_tokens": 17, "output_tokens": 1, "cache_read_input_tokens": 5},
            {"input_tokens": 17, "output_tokens": 9, "cache_read_input_tokens": 5},
        ),
    ],
)
@pytest.mark.parametrize("tool_name", ["web_search", "web_fetch"])
@pytest.mark.parametrize("failed", [False, True])
def test_local_tool_event_order_linked_results_and_usage(
    usage, expected_start, expected_end, tool_name, failed
):
    context = ServerToolResponseContext(
        message_id="msg_local",
        tool_id="srvtoolu_local",
        model="public/model",
        tool_name=tool_name,
        tool_input={"query": "q"}
        if tool_name == "web_search"
        else {"url": "https://example.com/"},
        input_tokens=7,
        provider_usage=usage,
    )
    if failed:
        block = web_tool_error_block(context)
    elif tool_name == "web_search":
        block = web_search_result_block(
            context, [WebSearchResult(title="Result", url="https://example.com/")]
        )
    else:
        block = web_fetch_result_block(
            context,
            WebFetchResult(
                url="https://example.com/final",
                title="Page",
                media_type="text/plain",
                data="contents",
            ),
            retrieved_at="2026-09-15T00:00:00+00:00",
        )
    events = parse_sse_text(
        "".join(
            [
                *server_tool_start_frames(context),
                *server_tool_completion_frames(context, block, summary="12345678"),
            ]
        )
    )
    assert [event.event for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0].data["message"]["model"] == "public/model"
    assert events[0].data["message"]["usage"] == expected_start
    assert events[1].data["content_block"]["id"] == "srvtoolu_local"
    assert events[3].data["content_block"]["tool_use_id"] == "srvtoolu_local"
    assert [events[index].data["index"] for index in (1, 3, 5)] == [0, 1, 2]
    assert events[-2].data["usage"] == {
        **expected_end,
        "server_tool_use": {f"{tool_name}_requests": 1},
    }
    assert events[-2].data["delta"]["stop_reason"] == "end_turn"
    if failed:
        assert block["content"] == {
            "type": "web_search_tool_result_error"
            if tool_name == "web_search"
            else "web_fetch_tool_error",
            "error_code": "unavailable",
        }
    elif tool_name == "web_search":
        assert block["content"] == [
            {
                "type": "web_search_result",
                "title": "Result",
                "url": "https://example.com/",
            }
        ]
    else:
        assert block["content"] == {
            "type": "web_fetch_result",
            "url": "https://example.com/final",
            "content": {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "contents",
                },
                "title": "Page",
                "citations": {"enabled": True},
            },
            "retrieved_at": "2026-09-15T00:00:00+00:00",
        }
