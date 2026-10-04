"""Direct regressions for folding fragmented Anthropic SSE content."""

from collections.abc import AsyncIterator

import pytest

from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.json_types import JsonObject


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 17, 4096])
async def test_thinking_signature_fragments_preserve_each_block(chunk_size):
    payloads: list[JsonObject] = [
        {
            "type": "message_start",
            "message": {
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "Initial ",
                "signature": "prefix-",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "thought"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "first-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "second"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "Independent"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "signature_delta", "signature": "other-"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "signature_delta", "signature": "signature"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
    raw = "".join(
        format_sse_event(str(payload["type"]), payload) for payload in payloads
    )

    async def chunks() -> AsyncIterator[str]:
        for start in range(0, len(raw), chunk_size):
            yield raw[start : start + chunk_size]

    message, error = await aggregate_anthropic_sse_to_message(chunks())
    assert error is None
    assert message["content"] == [
        {
            "type": "thinking",
            "thinking": "Initial thought",
            "signature": "prefix-first-second",
        },
        {"type": "thinking", "thinking": "Independent", "signature": "other-signature"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("block_type", ["tool_use", "server_tool_use"])
async def test_input_json_fragments_aggregate_for_both_tool_kinds(block_type):
    payloads: list[JsonObject] = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": block_type,
                "id": "toolu_test",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"official docs"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]

    async def chunks() -> AsyncIterator[str]:
        for payload in payloads:
            yield format_sse_event(str(payload["type"]), payload)

    message, error = await aggregate_anthropic_sse_to_message(chunks())
    assert error is None
    assert message["content"][0]["input"] == {"query": "official docs"}
