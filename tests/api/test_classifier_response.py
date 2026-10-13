import asyncio
import json

import pytest

from free_claude_code.api.handlers.classifier_response import classifier_response
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event


def wire(payloads):
    return "".join(format_sse_event(payload["type"], payload) for payload in payloads)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 17, 10000])
async def test_projection_preserves_text_extensions_and_sparse_interleaved_blocks(
    chunk_size,
):
    literal = "<thinking>Classifier explanation</thinking><severity>0</severity>"
    start = {
        "type": "message_start",
        "message": {
            "id": "original",
            "model": "public",
            "content": [],
            "usage": {"input_tokens": 3},
        },
        "extension": {"a": 1},
    }
    terminal = {
        "type": "message_delta",
        "delta": {"stop_reason": "stop_sequence", "stop_sequence": "custom"},
        "usage": {"output_tokens": 50, "extra": 7},
    }
    payloads = [
        start,
        {
            "type": "content_block_start",
            "index": 5,
            "content_block": {"type": "thinking", "thinking": "hidden"},
        },
        {
            "type": "content_block_start",
            "index": 9,
            "content_block": {
                "type": "text",
                "text": literal,
                "citations": [],
                "extension": {"index": 9},
            },
        },
        {
            "type": "content_block_delta",
            "index": 5,
            "delta": {"type": "signature_delta", "signature": "hidden"},
        },
        {
            "type": "content_block_delta",
            "index": 9,
            "delta": {
                "type": "text_delta",
                "text": "<reason>fine</reason><category>read</category>",
            },
        },
        {"type": "content_block_stop", "index": 5},
        {
            "type": "content_block_start",
            "index": 12,
            "content_block": {"type": "redacted_thinking", "data": "opaque"},
        },
        {"type": "content_block_stop", "index": 12},
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {
                "type": "tool_use",
                "id": "tool-id",
                "name": "read",
                "input": {},
            },
        },
        {"type": "content_block_stop", "index": 9},
        {"type": "content_block_stop", "index": 3},
        {"type": "ping", "extension": 1},
        terminal,
        {"type": "message_stop"},
    ]
    raw = wire(payloads)
    closed = False

    async def source():
        nonlocal closed
        try:
            for offset in range(0, len(raw), chunk_size):
                yield raw[offset : offset + chunk_size]
        finally:
            closed = True

    output = "".join([chunk async for chunk in classifier_response(source())])
    actual = [event.data for event in parse_sse_text(output)]
    assert actual == [
        start,
        {**payloads[2], "index": 0},
        {**payloads[4], "index": 0},
        {**payloads[8], "index": 1},
        {**payloads[9], "index": 0},
        {**payloads[10], "index": 1},
        payloads[11],
        terminal,
        payloads[13],
    ]
    assert closed


@pytest.mark.asyncio
async def test_reasoning_only_completion_stays_empty_with_original_termination():
    start = {"type": "message_start", "message": {"content": []}}
    end = {
        "type": "message_delta",
        "delta": {"stop_reason": "max_tokens"},
        "usage": {"output_tokens": 64},
    }

    async def source():
        yield wire(
            [
                start,
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "analysis"},
                },
                {"type": "content_block_stop", "index": 0},
                end,
                {"type": "message_stop"},
            ]
        )

    output = [
        event.data
        for event in parse_sse_text(
            "".join([chunk async for chunk in classifier_response(source())])
        )
    ]
    assert output == [start, end, {"type": "message_stop"}]


@pytest.mark.asyncio
async def test_cancellation_during_hidden_reasoning_closes_input():
    blocked = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            yield wire([{"type": "message_start", "message": {"content": []}}])
            yield wire(
                [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                ]
            )
            blocked.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    filtered = classifier_response(source())
    assert "message_start" in await anext(filtered)

    async def advance():
        return await anext(filtered)

    pending = asyncio.create_task(advance())
    await asyncio.wait_for(blocked.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_only_normal_exhaustion_flushes_unterminated_event(fails):
    failure = RuntimeError("upstream failed")
    raw = "event: error\ndata: " + json.dumps(
        {"type": "error", "error": {"message": "unchanged"}}
    )

    async def source():
        yield raw
        if fails:
            raise failure

    if fails:
        with pytest.raises(RuntimeError) as caught:
            _ = [chunk async for chunk in classifier_response(source())]
        assert caught.value is failure
    else:
        output = "".join([chunk async for chunk in classifier_response(source())])
        assert parse_sse_text(output)[0].data == {
            "type": "error",
            "error": {"message": "unchanged"},
        }
