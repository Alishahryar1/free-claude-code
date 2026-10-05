"""Tool delivery is gated at HTTP send, independently of provider execution."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.types import Message

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    anthropic_sse_streaming_response,
    openai_responses_sse_streaming_response,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from tests.api.test_response_streams import _json_error, _serve


def _start(wire: str) -> tuple[str, dict[str, Any]]:
    if wire == "messages":
        return "message_start", {
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [],
            }
        }
    return "response.created", {
        "response": {
            "id": "resp_test",
            "object": "response",
            "status": "in_progress",
            "output": [],
        }
    }


def _item(index: int, arguments: str = '{"path":"x"}') -> dict[str, Any]:
    return {
        "id": f"fc_{index}",
        "type": "function_call",
        "call_id": f"call_{index}",
        "name": "read",
        "arguments": arguments,
        "status": "completed",
    }


def _call(wire: str, index: int = 0, arguments: str = '{"path":"x"}'):
    if wire == "messages":
        return [
            (
                "content_block_start",
                {
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"call_{index}",
                        "name": "read",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments},
                },
            ),
            ("content_block_stop", {"index": index}),
        ]
    return [
        (
            "response.output_item.added",
            {
                "output_index": index,
                "item": {**_item(index, ""), "status": "in_progress"},
            },
        ),
        (
            "response.function_call_arguments.delta",
            {"item_id": f"fc_{index}", "output_index": index, "delta": arguments},
        ),
        (
            "response.function_call_arguments.done",
            {"item_id": f"fc_{index}", "output_index": index, "arguments": arguments},
        ),
        (
            "response.output_item.done",
            {"output_index": index, "item": _item(index, arguments)},
        ),
    ]


def _end(wire: str, *, failed: bool = False, output: list | None = None):
    if wire == "messages":
        if failed:
            return [("error", {"error": {"type": "api_error", "message": "cut"}})]
        return [
            (
                "message_delta",
                {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 10}},
            ),
            ("message_stop", {}),
        ]
    kind = "response.failed" if failed else "response.completed"
    return [
        (
            kind,
            {
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "status": "failed" if failed else "completed",
                    "output": output or [],
                    "error": {"message": "cut", "code": "upstream"} if failed else None,
                    "extension": {"keep": 7},
                }
            },
        )
    ]


def _frames(wire: str, events) -> list[str]:
    return [
        f"event: {kind}\ndata: "
        + json.dumps(
            {
                "type": kind,
                **data,
                **({"sequence_number": index} if wire == "responses" else {}),
            },
            ensure_ascii=False,
        )
        + "\n\n"
        for index, (kind, data) in enumerate(events)
    ]


async def _response(wire: str, body: AsyncIterator[str]) -> ManagedStreamingResponse:
    if wire == "messages":
        response = await anthropic_sse_streaming_response(
            body, pre_start_error_response=_json_error, request_id="tool-buffer-test"
        )
    else:
        response = await openai_responses_sse_streaming_response(
            body,
            headers={},
            pre_start_error_response=_json_error,
            request_id="tool-buffer-test",
        )
    assert isinstance(response, ManagedStreamingResponse)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("arguments", ['{"path":"x"}', '{"path":', "null"])
async def test_call_is_hidden_until_completion_then_sent_before_response_end(
    wire, arguments
):
    call = _call(wire, arguments=arguments)
    frames = _frames(
        wire, [_start(wire), *call, *_end(wire, output=[_item(0, arguments)])]
    )
    waiting_for_completion = asyncio.Event()
    complete = asyncio.Event()
    waiting_for_tail = asyncio.Event()
    finish = asyncio.Event()
    sent: list[str] = []

    async def body():
        for frame in frames[: len(call)]:
            yield frame
        waiting_for_completion.set()
        await complete.wait()
        yield frames[len(call)]
        waiting_for_tail.set()
        await finish.wait()
        for frame in frames[len(call) + 1 :]:
            yield frame

    async def send(message: Message):
        if message["type"] == "http.response.body":
            sent.append(message.get("body", b"").decode())

    response = await _response(wire, body())
    task = asyncio.create_task(_serve(response, send=send))
    try:
        await asyncio.wait_for(waiting_for_completion.wait(), 1)
        assert "".join(sent) == (frames[0] if wire == "responses" else "")
        assert "call_0" not in "".join(sent)
        complete.set()
        await asyncio.wait_for(waiting_for_tail.wait(), 1)
        assert "".join(sent) == "".join(frames[: len(call) + 1])
        assert not task.done()
    finally:
        complete.set()
        finish.set()
        await asyncio.wait_for(task, 1)
    assert "".join(sent) == "".join(frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("failed", [False, True])
async def test_overlapping_group_keeps_order_and_is_discarded_on_failure(wire, failed):
    first, second = _call(wire), _call(wire, 1)
    text = "text while calls overlap"
    text_events = (
        [
            (
                "content_block_start",
                {"index": 2, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"index": 2, "delta": {"type": "text_delta", "text": text}},
            ),
            ("content_block_stop", {"index": 2}),
        ]
        if wire == "messages"
        else [
            (
                "response.output_item.added",
                {
                    "output_index": 2,
                    "item": {
                        "id": "msg_text",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                },
            ),
            (
                "response.output_text.delta",
                {
                    "item_id": "msg_text",
                    "output_index": 2,
                    "content_index": 0,
                    "delta": text,
                },
            ),
        ]
    )
    group = [*first[:-1], *second[:-1], first[-1], *text_events]
    frames = _frames(
        wire,
        [
            _start(wire),
            *group,
            *(
                _end(
                    wire,
                    failed=True,
                    output=[_item(0), {**_item(1), "status": "incomplete"}],
                )
                if failed
                else [second[-1], *_end(wire, output=[_item(0), _item(1)])]
            ),
        ],
    )
    waiting = asyncio.Event()
    finish = asyncio.Event()
    sent: list[str] = []

    async def body():
        for frame in frames[: len(group) + 1]:
            yield frame
        waiting.set()
        await finish.wait()
        for frame in frames[len(group) + 1 :]:
            yield frame

    async def send(message: Message):
        if message["type"] == "http.response.body":
            sent.append(message.get("body", b"").decode())

    task = asyncio.create_task(_serve(await _response(wire, body()), send=send))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        assert "".join(sent) == (frames[0] if wire == "responses" else "")
    finally:
        finish.set()
        await asyncio.wait_for(task, 1)
    if failed:
        assert text not in "".join(sent)
        assert "call_0" not in "".join(sent)
        assert "call_1" not in "".join(sent)
        events = parse_sse_text("".join(sent))
        assert events[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )
        if wire == "responses":
            assert events[-1].data["response"]["output"] == []
            assert events[-1].data["response"]["extension"] == {"keep": 7}
    else:
        assert "".join(sent) == "".join(frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_completed_call_survives_later_incomplete_call_and_generated_error(wire):
    frames = _frames(wire, [_start(wire), *_call(wire), *_call(wire, 1)[:-1]])

    async def body():
        for frame in frames:
            yield frame
        raise RuntimeError("connection cut")

    response = await _response(wire, body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    events = parse_sse_text(result)
    assert "call_0" in result
    assert "call_1" not in result
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("response_status", [None, "failed", "completed"])
async def test_failed_snapshot_retains_only_previously_released_client_calls(
    response_status,
):
    terminal = _end("responses", failed=True, output=[_item(0), _item(1)])[0]
    terminal[1]["response"]["status"] = response_status
    frames = _frames("responses", [_start("responses"), *_call("responses"), terminal])

    async def body():
        for frame in frames:
            yield frame

    response = await _response("responses", body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    final = parse_sse_text(result)[-1]
    assert final.event == "response.failed"
    assert final.data["response"]["output"] == [_item(0)]
    assert "call_1" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_unfinished_call_is_discarded_at_eof(wire):
    frames = _frames(wire, [_start(wire), *_call(wire)[:-1]])

    async def body():
        for frame in frames:
            yield frame

    response = await _response(wire, body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    assert result == (frames[0] if wire == "responses" else "")


@pytest.mark.asyncio
async def test_incomplete_done_item_does_not_reach_harness():
    item = {**_item(0), "status": "incomplete"}
    frames = _frames(
        "responses",
        [
            _start("responses"),
            ("response.output_item.done", {"output_index": 0, "item": item}),
            *_end("responses", output=[item]),
        ],
    )

    async def body():
        for frame in frames:
            yield frame

    response = await _response("responses", body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    assert "call_0" not in result
    assert parse_sse_text(result)[-1].data["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_progress_snapshot_cannot_publish_incomplete_call():
    partial = {**_item(0, '{"path":'), "status": "in_progress"}
    frames = _frames(
        "responses",
        [
            _start("responses"),
            (
                "response.in_progress",
                {
                    "response": {
                        "id": "resp_test",
                        "status": "in_progress",
                        "output": [partial],
                    }
                },
            ),
            *_end("responses", output=[_item(0)]),
        ],
    )

    async def body():
        for frame in frames:
            yield frame

    response = await _response("responses", body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    events = parse_sse_text(result)
    assert events[1].data["response"]["output"] == []
    assert events[-1].data["response"]["output"] == [_item(0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "extra", "buffered"),
    [
        ("custom_tool_call", {"input": "opaque\ntext"}, True),
        (
            "tool_search_call",
            {"execution": "client", "arguments": {"query": "x"}},
            True,
        ),
        ("computer_call", {"action": {"type": "click", "x": 1, "y": 2}}, True),
        (
            "local_shell_call",
            {"action": {"type": "exec", "command": ["pwd"], "env": {}}},
            True,
        ),
        ("apply_patch_call", {"operation": {"type": "delete_file", "path": "x"}}, True),
        (
            "shell_call",
            {"environment": {"type": "local"}, "action": {"commands": ["pwd"]}},
            True,
        ),
        (
            "shell_call",
            {"environment": {"type": "container_reference", "container_id": "c"}},
            False,
        ),
        ("tool_search_call", {"execution": "server"}, False),
        ("web_search_call", {"action": {"type": "search", "query": "x"}}, False),
    ],
)
async def test_native_client_call_types_wait_while_hosted_calls_pass(
    kind, extra, buffered
):
    item = {"id": "item_call", "call_id": "call_native", "type": kind, **extra}
    frames = _frames(
        "responses",
        [
            _start("responses"),
            (
                "response.output_item.added",
                {"output_index": 0, "item": {**item, "status": "in_progress"}},
            ),
            (
                "response.output_item.done",
                {"output_index": 0, "item": {**item, "status": "completed"}},
            ),
            *_end("responses", output=[{**item, "status": "completed"}]),
        ],
    )
    waiting = asyncio.Event()
    complete = asyncio.Event()

    async def body():
        yield frames[0]
        yield frames[1]
        waiting.set()
        await complete.wait()
        for frame in frames[2:]:
            yield frame

    response = await _response("responses", body())
    stream = aiter(response.body_iterator)
    result = str(await anext(stream))
    read = asyncio.ensure_future(anext(stream))
    try:
        if buffered:
            await asyncio.wait_for(waiting.wait(), 1)
            assert not read.done()
        else:
            assert str(await asyncio.wait_for(read, 1)) == frames[1]
            result += frames[1]
    finally:
        complete.set()
    if buffered:
        result += str(await asyncio.wait_for(read, 1))
    result += "".join([str(chunk) async for chunk in stream])
    await response.aclose()
    assert result == "".join(frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_buffer_can_close_before_first_read_without_losing_body_ownership(wire):
    class Body:
        closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            return _frames(wire, [_start(wire)])[0]

        async def aclose(self):
            self.closed += 1

    body = Body()
    response = await _response(wire, body)
    await response.aclose()
    await response.aclose()
    assert body.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_cancellation_discards_pending_group_and_closes_body_once(wire):
    frames = _frames(wire, [_start(wire), *_call(wire)[:-1]])
    waiting = asyncio.Event()
    closed = 0
    released = 0
    sent = []

    async def body():
        nonlocal closed
        try:
            for frame in frames:
                yield frame
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed += 1

    async def release():
        nonlocal released
        released += 1

    async def send(message: Message):
        if message["type"] == "http.response.body":
            sent.append(message.get("body", b"").decode())

    response = await _response(wire, body())
    response.bind_release(release)
    task = asyncio.create_task(_serve(response, send=send))
    await asyncio.wait_for(waiting.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert sent == ([frames[0]] if wire == "responses" else [])
    assert closed == released == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_only", [False, True])
async def test_sparse_native_call_preserves_text_and_uses_final_snapshot_completion(
    snapshot_only,
):
    call = _call("responses")
    frames = _frames(
        "responses",
        [
            _start("responses"),
            *call[1:-1],
            *([] if snapshot_only else [call[-1]]),
            *_end("responses", output=[_item(0)]),
        ],
    )

    async def body():
        raw = "".join(frames)
        for offset in range(0, len(raw), 7):
            yield raw[offset : offset + 7]

    response = await _response("responses", body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    assert result == "".join(frames)


@pytest.mark.asyncio
async def test_incomplete_terminal_without_item_completion_discards_pending_call():
    call = _call("responses")
    item = _item(0)
    del item["status"]
    terminal = _end("responses", output=[item])[0]
    terminal = (
        "response.incomplete",
        {
            **terminal[1],
            "response": {
                **terminal[1]["response"],
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
    )
    frames = _frames("responses", [_start("responses"), *call[:-1], terminal])

    async def body():
        for frame in frames:
            yield frame

    response = await _response("responses", body())
    result = "".join([str(chunk) async for chunk in response.body_iterator])
    await response.aclose()
    assert "call_0" not in result
    final = parse_sse_text(result)[-1]
    assert final.event == "response.incomplete"
    assert final.data["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
