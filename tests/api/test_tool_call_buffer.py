"""Tool delivery is gated at HTTP send, independently of provider execution."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    anthropic_sse_streaming_response,
    openai_responses_sse_streaming_response,
)
from free_claude_code.api.stream_delivery import PublicResponseStream
from free_claude_code.core.stream_output import StreamOutput, completed_output
from tests.api.test_response_streams import _json_error


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


async def _response(
    wire: str, body: AsyncIterator[StreamOutput]
) -> ManagedStreamingResponse:
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
async def test_complete_packet_is_released_before_response_terminal(wire):
    continue_stream = asyncio.Event()
    closed = []

    async def body():
        try:
            yield StreamOutput(tuple(_frames(wire, [_start(wire)])))
            yield completed_output(_frames(wire, _call(wire)))
            await continue_stream.wait()
            yield StreamOutput(tuple(_frames(wire, _end(wire))))
        finally:
            closed.append(True)

    response = await _response(wire, body())
    assert isinstance(response.body_iterator, PublicResponseStream)
    try:
        seen = ""
        while "call_0" not in seen:
            seen += await anext(response.body_iterator)
        assert response.body_iterator.state.prefix is not None
        assert response.body_iterator.state.prefix.has_calls
        assert (
            "content_block_stop" if wire == "messages" else "response.output_item.done"
        ) in seen
        continue_stream.set()
        async for _ in response.body_iterator:
            pass
    finally:
        await response.aclose()
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_late_packet_projection_failure_releases_no_call(wire, monkeypatch):
    original = PublicResponseStream._finalize_frame

    def fail_on_call_delta(self, frame):
        if (
            "input_json_delta" in frame
            or "response.function_call_arguments.delta" in frame
        ):
            raise ValueError("synthetic late projection failure")
        return original(self, frame)

    monkeypatch.setattr(PublicResponseStream, "_finalize_frame", fail_on_call_delta)

    async def body():
        yield StreamOutput(tuple(_frames(wire, [_start(wire)])))
        yield completed_output(_frames(wire, _call(wire)))

    response = await _response(wire, body())
    assert isinstance(response.body_iterator, PublicResponseStream)
    try:
        output = "".join([chunk async for chunk in response.body_iterator])
        assert "call_0" not in output
        assert response.body_iterator.state.prefix is not None
        assert not response.body_iterator.state.prefix.has_calls
        assert not response.body_iterator._delivered.items
    finally:
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_wire_stop_alone_is_not_source_completion_evidence(wire):
    async def body():
        yield StreamOutput(tuple(_frames(wire, [_start(wire)])))
        yield StreamOutput(tuple(_frames(wire, _call(wire))))

    response = await _response(wire, body())
    assert isinstance(response.body_iterator, PublicResponseStream)
    try:
        async for _ in response.body_iterator:
            pass
        assert response.body_iterator.state.prefix is not None
        assert not response.body_iterator.state.prefix.has_calls
    finally:
        await response.aclose()
