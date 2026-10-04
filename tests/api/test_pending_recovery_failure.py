"""A pending recovery retains public output even before its first new event."""

import asyncio
import json
from itertools import pairwise

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import public_text, text_events
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_stream_delivery import drained
from tests.api.test_tool_call_buffer import _response
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("metadata_first", [False, True])
async def test_continuation_timeout_retains_delivered_output(
    monkeypatch, protocol, wire, metadata_first
):
    module = httpx if protocol == "messages" else httpx2
    native_transport = module.MockTransport
    closed = []

    class WaitingStream(httpx.AsyncByteStream, httpx2.AsyncByteStream):
        async def __aiter__(self):
            if metadata_first:
                start = text_events(protocol, "")[0]
                yield ("data: " + json.dumps(start) + "\n\n").encode()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.append(True)

    def transport(handler):
        calls = 0

        async def handle(request):
            nonlocal calls
            response = handler(request)
            calls += 1
            if calls > 1:
                await response.aclose()
                return module.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=WaitingStream(),
                )
            return response

        return native_transport(handle)

    def executor(*args, **kwargs):
        kwargs["progress_timeout_seconds"] = 0.2
        return ProviderExecutor(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, "MockTransport", transport)
        patch.setattr("tests.api.test_hidden_stream_retries.ProviderExecutor", executor)
        patch.setattr(
            "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
            lambda: RecoveryHoldbackBuffer(max_bytes=1),
        )
        async with _harness(
            protocol,
            lambda _: (200, text_events(protocol, "Hello ", complete=False)),
            max_attempts=3,
        ) as (send, bodies, _):
            raw = await delivered(
                send(wire, [{"role": "user", "content": "greet"}]), wire
            )

    events = parse_sse_text(raw)
    assert len(bodies) == 2 and closed == [True]
    assert public_text(events, wire) == "Hello "
    failure_type = "error" if wire == "messages" else "response.failed"
    assert events[-1].event == failure_type
    assert sum(event.event == failure_type for event in events) == 1
    assert not any(
        event.event in {"message_stop", "response.completed"} for event in events
    )
    failure = events[-1].data if wire == "messages" else events[-1].data["response"]
    assert failure["error"]["type"] == "timeout_error"
    if wire == "messages":
        return

    assert failure["id"] == events[0].data["response"]["id"]
    assert failure["status"] == "failed"
    assert failure["usage"]["output_tokens"] > 0
    assert all(
        left.data["sequence_number"] < right.data["sequence_number"]
        for left, right in pairwise(events)
    )
    assert (
        "".join(
            part.get("text", "")
            for item in failure["output"]
            for part in item.get("content") or []
        )
        == "Hello "
    )

    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=raw.encode()
            )
        )
    )
    async with (
        AsyncOpenAI(api_key="fixture", max_retries=0, http_client=http) as client,
        client.responses.stream(model="fixture", input="greet") as stream,
    ):
        seen = [event async for event in stream]
        assert seen[-1].type == "response.failed"
        output = seen[-1].response.output[0]
        assert output.type == "message"
        part = output.content[0]
        assert part.type == "output_text" and part.text == "Hello "
        with pytest.raises(RuntimeError, match=r"response\.completed"):
            await stream.get_final_response()


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", [False, True])
async def test_pending_recovery_failure_does_not_require_successful_closures(handoff):
    prefix = (
        tool_events("responses", '{"path":"x"}')[:-1]
        if handoff
        else text_events("responses", "Hello ", complete=False)
    )

    async def source():
        for payload in prefix:
            yield f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"
        state = current_stream_delivery()
        assert state is not None
        state.begin_continuation(handoff=handoff)
        raise ValueError("recovery boundary failed")

    events = parse_sse_text(await drained(source(), "responses"))
    failure = events[-1].data["response"]
    assert events[-1].event == "response.failed"
    assert "recovery boundary failed" in failure["error"]["message"]
    assert len(events) == len(prefix) + 1
    assert len(failure["output"]) == 1
    if handoff:
        assert failure["output"][0] == prefix[-1]["item"]
    else:
        assert failure["output"][0]["status"] == "in_progress"
        assert failure["output"][0]["content"][0]["text"] == "Hello "


@pytest.mark.asyncio
async def test_cancellation_during_pending_continuation_emits_no_failure():
    closed = []
    events = []

    async def source():
        try:
            for payload in text_events("responses", "Hello ", complete=False):
                yield f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"
            state = current_stream_delivery()
            assert state is not None
            state.begin_continuation()
            raise asyncio.CancelledError
        finally:
            closed.append(True)

    response = await _response("responses", source())
    try:
        with pytest.raises(asyncio.CancelledError):
            async for chunk in response.body_iterator:
                events.extend(parse_sse_text(str(chunk)))
    finally:
        await response.aclose()
    assert public_text(events, "responses") == "Hello "
    assert not any(event.event == "response.failed" for event in events)
    assert closed == [True]
