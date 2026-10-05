"""Initial output is released while the real upstream read remains blocked."""

import asyncio
import json

import httpx
import httpx2
import pytest

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    anthropic_sse_streaming_response,
    bind_response_lifetime,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import (
    assert_winning_tool,
    executed,
    partial_tool,
)
from tests.api.test_response_streams import _json_error, _serve
from tests.api.test_tool_call_buffer import _response
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_history_transports import _events_for, _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture
def short_holdback(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(holdback_seconds=0.02),
    )


def encode(protocol, events):
    return "".join(
        (f"event: {event['type']}\n" if protocol == "messages" else "")
        + f"data: {json.dumps(event)}\n\n"
        for event in events
    ).encode()


class GatedBody(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.close_count = 0

    async def __aiter__(self):
        yield self.prefix
        self.waiting.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield self.suffix

    async def aclose(self):
        self.close_count += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_initial_text_reaches_http_without_another_upstream_event(
    protocol, wire, short_holdback
):
    events = text_events(protocol, "buffered text")
    prefix_size = len(text_events(protocol, "buffered text", complete=False))
    body = GatedBody(
        encode(protocol, events[:prefix_size]),
        encode(protocol, events[prefix_size:]),
    )
    visible = asyncio.Event()
    sent = []

    async def capture(message):
        data = message.get("body", b"")
        sent.append(data)
        if b"buffered text" in data:
            visible.set()

    async with _harness(protocol, lambda _: (200, body)) as (send, bodies, _):

        async def serve():
            source = send(wire, [{"role": "user", "content": "hello"}])
            response = await _response(wire, executed(source, wire))
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        wait_visible = asyncio.create_task(visible.wait())
        try:
            await asyncio.wait_for(body.waiting.wait(), 2)
            done, _ = await asyncio.wait({wait_visible, serving}, timeout=1)
            assert wait_visible in done, "Buffered text is still waiting for upstream"
            assert not body.cancelled
            assert not body.release.is_set()
            assert not serving.done()
            assert len(bodies) == 1
        finally:
            body.release.set()
            wait_visible.cancel()
            await asyncio.gather(wait_visible, return_exceptions=True)
            await asyncio.wait_for(serving, 2)

    output = parse_sse_text(b"".join(sent).decode())
    assert public_text(output, wire) == "buffered text"
    assert_completed(output, wire)
    assert body.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("ending", ["eof", "error"])
async def test_ready_failure_at_deadline_retries_without_releasing_old_text(
    protocol, wire, ending, monkeypatch
):
    now = [0.0]
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(now=lambda: now[0]),
    )
    error = {"type": "api_error", "message": "internal server error"}
    if protocol == "responses":
        failure = {
            "type": "response.failed",
            "sequence_number": 4,
            "response": {
                **text_events(protocol, "abandoned")[-1]["response"],
                "status": "failed",
                "error": {"code": "server_error", "message": "interrupted"},
            },
        }
    else:
        failure = {"type": "error", "error": error}
    first = GatedBody(
        encode(protocol, text_events(protocol, "abandoned", complete=False)),
        encode(protocol, [failure]) if ending == "error" else b"",
    )

    def reply(bodies):
        if len(bodies) == 1:
            return 200, first
        assert first.close_count == 1
        return 200, text_events(protocol, "winning")

    sent = []

    async def capture(message):
        sent.append(message.get("body", b""))

    async with _harness(protocol, reply) as (send, bodies, _):

        async def serve():
            source = send(wire, [{"role": "user", "content": "hello"}])
            response = await _response(wire, executed(source, wire))
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(first.waiting.wait(), 2)
            # The pending read completes after its logical deadline. Its failure
            # must be validated before the transport adds a deadline flush.
            now[0] = 1.0
            first.release.set()
            await asyncio.wait_for(serving, 2)
        finally:
            first.release.set()
            if not serving.done():
                serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]

    output = parse_sse_text(b"".join(sent).decode())
    assert public_text(output, wire) == "winning"
    assert_completed(output, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_failure_after_timed_release_continues_the_visible_answer(
    protocol, wire, short_holdback
):
    first = GatedBody(
        encode(protocol, text_events(protocol, "Hello ", complete=False)), b""
    )
    visible = asyncio.Event()
    sent = []

    async def capture(message):
        data = message.get("body", b"")
        sent.append(data)
        if b"Hello " in data:
            visible.set()

    async with _harness(
        protocol,
        lambda bodies: (
            200,
            first if len(bodies) == 1 else text_events(protocol, "world."),
        ),
    ) as (send, bodies, _):

        async def serve():
            source = send(wire, [{"role": "user", "content": "hello"}])
            response = await _response(wire, executed(source, wire))
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(visible.wait(), 2)
            assert not first.release.is_set()
        finally:
            first.release.set()
            await asyncio.wait_for(serving, 2)
        assert len(bodies) == 2
        assert "Hello " in json.dumps(bodies[1])

    output = parse_sse_text(b"".join(sent).decode())
    assert public_text(output, wire) == "Hello world."
    assert_completed(output, wire)
    assert first.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("released", [False, True])
async def test_cancellation_drains_the_read_before_closing_its_http_body(
    protocol, wire, released, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(holdback_seconds=0.01 if released else 60),
    )
    body = GatedBody(
        encode(protocol, text_events(protocol, "visible", complete=False)), b""
    )
    visible = asyncio.Event()

    async def capture(message):
        if b"visible" in message.get("body", b""):
            visible.set()

    async with _harness(protocol, lambda _: (200, body)) as (send, bodies, _):

        async def serve():
            source = send(wire, [{"role": "user", "content": "hello"}])
            response = await _response(wire, executed(source, wire))
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(body.waiting.wait(), 2)
            if released:
                await asyncio.wait_for(visible.wait(), 2)
            else:
                assert not visible.is_set()
            serving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(serving, 2)
            assert body.cancelled
            assert body.close_count == 1
            assert len(bodies) == 1
        finally:
            body.release.set()
            if not serving.done():
                serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_closing_prefetched_response_drains_timed_read(
    protocol, wire, short_holdback
):
    body = GatedBody(
        encode(protocol, text_events(protocol, "held", complete=False)), b""
    )
    released = []

    async def release():
        released.append("released")

    async with _harness(protocol, lambda _: (200, body)) as (send, bodies, _):
        source = send(wire, [{"role": "user", "content": "hello"}])
        response = await asyncio.wait_for(_response(wire, executed(source, wire)), 2)
        await bind_response_lifetime(response, release)
        try:
            assert body.waiting.is_set()
            assert not body.release.is_set()
            await response.aclose()
            await response.aclose()
            assert body.cancelled
            assert body.close_count == 1
            assert released == ["released"]
            assert len(bodies) == 1
        finally:
            body.release.set()
            await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_timed_release_keeps_incomplete_tools_hidden_and_retryable(
    protocol, wire, short_holdback
):
    first = GatedBody(encode(protocol, partial_tool(protocol)), b"")
    started = asyncio.Event()
    sent = []

    async def capture(message):
        if message["type"] == "http.response.start":
            started.set()
        sent.append(message.get("body", b""))

    async with _harness(
        protocol,
        lambda bodies: (
            200,
            first if len(bodies) == 1 else tool_events(protocol, '{"path":"winning"}'),
        ),
    ) as (send, bodies, _):

        async def serve():
            source = send(
                wire, [{"role": "user", "content": "read"}], tools=_tools(wire)
            )
            response = await _response(wire, executed(source, wire))
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert not first.release.is_set()
            assert b"call_abandoned" not in b"".join(sent)
        finally:
            first.release.set()
            await asyncio.wait_for(serving, 2)
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
    assert_winning_tool(b"".join(sent).decode(), wire)
    assert first.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
async def test_timed_release_keeps_projected_reasoning_hidden_and_retryable(
    protocol, short_holdback
):
    prefix_size = {"chat": 1, "messages": 3, "responses": 4}[protocol]
    first = GatedBody(encode(protocol, _events_for(protocol)[:prefix_size]), b"")
    started = asyncio.Event()
    sent = []

    async def capture(message):
        if message["type"] == "http.response.start":
            started.set()
        sent.append(message.get("body", b""))

    async with _harness(
        protocol,
        lambda bodies: (
            200,
            first if len(bodies) == 1 else text_events(protocol, "winning"),
        ),
    ) as (send, bodies, _):

        async def serve():
            source = send("messages", [{"role": "user", "content": "hello"}])
            response = await anthropic_sse_streaming_response(
                executed(source, "messages"),
                pre_start_error_response=_json_error,
                request_id="hidden-reasoning",
                hide_reasoning=True,
            )
            assert isinstance(response, ManagedStreamingResponse)
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert not first.release.is_set()
            assert b"Find 17." not in b"".join(sent)
        finally:
            first.release.set()
            await asyncio.wait_for(serving, 2)
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
    raw = b"".join(sent).decode()
    assert "Find 17." not in raw
    output = parse_sse_text(raw)
    assert public_text(output, "messages") == "winning"
    assert_completed(output, "messages")
