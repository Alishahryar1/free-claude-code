"""Stopped public generations release recovery ownership for later requests."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.responses import StreamingResponse

from free_claude_code.api.response_streams import (
    anthropic_sse_streaming_response,
    openai_responses_sse_streaming_response,
)
from free_claude_code.providers.anthropic_messages.passthrough import (
    stream_native_messages,
)
from free_claude_code.providers.failure_policy import RetryableProviderProtocolError
from tests.api.test_response_streams import _json_error, _serve
from tests.providers.test_anthropic_messages_transport import (
    Endpoint,
    _events,
    _sse,
    _stream,
    _transport,
)
from tests.providers.test_history_transports import _events_for, _harness
from tests.providers.test_provider_admission import _controller, _open, _status_error
from tests.providers.test_request_recovery_transitions import Wire


async def _response(source, wire):
    options = {"pre_start_error_response": _json_error, "request_id": "stopped"}
    return await (
        anthropic_sse_streaming_response(source, **options)
        if wire == "messages"
        else openai_responses_sse_streaming_response(source, headers={}, **options)
    )


@asynccontextmanager
async def _messages_path(route, respond, *, max_attempts=3):
    bodies = []
    wires = []
    admission = _controller(max_attempts=max_attempts, max_concurrency=2)

    def reply(request):
        bodies.append(json.loads(request.content))
        status, payload = respond(len(bodies))
        wire = (
            payload
            if isinstance(payload, Wire)
            else Wire(_sse(*payload) if status == 200 else json.dumps(payload).encode())
        )
        wires.append(wire)
        return httpx.Response(
            status,
            headers={"content-type": "text/event-stream"},
            stream=wire,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        transport = _transport(client, admission)

        def source():
            if route == "native":
                return stream_native_messages(
                    client,
                    admission,
                    base_url="https://native.invalid/v1",
                    headers={},
                    body={
                        "model": "native",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    public_model="public",
                    provider_name="TEST",
                    read_timeout_s=3,
                    request_id="stopped",
                )
            return _stream(transport, Endpoint(), responses=route == "responses")

        yield source, bodies, wires, admission


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "messages", "responses"])
@pytest.mark.parametrize("terminal", ["message_delta", "message_stop"])
@pytest.mark.parametrize(
    "max_attempts,probe", [(3, False), (1, False), (3, True), (2, True)]
)
async def test_stopped_first_messages_event_releases_admission(
    route, terminal, max_attempts, probe
):
    stopped = {"type": terminal}
    if terminal == "message_delta":
        stopped["delta"] = {"stop_reason": "end_turn"}

    def respond(ordinal):
        if probe and ordinal == 1:
            return 503, {"error": {"type": "overloaded_error", "message": "busy"}}
        return 200, [stopped] if ordinal == 1 + probe else _events()

    wire = "responses" if route == "responses" else "messages"
    async with _messages_path(route, respond, max_attempts=max_attempts) as (
        source,
        bodies,
        wires,
        admission,
    ):
        failed = await _response(source(), wire)
        assert failed.status_code == 502
        assert len(bodies) == 1 + probe
        assert all(item.close_calls == 1 for item in wires)
        assert admission._episode is None
        recovered = await asyncio.wait_for(_response(source(), wire), timeout=1)
        assert isinstance(recovered, StreamingResponse)
        await _serve(recovered)
        assert len(bodies) == 2 + probe
        assert all(item.close_calls == 1 for item in wires)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "messages", "responses"])
async def test_unstopped_first_messages_event_still_retries(route):
    def respond(ordinal):
        return 200, (
            [{"type": "message_delta", "delta": {"stop_reason": None}}]
            if ordinal == 1
            else _events()
        )

    async with _messages_path(route, respond) as (source, bodies, wires, admission):
        result = await _response(
            source(), "responses" if route == "responses" else "messages"
        )
        assert isinstance(result, StreamingResponse)
        await _serve(result)
        assert len(bodies) == 2 and bodies[0] == bodies[1]
        assert all(item.close_calls == 1 for item in wires)
        assert admission._episode is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete"])
async def test_stopped_first_responses_event_releases_admission(wire, terminal):
    event = {
        "type": terminal,
        "response": {"status": terminal.split(".")[1], "output": []},
    }
    async with _harness(
        "responses",
        lambda bodies: (200, [event] if len(bodies) == 1 else _events_for("responses")),
    ) as (send, bodies, transport):

        def adapter(kind, payload):
            if len(bodies) == 1:
                raise RetryableProviderProtocolError(
                    "terminal adapter rejected snapshot"
                )
            return payload

        transport._event_adapter_factory = lambda: adapter
        history = [{"role": "user", "content": "hi"}]
        failed = await _response(send(wire, history), wire)
        assert failed.status_code == 502 and len(bodies) == 1
        recovered = await asyncio.wait_for(
            _response(send(wire, history), wire), timeout=1
        )
        assert isinstance(recovered, StreamingResponse)
        await _serve(recovered)
        assert len(bodies) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "messages", "responses"])
async def test_stopped_messages_cannot_close_another_executions_recovery(route):
    leader = None

    class ConcurrentWire(Wire):
        async def __aiter__(self):
            assert leader is not None
            await leader.fail(_status_error(503))
            await leader.aclose()
            async for chunk in super().__aiter__():
                yield chunk

    async with _messages_path(
        route, lambda _: (200, ConcurrentWire(_sse({"type": "message_stop"})))
    ) as (source, _bodies, _wires, admission):
        other = admission.start_execution()
        leader = await _open(other)
        result = await _response(
            source(), "responses" if route == "responses" else "messages"
        )
        assert result.status_code == 502
        assert admission._episode is not None and admission._episode.leader is other
        probe = await asyncio.wait_for(_open(other), timeout=1)
        await probe.accept()
        await probe.aclose()
        other.succeed()
