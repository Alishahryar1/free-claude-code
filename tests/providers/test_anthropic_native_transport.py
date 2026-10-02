"""Native controls, upstream recovery, and admitted resource lifetime."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import ReplayRecord, encode_replay
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.history_replay import replay_origin
from tests.providers.test_anthropic_messages_transport import Wire, _events, _sse
from tests.providers.test_anthropic_provider import native_body, provider


async def collect(p, body=None, **kwargs):
    return [
        event
        async for event in p.stream_native_messages(
            NativeMessagesRequest(body or native_body()), **kwargs
        )
    ]


@pytest.mark.asyncio
async def test_native_omissions_helpers_controls_and_cookie_isolation():
    received = []

    def handle(request):
        received.append(request)
        return httpx.Response(
            200,
            json={"type": "message", "content": []},
            headers={"set-cookie": "upstream=session"},
        )

    p = provider(handle)
    body = native_body()
    del body["max_tokens"]
    del body["stream"]
    body.update(
        {
            "original_model": "fcc",
            "resolved_provider_model": "fcc",
            "betas": ["a-beta", "b-beta"],
            "extra_body": {"new_option": {"data": True}},
            "stop_sequences": ["</severity>"],
            "temperature": 0.7,
            "cache_control": {"type": "ephemeral"},
            "thinking": {
                "type": "enabled",
                "budget_tokens": 3579,
                "display": "summarized",
            },
        }
    )
    try:
        await collect(
            p,
            body,
            request_headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "a-beta,c-beta",
                "x-api-key": "foreign",
                "Host": "foreign.test",
            },
        )
        await collect(p)
        sent = json.loads(received[0].content)
        assert "stream" not in sent and "max_tokens" not in sent
        assert "original_model" not in sent and "resolved_provider_model" not in sent
        assert "betas" not in sent and "extra_body" not in sent
        assert sent["new_option"] == {"data": True}
        assert sent["thinking"] == body["thinking"]
        assert sent["stop_sequences"] == body["stop_sequences"]
        assert sent["temperature"] == 0.7
        assert sent["cache_control"] == body["cache_control"]
        assert received[0].headers["anthropic-beta"] == "a-beta,c-beta,b-beta"
        assert "anthropic-beta" not in received[1].headers
        assert all(
            "cookie" not in request.headers and "x-api-key" not in request.headers
            for request in received
        )
        assert received[0].headers["host"] == "api.anthropic.com"
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper",
    [
        {"model": "other"},
        {"headers": {"Authorization": "other"}},
        {"api_key": "other"},
        {"stream": True},
    ],
)
async def test_helpers_cannot_override_transport_ownership(helper):
    def handle(request):
        raise AssertionError("Invalid helpers reached upstream")

    p = provider(handle)
    try:
        with pytest.raises(InvalidRequestError):
            await collect(p, {**native_body(), "extra_body": helper})
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_hosted_tool_stream_preserves_signatures_usage_and_pause_turn():
    events = _events(stop="pause_turn")
    events[1]["content_block"] = {
        "type": "server_tool_use",
        "id": "srv",
        "name": "web_search",
        "input": {},
    }
    events[2]["delta"] = {
        "type": "input_json_delta",
        "partial_json": '{"query":"hello"}',
    }
    events[4]["usage"] = {
        "output_tokens": 15,
        "cache_read_input_tokens": 8,
        "server_tool_use": {"web_search_requests": 1},
    }
    events[4:4] = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srv",
                "content": [
                    {"encrypted_content": "opaque", "url": "https://example.test"}
                ],
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "future_execution_result",
                "data": {"keep": True},
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "native-signature"},
        },
        {"type": "content_block_stop", "index": 2},
    ]
    wire = Wire([_sse(*events)])
    p = provider(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )
    )
    try:
        output = parse_sse_text(
            "".join(await collect(p, native_body(True), response_model="alias"))
        )
        expected = json.loads(json.dumps(events))
        expected[0]["message"]["model"] = "alias"
        assert [event.data for event in output] == expected
        assert wire.closed
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [401, 402, 403, 429, 529])
async def test_http_failure_status_evidence_and_bounded_recovery(streaming, status):
    wires = []

    def handle(request):
        wire = Wire(
            [
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "test provider evidence",
                        },
                        "request_id": "upstream-request",
                    }
                ).encode()
            ]
        )
        wires.append(wire)
        return httpx.Response(
            status, stream=wire, headers={"request-id": "upstream-request"}
        )

    p = provider(handle)
    try:
        with pytest.raises(ExecutionFailure) as result:
            await collect(p, native_body(streaming), request_id="local-request")
        assert result.value.status_code == status
        assert "test provider evidence" in result.value.message
        assert "local-request" in result.value.message
        assert "upstream-request" in result.value.message
        assert len(wires) == (2 if status in {429, 529} else 1)
        assert all(wire.closed for wire in wires)
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_connection_recovery_before_output_closes_failed_response(streaming):
    first = Wire([httpx.ReadError("connection closed")])
    second = Wire(
        [_sse(*_events()) if streaming else b'{"type":"message","content":[]}']
    )
    wires = iter([first, second])
    p = provider(
        lambda request: httpx.Response(
            200,
            stream=next(wires),
            headers={
                "content-type": "text/event-stream" if streaming else "application/json"
            },
        )
    )
    try:
        assert await collect(p, native_body(streaming))
        assert first.closed and second.closed
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_tail",
    [
        b"",
        b"event: content_block_delta\ndata: not-json\n\n",
        _sse({"type": "message_start", "message": {}}),
    ],
)
async def test_malformed_or_incomplete_stream_after_output_never_replays(bad_tail):
    calls = []
    wire = Wire([_sse(_events()[0]), bad_tail])

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    p = provider(handle)
    try:
        stream = p.stream_native_messages(NativeMessagesRequest(native_body(True)))
        assert "message_start" in await anext(stream)
        with pytest.raises(ExecutionFailure):
            await anext(stream)
        assert len(calls) == 1
        assert wire.closed
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_early_consumer_exit_releases_admission_and_response():
    wire = Wire([_sse(*_events())])
    p = provider(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )
    )
    stream = p.stream_native_messages(NativeMessagesRequest(native_body(True)))
    try:
        await anext(stream)
        await stream.aclose()
        assert wire.closed
        assert await collect(p, native_body(True))
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_cancelled_read_closes_response_without_retry(streaming):
    reading = asyncio.Event()
    closed = asyncio.Event()
    calls = []

    class BlockingWire(httpx.AsyncByteStream):
        async def __aiter__(self):
            reading.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.set()

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=BlockingWire()
        )

    p = provider(handle)
    task = asyncio.create_task(collect(p, native_body(streaming)))
    try:
        await asyncio.wait_for(reading.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert len(calls) == 1
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_native_history_restores_matching_fcc_carriers_and_rejects_foreign():
    received = []

    def handle(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json={"type": "message", "content": []})

    p = provider(handle)
    native: JsonObject = {
        "type": "thinking",
        "thinking": "native",
        "signature": "original-signature",
    }
    origin = replay_origin(
        "anthropic", "messages", "claude-test", endpoint=await p.endpoint()
    )
    body = native_body()
    body["messages"] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "client",
                    "signature": encode_replay(ReplayRecord(origin, native)),
                }
            ],
        }
    ]
    try:
        await collect(p, body)
        assert received[0]["messages"][0]["content"] == [native]
        body["messages"][0]["content"][0]["signature"] = encode_replay(
            ReplayRecord(replace(origin, provider="foreign"), native)
        )
        with pytest.raises(InvalidRequestError, match="different provider"):
            await collect(p, body)
        assert len(received) == 1
    finally:
        await p.cleanup()
