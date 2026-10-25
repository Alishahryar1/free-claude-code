"""Gemini recovery preserves published tool calls and their replay state."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.responses import JSONResponse

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    bind_response_lifetime,
    gemini_sse_streaming_response,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.gemini.response import GeminiOutput, gemini_parts
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_response_streams import _serve
from tests.providers.test_anthropic_messages_transport import (
    Endpoint,
    Wire,
    _events,
    _sse,
    _transport,
)
from tests.providers.test_gemini_transport import collect, gemini_request
from tests.providers.test_history_transports import _harness


@pytest.mark.asyncio
async def test_messages_handoff_keeps_completed_reasoning_replay_and_calls():
    requests = []

    def reply(request):
        requests.append(request)
        events: list[JsonObject] = [
            *_events("x" * 70_000)[:4],
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "thinking",
                    "thinking": "thought",
                    "signature": "signed",
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "call-a",
                    "name": "read",
                    "input": {"path": "a"},
                },
            },
            {"type": "content_block_stop", "index": 2},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*events), httpx.ReadError("cutoff")]),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        body = _transport(client).stream_gemini(
            gemini_request(tools=[{"functionDeclarations": [{"name": "read"}]}]),
            endpoint_context=Endpoint(),
        )
        response = await gemini_sse_streaming_response(
            body,
            request_id="handoff",
            pre_start_error_response=lambda error: JSONResponse(
                status_code=502, content={"failure": str(error)}
            ),
        )
        assert isinstance(response, ManagedStreamingResponse)
        try:
            value, _ = await collect(response.body_iterator)
        finally:
            await response.aclose()
    parts = value["candidates"][0]["content"]["parts"]
    assert len(requests) == 1
    assert len([part for part in parts if "functionCall" in part]) == 1
    carriers = [
        part["thoughtSignature"] for part in parts if "thoughtSignature" in part
    ]
    assert len(carriers) == 1
    bundle = decode_bundle(carriers[0])
    assert bundle is not None and bundle.records[0].native["signature"] == "signed"
    assert "usageMetadata" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("stage", ["hidden", "portable", "structured", "retention"])
async def test_gemini_recovery_uses_published_evidence(protocol, stage, monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    prefix = (
        "x" * 1_100_000
        if stage == "retention"
        else '{"value":'
        if stage == "structured"
        else "Hello world"
    )

    def reply(bodies):
        if len(bodies) == 1 and stage == "hidden":
            return 200, []
        return 200, text_events(
            protocol,
            prefix if len(bodies) == 1 else "world again",
            complete=len(bodies) > 1,
        )

    options = (
        {"endpoint_context": Endpoint()}
        if protocol == "messages"
        else {"input_tokens": 0}
    )
    request = gemini_request(
        **(
            {
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": {"type": "object"},
                }
            }
            if stage == "structured"
            else {}
        )
    )
    async with _harness(protocol, reply, max_attempts=2) as (_, bodies, transport):
        response = await gemini_sse_streaming_response(
            transport.stream_gemini(
                request,
                request_id="recovery",
                response_model="public",
                reasoning=ReasoningPolicy.provider_default(),
                **options,
            ),
            request_id="recovery",
            pre_start_error_response=lambda error: JSONResponse(
                status_code=502, content={"failure": str(error)}
            ),
        )
        assert isinstance(response, ManagedStreamingResponse)
        try:
            raw = "".join([str(chunk) async for chunk in response.body_iterator])
        finally:
            await response.aclose()
    values = [event.data for event in parse_sse_text(raw)]
    text = "".join(
        part.get("text", "") for value in values for part in gemini_parts(value)
    )
    if stage in {"structured", "retention"}:
        assert len(bodies) == 1
        assert values[-1]["error"]["code"] >= 400
        assert not any(
            candidate.get("finishReason")
            for value in values
            for candidate in value.get("candidates", [])
        )
        assert text == prefix
    else:
        assert len(bodies) == 2
        assert text == ("world again" if stage == "hidden" else "Hello world again")
        assert values[-1]["candidates"][0]["finishReason"] == "STOP"
        assert len({value["responseId"] for value in values}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prefetch", "text", "call"])
async def test_gemini_cancel_closes_source_before_releasing_lease(phase):
    entered, closed = asyncio.Event(), asyncio.Event()
    release = AsyncMock()

    async def source():
        try:
            output = GeminiOutput("cancel", "model", include_thoughts=False)
            if phase != "prefetch":
                part: JsonObject = (
                    {"text": "started"}
                    if phase == "text"
                    else {"functionCall": {"id": "call", "name": "read", "args": {}}}
                )
                for frame in output.part(part):
                    yield frame
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def consume():
        response = await gemini_sse_streaming_response(
            source(),
            request_id="cancel",
            pre_start_error_response=lambda error: JSONResponse(
                status_code=502, content={"error": str(error)}
            ),
        )
        assert isinstance(response, ManagedStreamingResponse)
        await bind_response_lifetime(response, release)
        await _serve(response)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    if phase != "prefetch":
        release.assert_awaited_once()
    else:
        release.assert_not_awaited()  # The handler still owns a lease before binding.
