"""Native Messages forwarding keeps the client's protocol on the Messages transport.

Forwarding differs from the translated Messages paths only where the client
speaks the upstream protocol itself: its own headers, model stops and failure
evidence reach the upstream or the client unchanged.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from loguru import logger

from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.trace import TRACE_PAYLOAD_BINDING
from tests.providers.test_anthropic_messages_transport import Wire, _events, _sse
from tests.providers.test_anthropic_provider import native_body, provider


@pytest.fixture
def traces() -> Iterator[list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        payload = message.record["extra"].get(TRACE_PAYLOAD_BINDING)
        if isinstance(payload, dict):
            payloads.append(payload)

    handle = logger.add(sink, level="DEBUG")
    try:
        yield payloads
    finally:
        logger.remove(handle)


async def _forward(p, body, **kwargs) -> list[str]:
    with bind_stream_delivery(StreamDeliveryState()):
        return [
            frame
            async for chunk in p.stream_native_messages(
                NativeMessagesRequest(body), **kwargs
            )
            for frame in chunk.frames
        ]


def _stream(*chunks: bytes) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=Wire([*chunks])
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_client_protocol_headers_reach_upstream(streaming):
    received: list[httpx.Request] = []

    def handle(request):
        received.append(request)
        if streaming:
            return _stream(_sse(*_events()))
        return httpx.Response(
            200, json={"type": "message", "content": [], "stop_reason": "end_turn"}
        )

    p = provider(handle)
    try:
        await _forward(
            p,
            native_body(streaming),
            request_headers={"anthropic-version": "2024-01-01"},
        )
    finally:
        await p.cleanup()

    (request,) = received
    assert request.headers["anthropic-version"] == "2024-01-01"
    assert request.headers["authorization"] == "Bearer upstream-secret"
    assert request.headers["anthropic-workspace-id"] == "wrkspc_test"
    assert "text/event-stream" not in request.headers.get("accept", "")


@pytest.mark.asyncio
async def test_context_window_stop_reaches_the_client_as_a_stop():
    p = provider(
        lambda request: _stream(_sse(*_events(stop="model_context_window_exceeded")))
    )
    try:
        frames = await _forward(p, native_body(True))
    finally:
        await p.cleanup()

    stops = [
        event.data["delta"]["stop_reason"]
        for event in parse_sse_text("".join(frames))
        if event.data["type"] == "message_delta"
    ]
    assert stops == ["model_context_window_exceeded"]


@pytest.mark.asyncio
async def test_error_event_failure_carries_upstream_evidence(traces):
    error: JsonObject = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "native evidence"},
    }
    p = provider(lambda request: _stream(_sse(_events()[0], error)))
    try:
        with pytest.raises(ExecutionFailure) as caught:
            await _forward(p, native_body(True), request_id="local-request")
    finally:
        await p.cleanup()

    message = caught.value.message
    assert caught.value.status_code == 400
    assert message.startswith("Upstream provider anthropic returned HTTP 400.")
    assert json.dumps(error, separators=(",", ":")) in message
    assert "local-request" in message
    (trace,) = [p for p in traces if p["event"] == "provider.response.error"]
    assert trace["transport"] == "messages"
    assert trace["status_code"] == 400
