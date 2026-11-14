"""Each streaming transport resolves a failed attempt by one shared rule set.

These tests drive the real transports over mock HTTP and observe only what
callers can see: upstream requests, admission outcomes, raised failures and
structured traces.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import httpx2
import pytest
from loguru import logger

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.trace import TRACE_PAYLOAD_BINDING
from free_claude_code.providers.gemini import GeminiProvider
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_gemini import _request as gemini_ingress_request
from tests.providers.test_gemini import _sse as gemini_sse
from tests.providers.test_history_transports import _harness


class _TimedOut(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    """An opened stream that times out before its first event."""

    def __init__(self, module: Any) -> None:
        self._module = module

    async def __aiter__(self):
        raise self._module.ReadTimeout("upstream went quiet")
        yield b""


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


def _events(traces: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [payload for payload in traces if payload["event"] == name]


def _outcomes(traces: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (payload["operation_kind"], payload["outcome"])
        for payload in _events(traces, "provider.attempt.resolved")
    ]


def _gemini(reply, *, max_attempts: int) -> GeminiProvider:
    return GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=max_attempts),
        transport=httpx.MockTransport(reply),
    )


def _gemini_payloads(*payloads: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads),
    )


async def _failure(stream) -> ExecutionFailure:
    with pytest.raises(ExecutionFailure) as caught:
        _ = [frame async for chunk in stream for frame in chunk.frames]
    return caught.value


async def _drain_gemini(provider: GeminiProvider) -> None:
    try:
        _ = [
            frame
            async for chunk in provider.stream_messages(
                gemini_ingress_request("messages"),
                response_model="public",
                request_id="matrix",
            )
            for frame in chunk.frames
        ]
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
async def test_continuation_failure_before_first_event_reaches_admission(
    protocol, traces
):
    module = httpx if protocol == "messages" else httpx2

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events(protocol, "Hello ", complete=False)
        if len(bodies) == 2:
            return 200, _TimedOut(module)
        return 200, text_events(protocol, "world.")

    async with _harness(protocol, reply, max_attempts=4) as (send, bodies, _):
        await delivered(
            send("messages", [{"role": "user", "content": "greet"}]), "messages"
        )

    assert len(bodies) == 3
    failed_continuation = "abandoned" if protocol == "chat" else "retryable_failure"
    assert _outcomes(traces) == [
        ("generation", "accepted"),
        ("continuation", failed_continuation),
        ("continuation", "accepted"),
    ]


@pytest.mark.asyncio
async def test_messages_stop_rejected_before_acceptance_accepts_the_attempt(traces):
    def reply(bodies):
        return 200, [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "model_context_window_exceeded"},
                "usage": {"output_tokens": 0},
            }
        ]

    async with _harness("messages", reply, max_attempts=3) as (send, bodies, _):
        failure = await _failure(
            send("messages", [{"role": "user", "content": "greet"}])
        )

    assert len(bodies) == 1
    assert _outcomes(traces) == [("generation", "accepted")]
    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED


@pytest.mark.asyncio
async def test_gemini_stop_rejected_before_acceptance(traces):
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return _gemini_payloads(
            {
                "candidates": [
                    {
                        "index": 0,
                        "finishReason": "STOP",
                        "groundingMetadata": {"webSearchQueries": ["q"]},
                    }
                ]
            }
        )

    with pytest.raises(ExecutionFailure) as caught:
        await _drain_gemini(_gemini(reply, max_attempts=3))

    assert calls == 1
    assert "grounding" in caught.value.message
    assert _outcomes(traces) == [("generation", "rejected")]


@pytest.mark.asyncio
async def test_gemini_inband_authentication_failure_refreshes_once(traces):
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _gemini_payloads({"error": {"code": 401, "message": "expired"}})
        return gemini_sse([{"text": "ok"}])

    await _drain_gemini(_gemini(reply, max_attempts=3))

    assert calls == 2


def _trace_shape(traces: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    return {
        name: [
            sorted(set(payload) - {"stage", "event", "source"})
            for payload in _events(traces, name)
        ]
        for name in ("provider.recovery.early_retry", "provider.response.error")
    }


_CHAT_SHAPE = {
    "provider.recovery.early_retry": [
        ["attempts_started", "max_attempts", "provider", "request_id", "retryable"]
    ],
    "provider.response.error": [
        [
            "exc_type",
            "failure_kind",
            "provider",
            "provider_retryable",
            "request_id",
            "status_code",
        ]
    ],
}
_RESPONSES_SHAPE = {
    "provider.recovery.early_retry": [
        ["attempts_started", "max_attempts", "provider", "request_id", "transport"]
    ],
    "provider.response.error": [
        [
            "exc_type",
            "failure_kind",
            "provider",
            "provider_retryable",
            "request_id",
            "status_code",
            "transport",
        ]
    ],
}
_MESSAGES_SHAPE = {
    "provider.recovery.early_retry": [],
    "provider.response.error": [
        ["failure_kind", "provider", "request_id", "transport"]
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "shape"),
    [
        ("chat", _CHAT_SHAPE),
        ("messages", _MESSAGES_SHAPE),
        ("responses", _RESPONSES_SHAPE),
    ],
)
async def test_exhausted_undelivered_stream_traces(protocol, shape, traces):
    module = httpx if protocol == "messages" else httpx2

    def reply(bodies):
        return 200, _TimedOut(module)

    async with _harness(protocol, reply, max_attempts=2) as (send, bodies, _):
        await _failure(send("messages", [{"role": "user", "content": "greet"}]))

    assert len(bodies) == 2
    assert _trace_shape(traces) == shape


@pytest.mark.asyncio
async def test_gemini_exhausted_undelivered_stream_traces(traces):
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TimedOut(httpx),
        )

    with pytest.raises(ExecutionFailure):
        await _drain_gemini(_gemini(reply, max_attempts=2))

    assert calls == 2
    assert _trace_shape(traces) == {
        "provider.recovery.early_retry": [],
        "provider.response.error": [],
    }
