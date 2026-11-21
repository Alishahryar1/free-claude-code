"""What each transport's attempt loop does around its upstream calls.

These drive the real transports over mock HTTP and observe what callers can
see: upstream requests, admission outcomes, raised failures and traces.
"""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from loguru import logger

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.trace import TRACE_PAYLOAD_BINDING
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.gemini.transport import GeminiTransport
from free_claude_code.providers.history_negotiation import HistoryModeCache
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_gemini import _request as gemini_ingress_request
from tests.providers.test_gemini import _sse as gemini_sse
from tests.providers.test_history_transports import _harness
from tests.providers.test_stream_failure_resolution_transports import (
    _drain_gemini,
    _events,
    _gemini,
    _outcomes,
)

HISTORY = [{"role": "user", "content": "greet"}]
HARNESSED = ["chat", "messages", "responses"]


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


_SENT_FIELDS = {
    "provider",
    "request_id",
    "execution_id",
    "gateway_model",
    "downstream_model",
    "transport",
}
_COMPLETED_FIELDS = {"provider", "request_id", "transport"}


def _lifecycle(payloads: list[dict[str, Any]]) -> tuple[int, int]:
    sent = _events(payloads, "provider.request.sent")
    completed = _events(payloads, "provider.response.completed")
    assert all(set(payload) >= _SENT_FIELDS for payload in sent)
    assert all(set(payload) >= _COMPLETED_FIELDS for payload in completed)
    return len(sent), len(completed)


async def _drain(stream: AsyncIterator[StreamOutput]) -> int | None:
    """Drain a stream; return the raised failure's status, if any."""
    with bind_stream_delivery(StreamDeliveryState()):
        try:
            _ = [frame async for chunk in stream for frame in chunk.frames]
        except ExecutionFailure as failure:
            return failure.status_code
    return None


class _UnreachableEndpoint:
    """An endpoint context whose credential exchange always fails."""

    def __init__(self) -> None:
        self.calls = 0

    async def endpoint(self, *, force_refresh: bool = False):
        self.calls += 1
        raise ExecutionFailure(
            kind=FailureKind.UPSTREAM,
            status_code=503,
            message="credential exchange unavailable",
            retryable=True,
        )


LIFECYCLE_TRACES = {
    "chat": (1, 1),
    "responses": (1, 1),
    "messages": (1, 1),
    "gemini": (1, 1),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", HARNESSED)
async def test_successful_request_lifecycle_traces(
    protocol: str, traces: list[JsonObject]
):
    async with _harness(
        protocol, lambda _bodies: (200, text_events(protocol, "hi"))
    ) as (send, _bodies, _):
        assert await _drain(send("messages", HISTORY)) is None

    assert _lifecycle(traces) == LIFECYCLE_TRACES[protocol]
    assert _outcomes(traces) == [("generation", "accepted")]


@pytest.mark.asyncio
async def test_gemini_successful_request_lifecycle_traces(traces: list[JsonObject]):
    def callback(_request: object):
        return gemini_sse([{"text": "ok"}])

    await _drain_gemini(_gemini(callback, max_attempts=2))

    assert _lifecycle(traces) == LIFECYCLE_TRACES["gemini"]
    assert _outcomes(traces) == [("generation", "accepted")]


# A credential failure is not a provider call: it opens no attempt and is
# terminal on every transport. (credential exchanges, admission outcomes)
ENDPOINT_FAILURE: tuple[int, list[tuple[str, str]]] = (1, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", HARNESSED)
async def test_endpoint_failure_opens_no_attempt(
    protocol: str, traces: list[JsonObject]
):
    context = _UnreachableEndpoint()
    async with _harness(
        protocol,
        lambda _bodies: (200, text_events(protocol, "hi")),
        max_attempts=2,
    ) as (send, bodies, _):
        status = await _drain(send("messages", HISTORY, endpoint_context=context))

    assert status == 503
    assert bodies == []
    assert (context.calls, _outcomes(traces)) == ENDPOINT_FAILURE


@pytest.mark.asyncio
async def test_gemini_endpoint_failure_opens_no_attempt(traces: list[JsonObject]):
    context = _UnreachableEndpoint()
    sent: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return gemini_sse([{"text": "ok"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        transport = GeminiTransport(
            client,
            immediate_admission(max_attempts=2),
            HistoryModeCache(),
            read_timeout=30,
        )
        status = await _drain(
            transport.stream(
                {"model": "gemini-test", "contents": [{"parts": [{"text": "hi"}]}]},
                endpoint_context=context,
                request_id="endpoint",
                response_model="public",
                output_factory=lambda: None,
            )
        )

    assert status == 503
    assert sent == []
    assert (context.calls, _outcomes(traces)) == ENDPOINT_FAILURE


EXHAUSTED_STATUS = {"chat": 503, "responses": 503, "messages": 529, "gemini": 529}


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", HARNESSED)
async def test_exhausted_attempts_resolve_each_failure_once(
    protocol: str, traces: list[JsonObject]
):
    async with _harness(
        protocol, lambda _bodies: (503, {"message": "busy"}), max_attempts=2
    ) as (send, bodies, _):
        status = await _drain(send("messages", HISTORY))

    assert len(bodies) == 2
    assert status == EXHAUSTED_STATUS[protocol]
    assert _outcomes(traces) == [("generation", "retryable_failure")] * 2
    assert len(_events(traces, "provider.response.error")) == 1


@pytest.mark.asyncio
async def test_gemini_exhausted_attempts_resolve_each_failure_once(
    traces: list[JsonObject],
):
    calls = []

    def reply(_request):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    with pytest.raises(ExecutionFailure) as caught:
        await _drain_gemini(_gemini(reply, max_attempts=2))

    assert len(calls) == 2
    assert caught.value.status_code == EXHAUSTED_STATUS["gemini"]
    assert _outcomes(traces) == [("generation", "retryable_failure")] * 2
    assert len(_events(traces, "provider.response.error")) == 1


THINKING_REQUIRED = {"message": "Thinking cannot be disabled"}

# The reasoning control each transport sends to turn reasoning off.
OFF_FIELD = {"chat": "reasoning", "messages": "thinking", "responses": "reasoning"}


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", HARNESSED)
async def test_continuation_still_drops_a_rejected_reasoning_control(protocol: str):
    def reply(bodies: list[JsonObject]):
        if len(bodies) == 1:
            return 200, text_events(protocol, "Hello ", complete=False)
        if len(bodies) == 2:
            return 400, THINKING_REQUIRED
        return 200, text_events(protocol, "world.")

    async with _harness(protocol, reply) as (send, bodies, _):
        await delivered(
            send("messages", HISTORY, reasoning=ReasoningPolicy.prefer_off()),
            "messages",
        )

    assert len(bodies) == 3
    assert [OFF_FIELD[protocol] in body for body in bodies] == [True, True, False]


@pytest.mark.asyncio
async def test_gemini_continuation_drops_a_rejected_reasoning_control():
    bodies: list[dict[str, Any]] = []

    def reply(request: httpx.Request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            partial = {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "Hello "}]},
                    }
                ]
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(partial)}\n\n",
            )
        if len(bodies) == 2:
            return httpx.Response(400, json={"error": THINKING_REQUIRED})
        return gemini_sse([{"text": "world."}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=5),
        transport=httpx.MockTransport(reply),
    )
    request = gemini_ingress_request("messages").model_copy(
        update={"model": "gemini-2.5-flash"}
    )
    try:
        await delivered(
            provider.stream_messages(
                request,
                reasoning=ReasoningPolicy.prefer_off(),
                response_model="public",
            ),
            "messages",
        )
    finally:
        await provider.cleanup()

    assert [
        "thinkingBudget" in body["generationConfig"].get("thinkingConfig", {})
        for body in bodies
    ] == [True, True, False]
