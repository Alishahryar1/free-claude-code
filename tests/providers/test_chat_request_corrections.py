"""A rejected Chat request is corrected at most once per kind, at the public boundary.

Each test drives a real Chat provider over mock HTTP and observes only the
request bodies the upstream received and the failure the caller sees.
"""

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import httpx2
import pytest

from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.providers.groq import GroqProvider
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_openai_responses_transport import _client

MODEL = "opaque-model-a"
Reply = Callable[[list[dict[str, Any]]], httpx2.Response]

_CAP = (
    "`max_completion_tokens` must be less than or equal to `16384`, the maximum "
    "value for `max_completion_tokens` is less than the `context_window` for this "
    "model"
)
_USAGE = "stream_options.include_usage is not supported by this model"
_TPM = (
    "Request too large for model `opaque-model-a` on tokens per minute "
    "(TPM): Limit 8000, Requested 26206, please reduce your message size"
)
_VOCABULARY = "`reasoning_effort` value `high` must be one of `none` or `default`"


def _ok() -> httpx2.Response:
    chunks = [
        {"role": "assistant", "content": "ok"},
        {},
    ]
    raw = "".join(
        "data: "
        + json.dumps(
            {
                "id": "chat_test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": "stop" if not delta else None,
                    }
                ],
            }
        )
        + "\n\n"
        for delta in chunks
    )
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=raw + "data: [DONE]\n\n",
    )


def _rejected(message: str, *, status: int = 400, **detail: str) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"error": {"message": message, "type": "invalid_request_error", **detail}},
    )


def _in_stream(message: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps({'error': {'message': message, 'code': 400}})}\n\n",
    )


def _groq(reply: Reply, bodies: list[dict[str, Any]]) -> GroqProvider:
    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return reply(bodies)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=_client(handle),
    ):
        return GroqProvider(
            make_provider_config(api_key="key", base_url=GROQ_DEFAULT_BASE),
            admission=immediate_admission(provider_name="GROQ", max_attempts=5),
        )


async def _send(
    provider: GroqProvider,
    *,
    max_tokens: int = 64000,
    reasoning: ReasoningPolicy = ReasoningPolicy.off(),
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    with bind_stream_delivery(StreamDeliveryState()):
        _ = [
            frame
            async for chunk in provider.stream_messages(
                request, request_id="corrections", reasoning=reasoning
            )
            for frame in chunk.frames
        ]


def _first_then_ok(first: httpx2.Response) -> Reply:
    return lambda bodies: first if len(bodies) == 1 else _ok()


@pytest.mark.asyncio
async def test_output_cap_is_clamped_and_learned_for_the_next_request():
    bodies: list[dict[str, Any]] = []
    provider = _groq(
        _first_then_ok(_rejected(_CAP, param="max_completion_tokens")), bodies
    )
    try:
        await _send(provider)
        await _send(provider)
    finally:
        await provider.cleanup()

    assert [body["max_completion_tokens"] for body in bodies] == [64000, 16384, 16384]


@pytest.mark.asyncio
async def test_stream_usage_rejection_drops_only_the_usage_option():
    bodies: list[dict[str, Any]] = []
    provider = _groq(_first_then_ok(_rejected(_USAGE)), bodies)
    try:
        await _send(provider)
    finally:
        await provider.cleanup()

    assert len(bodies) == 2
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in bodies[1]
    assert {k: v for k, v in bodies[0].items() if k != "stream_options"} == bodies[1]


@pytest.mark.asyncio
async def test_provider_correction_adjusts_the_rejected_field():
    bodies: list[dict[str, Any]] = []
    provider = _groq(_first_then_ok(_rejected(_VOCABULARY)), bodies)
    try:
        await _send(provider, reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH))
    finally:
        await provider.cleanup()

    assert [body["reasoning_effort"] for body in bodies] == ["high", "default"]


@pytest.mark.asyncio
async def test_late_provider_correction_runs_after_the_standard_ones():
    bodies: list[dict[str, Any]] = []
    provider = _groq(
        _first_then_ok(
            _rejected(_TPM, status=413, type="tokens", code="rate_limit_exceeded")
        ),
        bodies,
    )
    try:
        await _send(provider, max_tokens=20000)
    finally:
        await provider.cleanup()

    assert len(bodies) == 2
    assert bodies[1]["max_completion_tokens"] < bodies[0]["max_completion_tokens"]


@pytest.mark.asyncio
async def test_each_correction_is_applied_at_most_once():
    bodies: list[dict[str, Any]] = []
    provider = _groq(lambda bodies: _rejected(_USAGE), bodies)
    try:
        with pytest.raises(ExecutionFailure) as caught:
            await _send(provider)
    finally:
        await provider.cleanup()

    assert len(bodies) == 2
    assert caught.value.status_code == 400


@pytest.mark.asyncio
async def test_unrelated_rejection_is_not_corrected():
    bodies: list[dict[str, Any]] = []
    provider = _groq(
        lambda bodies: _rejected("messages: invalid role 'wizard'"), bodies
    )
    try:
        with pytest.raises(ExecutionFailure):
            await _send(provider)
    finally:
        await provider.cleanup()

    assert len(bodies) == 1


@pytest.mark.asyncio
async def test_in_stream_rejection_without_a_status_is_not_corrected():
    # The SDK raises an in-stream error without an HTTP status, which none of
    # the Chat corrections accept as a validation rejection.
    bodies: list[dict[str, Any]] = []
    provider = _groq(_first_then_ok(_in_stream(_CAP)), bodies)
    try:
        with pytest.raises(ExecutionFailure):
            await _send(provider)
    finally:
        await provider.cleanup()

    assert [body["max_completion_tokens"] for body in bodies] == [64000]
