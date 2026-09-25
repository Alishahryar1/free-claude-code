"""Sampling compatibility follows actual upstream rejections, not model IDs."""

import asyncio
import json
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import httpx2
import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_api.provider import OpenAIAPIProvider
from tests.providers.test_openai_api_provider import (
    _collect,
    _complete_stream,
    _provider,
    _sse,
)
from tests.providers.test_openai_responses_transport import (
    _client,
    _collect_native,
    _transport,
)


def _request(ingress: str, model: str = "future-model"):
    sampling = {"model": model, "temperature": 0.5, "top_p": 0.8}
    if ingress == "messages":
        return MessagesRequest.model_validate(
            {**sampling, "messages": [{"role": "user", "content": "hello"}]}
        )
    return OpenAIResponsesRequest.model_validate(
        {**sampling, "input": "hello", "metadata": {"source": "sampling-test"}}
    )


def _stream(
    provider: OpenAIAPIProvider,
    request: MessagesRequest | OpenAIResponsesRequest,
    reasoning: ReasoningPolicy = ReasoningPolicy.provider_default(),
):
    if isinstance(request, MessagesRequest):
        return provider.stream_messages(request, reasoning=reasoning)
    return provider.stream_responses(request, reasoning=reasoning)


def _rejection(param: str, *, default_only: bool = False) -> dict[str, Any]:
    # Captured HTTP error shapes; stream variants below are compatibility contracts.
    return {
        "type": "invalid_request_error",
        "param": param,
        "code": "unsupported_value" if default_only else "unsupported_parameter",
        "message": (
            f"Unsupported value: '{param}' does not support 0.5 with this model. "
            "Only the default (1) value is supported."
            if default_only
            else f"Unsupported parameter: '{param}' is not supported with this model."
        ),
    }


def _failure(error: Any, surface: str = "http", status: int = 400):
    if surface == "http":
        return httpx2.Response(status, json={"error": error})
    if surface == "flat":
        payload = {**error, "type": "error", "sequence_number": 0}
    elif surface == "failed":
        payload = {
            "type": "response.failed",
            "response": {"id": "resp_rejected", "error": error},
        }
    else:
        payload = {"type": "error", "error": error}
    return httpx2.Response(
        200,
        text=_sse((payload["type"], payload)),
        headers={"content-type": "text/event-stream"},
    )


def _success():
    return httpx2.Response(
        200,
        text=_complete_stream("hello"),
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize("model", ["future-model", "gpt-6-sol"])
@pytest.mark.parametrize(
    "reasoning", [ReasoningPolicy.provider_default(), ReasoningPolicy.prefer_off()]
)
async def test_supported_sampling_is_never_removed_proactively(
    ingress, model, reasoning
):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _success()

    provider = _provider(httpx2.MockTransport(handler))
    try:
        await _collect(_stream(provider, _request(ingress, model), reasoning))
    finally:
        await provider.cleanup()
    assert len(bodies) == 1
    assert bodies[0]["temperature"] == 0.5
    assert bodies[0]["top_p"] == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize("surface", ["http", "flat", "failed", "sdk"])
@pytest.mark.parametrize("param", ["temperature", "top_p"])
@pytest.mark.parametrize("default_only", [False, True])
async def test_only_explicitly_rejected_sampling_is_removed(
    ingress, surface, param, default_only
):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if param in body:
            return _failure(_rejection(param, default_only=default_only), surface)
        return _success()

    request = _request(ingress)
    original = request.model_dump()
    provider = _provider(httpx2.MockTransport(handler), max_attempts=2)
    try:
        output = await _collect(_stream(provider, request))
    finally:
        await provider.cleanup()
    assert len(bodies) == 2
    assert bodies[1] == {key: value for key, value in bodies[0].items() if key != param}
    assert request.model_dump() == original
    events = parse_sse_text(output)
    terminal = "message_stop" if ingress == "messages" else "response.completed"
    assert sum(event.event == terminal for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize("order", [("temperature", "top_p"), ("top_p", "temperature")])
@pytest.mark.parametrize("concurrent", [False, True])
async def test_two_rejections_are_request_local(ingress, order, concurrent):
    bodies = []
    both_started = asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if concurrent and all(param in body for param in order):
            if sum(all(param in item for param in order) for item in bodies) == 2:
                both_started.set()
            await both_started.wait()
        for param in order:
            if param in body:
                return _failure(_rejection(param))
        return _success()

    provider = _provider(httpx2.MockTransport(handler), max_attempts=3)
    request = _request(ingress)
    try:
        if concurrent:
            await asyncio.gather(
                *(_collect(_stream(provider, request)) for _ in range(2))
            )
        else:
            for _ in range(2):
                await _collect(_stream(provider, request))
    finally:
        await provider.cleanup()
    assert len(bodies) == 6
    assert sum(all(param in body for param in order) for body in bodies) == 2
    assert sum(order[0] not in body and order[1] in body for body in bodies) == 2
    assert sum(all(param not in body for param in order) for body in bodies) == 2
    assert request.temperature == 0.5 and request.top_p == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        {**_rejection("temperature"), "param": "max_output_tokens"},
        {**_rejection("temperature"), "param": None},
        {**_rejection("temperature"), "param": ["temperature"]},
        {**_rejection("temperature"), "code": "invalid_value"},
        {**_rejection("temperature"), "code": "unsupported_value"},
        {
            **_rejection("temperature"),
            "code": "unsupported_value",
            "message": "Must be between 0 and 2",
        },
        {**_rejection("temperature", default_only=True), "param": "top_p"},
        {
            **_rejection("temperature", default_only=True),
            "message": "Only the default (1) value is supported.",
        },
        {"message": "Sampling unsupported with this model"},
        None,
        "Unsupported parameter: temperature",
    ],
)
async def test_ambiguous_or_unrelated_validation_is_not_corrected(error):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _failure(error)

    provider = _provider(httpx2.MockTransport(handler), max_attempts=3)
    try:
        with pytest.raises(ExecutionFailure):
            await _collect(_stream(provider, _request("responses")))
    finally:
        await provider.cleanup()
    assert len(bodies) == 1
    assert bodies[0]["temperature"] == 0.5 and bodies[0]["top_p"] == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, attempts", [(401, 1), (403, 1), (422, 1), (429, 2), (500, 2)]
)
async def test_other_http_failures_do_not_remove_sampling(status, attempts):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _failure(_rejection("temperature"), status=status)

    provider = _provider(httpx2.MockTransport(handler), max_attempts=2)
    try:
        with pytest.raises(ExecutionFailure):
            await _collect(_stream(provider, _request("responses")))
    finally:
        await provider.cleanup()
    assert len(bodies) == attempts
    assert all(body == bodies[0] for body in bodies)
    assert bodies[0]["temperature"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_first", [False, True])
async def test_reasoning_and_sampling_corrections_compose(reasoning_first):
    reasoning_error = {
        "code": "unsupported_value",
        "param": "reasoning.effort",
        "message": "Value 'none' is not supported",
    }
    errors = [reasoning_error, _rejection("temperature")]
    if not reasoning_first:
        errors.reverse()
    errors.append(_rejection("top_p"))
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) <= len(errors):
            return _failure(errors[len(bodies) - 1])
        return _success()

    provider = _provider(httpx2.MockTransport(handler), max_attempts=5)
    try:
        output = await _collect(
            _stream(provider, _request("messages"), ReasoningPolicy.prefer_off())
        )
    finally:
        await provider.cleanup()
    assert text_content(parse_sse_text(output)) == "hello"
    assert len(bodies) == 4
    for index, error in enumerate(errors):
        field = "reasoning" if error["param"] == "reasoning.effort" else error["param"]
        assert field in bodies[index]
        assert bodies[index + 1] == {
            key: value for key, value in bodies[index].items() if key != field
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [1, 2, 3])
async def test_sampling_corrections_share_the_attempt_budget(budget):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return _failure({"code": "server_error", "message": "busy"}, status=500)
        param = "temperature" if "temperature" in body else "top_p"
        return _failure(_rejection(param))

    provider = _provider(httpx2.MockTransport(handler), max_attempts=budget)
    try:
        with pytest.raises(ExecutionFailure):
            await _collect(_stream(provider, _request("responses")))
    finally:
        await provider.cleanup()
    assert len(bodies) == budget
    if budget > 1:
        assert bodies[1] == bodies[0]
    if budget > 2:
        assert "temperature" not in bodies[2] and "top_p" in bodies[2]


@pytest.mark.asyncio
async def test_rejection_of_an_already_removed_field_stops_correction():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _failure(_rejection("temperature"))

    provider = _provider(httpx2.MockTransport(handler), max_attempts=5)
    try:
        with pytest.raises(ExecutionFailure):
            await _collect(_stream(provider, _request("responses")))
    finally:
        await provider.cleanup()
    assert len(bodies) == 2
    assert "temperature" not in bodies[1] and "top_p" in bodies[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("history_first", [False, True])
async def test_history_and_sampling_corrections_preserve_each_other(history_first):
    history_error = {
        "code": "invalid_encrypted_content",
        "param": "input[0].encrypted_content",
        "message": "The encrypted content could not be verified.",
    }
    errors = [history_error, _rejection("temperature")]
    if not history_first:
        errors.reverse()
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) <= 2:
            return _failure(errors[len(bodies) - 1])
        return _success()

    request = OpenAIResponsesRequest(
        model="future-model",
        temperature=0.5,
        top_p=0.8,
        input=[
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "opaque-history",
                "summary": [{"type": "summary_text", "text": "Saved reasoning"}],
            },
            {"role": "user", "content": "continue"},
        ],
    )
    original = request.model_dump()
    provider = _provider(httpx2.MockTransport(handler), max_attempts=3)
    try:
        await _collect(_stream(provider, request))
    finally:
        await provider.cleanup()
    assert len(bodies) == 3
    expected = deepcopy(bodies[0])
    for index, error in enumerate(errors):
        if error is history_error:
            expected["input"][0] = {
                "role": "assistant",
                "content": "[Earlier reasoning summary]\nSaved reasoning",
            }
        else:
            expected.pop("temperature")
        assert bodies[index + 1] == expected
    assert request.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize("committed", [False, True])
async def test_sampling_correction_respects_stream_commitment(ingress, committed):
    bodies = []
    partial = "x" * 70_000 if committed else "discarded"

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) > 1:
            return _success()
        return httpx2.Response(
            200,
            text=_sse(
                (
                    "response.created",
                    {"type": "response.created", "response": {"id": "resp_old"}},
                ),
                (
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": partial},
                ),
                (
                    "error",
                    {
                        **_rejection("temperature"),
                        "type": "error",
                        "sequence_number": 2,
                    },
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    provider = _provider(httpx2.MockTransport(handler), max_attempts=3)
    chunks = []
    try:
        expected_error = (
            pytest.raises(ExecutionFailure)
            if committed and ingress == "messages"
            else nullcontext()
        )
        with expected_error:
            async for chunk in _stream(provider, _request(ingress)):
                chunks.extend((chunk,))
    finally:
        await provider.cleanup()
    output = "".join(chunks)
    events = parse_sse_text(output)
    start = "message_start" if ingress == "messages" else "response.created"
    assert sum(event.event == start for event in events) == 1
    if committed:
        assert len(bodies) == 1
        assert partial in output
        if ingress == "responses":
            assert sum(event.event == "response.failed" for event in events) == 1
    else:
        assert len(bodies) == 2
        assert partial not in output
        assert "resp_old" not in output
        assert "temperature" not in bodies[1] and "top_p" in bodies[1]


@pytest.mark.asyncio
async def test_stream_failure_without_a_parameter_is_not_guessed():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return _failure({**_rejection("temperature"), "param": None}, "failed")

    provider = _provider(httpx2.MockTransport(handler), max_attempts=3)
    try:
        with pytest.raises(ExecutionFailure):
            await _collect(_stream(provider, _request("responses")))
    finally:
        await provider.cleanup()
    assert calls == 1


@pytest.mark.asyncio
async def test_shared_transport_does_not_apply_openai_api_sampling_correction():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _failure(_rejection("temperature"))

    client = _client(handler)
    try:
        with pytest.raises(ExecutionFailure):
            await _collect_native(
                _transport(client, max_attempts=3), _request("responses")
            )
    finally:
        await client.close()
    assert len(bodies) == 1
    assert bodies[0]["temperature"] == 0.5 and bodies[0]["top_p"] == 0.8
