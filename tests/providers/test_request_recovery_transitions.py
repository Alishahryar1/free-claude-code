"""Recovery state survives transitions across the real upstream transports."""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from io import StringIO
from unittest.mock import patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.sse import parse_sse_text
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from tests.providers.support import make_provider_config
from tests.providers.test_anthropic_messages_transport import _events
from tests.providers.test_anthropic_messages_transport import (
    _transport as messages_transport,
)
from tests.providers.test_history_transports import _events_for


class Endpoint:
    def __init__(self):
        self.calls = []
        self.token = "original"

    async def endpoint(self, *, force_refresh=False):
        self.calls.append(force_refresh)
        if force_refresh:
            self.token = "fresh"
        return HttpEndpoint("https://provider.invalid/v1/", {}, self.token)


class Wire(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, content):
        self.content = content
        self.close_calls = 0

    async def __aiter__(self):
        yield self.content

    async def aclose(self):
        self.close_calls += 1


@asynccontextmanager
async def _transport(protocol, responder, *, endpoint=None):
    endpoint = endpoint if endpoint is not None else Endpoint()
    bodies = []
    wires = []
    admission = ProviderAdmissionController(
        provider_name="TEST",
        rate_limit=1_000_000,
        rate_window=1,
        max_concurrency=1,
        max_attempts=5,
        base_delay=0,
        max_delay=0,
        jitter=0,
    )

    def handler(request):
        bodies.append(json.loads(request.content))
        status, payload = responder(len(bodies))
        module = httpx if protocol == "messages" else httpx2
        if status == 200:
            content = "".join(
                (f"event: {event['type']}\n" if protocol == "messages" else "")
                + f"data: {json.dumps(event)}\n\n"
                for event in payload
            )
            content_type = "text/event-stream"
        else:
            content = json.dumps({"error": payload})
            content_type = "application/json"
        wire = Wire(content.encode())
        wires.append(wire)
        return module.Response(
            status, headers={"content-type": content_type}, stream=wire
        )

    if protocol == "messages":
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = messages_transport(client, admission)
    else:
        pool = httpx2.MockTransport(handler)
        client = AsyncOpenAI(
            api_key="unused",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=pool),
        )
        if protocol == "responses":
            provider = OpenAIResponsesTransport(
                client=client,
                admission=admission,
                provider_name="TEST",
                read_timeout_s=3,
                log_raw_sse_events=False,
                endpoint_transport=pool,
            )
        else:
            provider = OpenAIChatProvider(
                make_provider_config(None, "https://provider.invalid/v1/"),
                profile=OpenAIChatProfile(
                    OpenAIChatRequestPolicy(
                        "TEST", ReasoningReplayMode.REASONING_CONTENT
                    ),
                    NamedEffortReasoning((), disabled_value="none"),
                    structured_reasoning_details=True,
                ),
                admission=admission,
                client=client,
                endpoint_transport=pool,
            )
    try:
        yield provider, endpoint, bodies, wires, admission
    finally:
        if isinstance(provider, OpenAIChatProvider):
            await provider.cleanup()
        if isinstance(client, httpx.AsyncClient):
            await client.aclose()
        else:
            await client.close()


def _request(protocol):
    block = (
        {"type": "thinking", "thinking": "previous", "signature": "opaque-original"}
        if protocol == "messages"
        else {"type": "redacted_thinking", "data": "opaque-original"}
    )
    return MessagesRequest.model_validate(
        {
            "model": "upstream",
            "messages": [
                {
                    "role": "assistant",
                    "content": [block, {"type": "text", "text": "17"}],
                },
                {"role": "user", "content": "continue"},
            ],
            "thinking": {"type": "disabled"},
        }
    )


def _stream(provider, endpoint, request):
    return provider.stream_messages(
        request,
        endpoint_context=endpoint,
        reasoning=ReasoningPolicy.prefer_off(),
        request_id="recovery-transition",
        response_model="public",
        **(
            {"input_tokens": 0}
            if isinstance(provider, OpenAIResponsesTransport)
            else {}
        ),
    )


def _history_error(protocol):
    return {
        "type": "invalid_request_error",
        "code": "invalid_signature"
        if protocol == "messages"
        else "invalid_encrypted_content",
        "message": "invalid signature in thinking block",
    }


def _reasoning_error():
    return {
        "type": "invalid_request_error",
        "message": "Reasoning is mandatory and cannot be disabled.",
    }


def _partial(protocol, text=""):
    if protocol == "messages":
        return _events(text)[:3] if text else _events()[:1]
    if protocol == "responses":
        events = _events_for(protocol)[:1]
        if text:
            events.append(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "text",
                    "content_index": 0,
                    "output_index": 0,
                    "delta": text,
                    "logprobs": [],
                }
            )
        return events
    chunk = deepcopy(_events_for("chat")[-1])
    chunk["choices"][0] = {
        "index": 0,
        "delta": {"content": text} if text else {"role": "assistant"},
        "finish_reason": None,
    }
    return [chunk]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_mixed_recovery_keeps_one_budget_and_correction_history(
    protocol, exhausted
):
    def respond(ordinal):
        if ordinal == 1:
            return 401, {"type": "authentication_error", "message": "expired"}
        if ordinal == 2:
            return 400, _history_error(protocol)
        if ordinal == 3:
            return 400, _reasoning_error()
        if ordinal == 4:
            return 200, _partial(protocol)
        if exhausted:
            return 503, {"type": "overloaded_error", "message": "unavailable"}
        return 200, _events_for(protocol)

    request = _request(protocol)
    original = deepcopy(request.model_dump())
    async with _transport(protocol, respond) as (provider, endpoint, bodies, wires, _):
        with patch("free_claude_code.providers.admission.trace_event") as trace:
            if exhausted:
                with pytest.raises(ExecutionFailure):
                    _ = [event async for event in _stream(provider, endpoint, request)]
            else:
                output = [event async for event in _stream(provider, endpoint, request)]
                parsed = parse_sse_text("".join(output))
                assert sum(event.event == "message_start" for event in parsed) == 1
                assert sum(event.event == "message_stop" for event in parsed) == 1
        starts = [
            call.kwargs
            for call in trace.call_args_list
            if call.kwargs.get("event") == "provider.attempt.started"
        ]
        assert len(starts) == 5
        assert len({start["execution_id"] for start in starts}) == 1
        assert endpoint.calls == [False, True, False, False, False]
        assert len(bodies) == 5 and bodies[0] == bodies[1]
        assert "opaque-original" in json.dumps(bodies[1])
        assert all("opaque-original" not in json.dumps(body) for body in bodies[2:])
        field = {
            "chat": "reasoning_effort",
            "responses": "reasoning",
            "messages": "thinking",
        }[protocol]
        assert field in bodies[2]
        assert all(field not in body for body in bodies[3:])
        assert all(wire.close_calls == 1 for wire in wires)
    assert request.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_refresh_allowance_survives_an_intervening_correction(protocol):
    def respond(ordinal):
        if ordinal == 2:
            return 400, _history_error(protocol)
        return 401, {"type": "authentication_error", "message": "expired"}

    async with _transport(protocol, respond) as (provider, endpoint, bodies, wires, _):
        with pytest.raises(ExecutionFailure) as failure:
            _ = [
                event async for event in _stream(provider, endpoint, _request(protocol))
            ]
        assert failure.value.status_code == 401
        assert len(bodies) == 3 and endpoint.calls == [False, True, False]
        assert all(wire.close_calls == 1 for wire in wires)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
@pytest.mark.parametrize("refresh", [False, True])
async def test_credential_resolution_preserves_admission_order(protocol, refresh):
    class FailingEndpoint(Endpoint):
        failed = False

        async def endpoint(self, *, force_refresh=False):
            if force_refresh == refresh and not self.failed:
                self.failed = True
                self.calls.append(force_refresh)
                raise httpx.ReadError("credential service unavailable")
            return await super().endpoint(force_refresh=force_refresh)

    def respond(ordinal):
        if refresh and ordinal == 1:
            return 401, {"type": "authentication_error", "message": "expired"}
        return 200, _events_for(protocol)

    async with _transport(protocol, respond, endpoint=FailingEndpoint()) as (
        provider,
        endpoint,
        bodies,
        wires,
        _,
    ):
        if protocol == "responses":
            with pytest.raises(
                ExecutionFailure, match="credential service unavailable"
            ):
                _ = [
                    event
                    async for event in _stream(provider, endpoint, _request(protocol))
                ]
            assert endpoint.calls == ([False, True] if refresh else [False])
            assert len(bodies) == int(refresh)
        else:
            assert [
                event async for event in _stream(provider, endpoint, _request(protocol))
            ]
            assert endpoint.calls == (
                [False, True, True] if refresh else [False, False]
            )
            assert len(bodies) == 1 + int(refresh)
        assert all(wire.close_calls == 1 for wire in wires)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
@pytest.mark.parametrize("failure_kind", ["authentication", "reasoning"])
async def test_committed_output_prevents_generation_corrections(protocol, failure_kind):
    error = (
        {
            "type": "authentication_error",
            "code": "invalid_api_key",
            "message": "expired",
        }
        if failure_kind == "authentication"
        else _reasoning_error()
    )
    if protocol == "responses" and failure_kind == "reasoning":
        error["code"] = "invalid_request_error"
    events = _partial(protocol, "already visible " * 5000)
    events.append({"type": "error", "error": error})
    async with _transport(protocol, lambda _: (200, events)) as (
        provider,
        endpoint,
        bodies,
        wires,
        _,
    ):
        output = StringIO()
        with pytest.raises(ExecutionFailure):
            async for event in _stream(provider, endpoint, _request(protocol)):
                output.write(event)
        assert "already visible" in output.getvalue()
        assert len(bodies) == 1 and endpoint.calls == [False]
        assert all(wire.close_calls == 1 for wire in wires)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_cancelled_forced_resolution_closes_attempt_and_releases_admission(
    protocol,
):
    entered = asyncio.Event()
    wires = []

    class BlockingEndpoint(Endpoint):
        async def endpoint(self, *, force_refresh=False):
            if force_refresh:
                self.calls.append(force_refresh)
                assert wires and all(wire.close_calls == 1 for wire in wires)
                entered.set()
                await asyncio.Event().wait()
            return await super().endpoint(force_refresh=force_refresh)

    async def consume(provider, endpoint):
        return [
            event async for event in _stream(provider, endpoint, _request(protocol))
        ]

    async with _transport(
        protocol,
        lambda _: (401, {"type": "authentication_error", "message": "expired"}),
        endpoint=BlockingEndpoint(),
    ) as (provider, endpoint, bodies, owned_wires, admission):
        wires = owned_wires
        task = asyncio.create_task(consume(provider, endpoint))
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(bodies) == 1 and endpoint.calls == [False, True]
        execution = admission.start_execution()
        attempt = await asyncio.wait_for(
            execution.open_attempt(ProviderOperationKind.GENERATION), timeout=3
        )
        await attempt.aclose()
        execution.abandon()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["authentication", "stream_usage"])
async def test_committed_chat_continuation_has_separate_body_correction_scope(
    failure_kind,
):
    def respond(ordinal):
        if ordinal == 1:
            return 200, _partial("chat", "already visible " * 5000)
        if ordinal == 2:
            if failure_kind == "authentication":
                return 401, {"type": "authentication_error", "message": "expired"}
            return 400, {
                "type": "invalid_request_error",
                "message": "stream_options is unsupported",
            }
        return 200, _events_for("chat")

    async with _transport("chat", respond) as (provider, endpoint, bodies, wires, _):
        output = StringIO()
        if failure_kind == "authentication":
            with pytest.raises(ExecutionFailure):
                async for event in _stream(provider, endpoint, _request("chat")):
                    output.write(event)
            assert len(bodies) == 2
        else:
            async for event in _stream(provider, endpoint, _request("chat")):
                output.write(event)
            assert len(bodies) == 3
            assert bodies[1]["stream_options"] == {"include_usage": True}
            assert "stream_options" not in bodies[2]
            parsed = parse_sse_text(output.getvalue())
            assert sum(event.event == "message_stop" for event in parsed) == 1
        assert "already visible" in output.getvalue()
        assert endpoint.calls == [False] * len(bodies)
        assert all(wire.close_calls == 1 for wire in wires)
