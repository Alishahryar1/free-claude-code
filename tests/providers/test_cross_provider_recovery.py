"""Recovery across real HTTP/SDK transports shares one public response."""

import json
from contextlib import AsyncExitStack

import pytest

from free_claude_code.application.recovery import RecoveryCoordinator
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.recovery_stream import MessagesRecoveryWriter
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesRecoveryWriter,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from tests.api.model_fallback_support import responses_text_stream
from tests.providers.test_anthropic_messages_transport import Endpoint, _events
from tests.providers.test_history_transports import _events_for, _harness


def text_events(protocol, text, *, complete=True):
    if protocol == "messages":
        events = _events(text)
        return events if complete else events[:3]
    if protocol == "responses":
        events = [
            event.data
            for event in parse_sse_text(
                "".join(responses_text_stream(text, model="upstream"))
            )
        ]
        return events if complete else events[:4]
    chunk = _events_for("chat")[-1]
    chunk["choices"][0].update(
        delta={"content": text}, finish_reason="stop" if complete else None
    )
    return [chunk]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "protocols",
    [
        ("chat", "responses", "messages"),
        ("responses", "messages", "chat"),
        ("messages", "chat", "responses"),
    ],
)
async def test_three_targets_resume_the_latest_prefix_with_tools_and_one_lifecycle(
    wire, protocols
):
    request = (
        MessagesRequest.model_validate(
            {
                "model": "requested",
                "messages": [{"role": "user", "content": "Continue the work"}],
                "system": "Keep these instructions",
                "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            }
        )
        if wire == "messages"
        else OpenAIResponsesRequest(
            model="requested",
            input="Continue the work",
            instructions="Keep these instructions",
            tools=[
                {
                    "type": "function",
                    "name": "read",
                    "parameters": {"type": "object"},
                    "strict": False,
                }
            ],
        )
    )
    original = request.model_dump()
    writer = (
        MessagesRecoveryWriter(model="public", input_tokens=12)
        if wire == "messages"
        else ResponsesRecoveryWriter(model="public", input_tokens=12)
    )
    providers, bodies = [], []
    selected, transitions = [], []
    async with AsyncExitStack() as resources:
        for index, protocol in enumerate(protocols):
            events = text_events(
                protocol,
                ("First. ", "First. Second. ", "Third.")[index],
                complete=index == 2,
            )
            _, sent, provider = await resources.enter_async_context(
                _harness(
                    protocol, lambda _, events=events: (200, events), key=str(index)
                )
            )
            providers.append(provider)
            bodies.append(sent)

        async def open_candidate(index, target):
            options = {
                "input_tokens": 12,
                "request_id": "matrix",
                "response_model": "public",
                "reasoning": ReasoningPolicy.provider_default(),
            }
            if protocols[index] == "messages":
                options["endpoint_context"] = Endpoint()
            return getattr(providers[index], f"open_{wire}")(
                request.model_copy(update={"model": target.provider_model}, deep=True),
                **options,
            )

        stream = RecoveryCoordinator(
            candidates=tuple(
                ProviderModelTarget(
                    f"provider{index}", f"model{index}", f"provider{index}/model{index}"
                )
                for index in range(3)
            ),
            opener=open_candidate,
            writer=writer,
            progress_timeout_seconds=5,
            timeout_failure=lambda _: ExecutionFailure(
                FailureKind.TIMEOUT, 504, "No progress", False
            ),
            request_id="matrix",
            on_selected=lambda target, index: selected.append(index),
            on_fallback=lambda old, new, error, index: transitions.append(index),
        ).stream()
        output = "".join([event async for event in stream])
    assert request.model_dump() == original
    assert [len(sent) for sent in bodies] == [1, 1, 1]
    assert selected == transitions == [1, 2]
    assert "First." in json.dumps(bodies[1][0])
    assert "First." in json.dumps(bodies[2][0]) and "Second." in json.dumps(
        bodies[2][0]
    )
    assert all(body[0]["tools"] for body in bodies)
    assert all("Keep these instructions" in json.dumps(body[0]) for body in bodies)
    assert writer.checkpoint.text == "First. Second. Third."
    events = parse_sse_text(output)
    if wire == "messages":
        assert sum(event.event == "message_start" for event in events) == 1
        assert sum(event.event == "message_stop" for event in events) == 1
        assert events[-2].data["usage"]["input_tokens"] == 12
    else:
        assert sum(event.event == "response.created" for event in events) == 1
        assert sum(event.event == "response.completed" for event in events) == 1
        assert events[-1].data["response"]["id"] == events[0].data["response"]["id"]
        assert events[-1].data["response"]["usage"]["input_tokens"] == 12
        assert [event.data["sequence_number"] for event in events] == list(
            range(len(events))
        )
    assert all(event.event not in {"error", "response.failed"} for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("stop", ["pause_turn", "refusal", "stop_sequence"])
async def test_explicit_native_stop_does_not_regenerate_an_empty_continuation(
    wire, stop
):
    async with _harness(
        "messages",
        lambda bodies: (
            200,
            _events("First")[:3] if len(bodies) == 1 else _events("", stop),
        ),
    ) as (send, bodies, _):
        output = "".join(
            [
                event
                async for event in send(wire, [{"role": "user", "content": "Continue"}])
            ]
        )
    assert len(bodies) == 2
    if wire == "responses" and stop == "pause_turn":
        # Responses has no representation for the native paused operation.
        assert output.count("event: response.failed") == 1
    else:
        assert "response.failed" not in output and "event: error" not in output
