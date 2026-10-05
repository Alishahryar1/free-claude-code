"""Recovery must not fabricate structured items or misalign token metadata."""

import json
from copy import deepcopy

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_anthropic_messages_transport import Endpoint, _sse
from tests.providers.test_anthropic_provider import native_body, provider
from tests.providers.test_history_transports import _events_for, _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def release_provider_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def output_format(kind):
    if kind != "json_schema":
        return {"type": kind}
    return {
        "type": kind,
        "name": "Answer",
        "schema": Answer.model_json_schema(),
        "strict": True,
    }


def responses_stream(transport, protocol, **fields):
    options = {
        "response_model": "public-alias",
        "request_id": "text-contract",
        "reasoning": ReasoningPolicy.provider_default(),
    }
    if protocol == "messages":
        options["endpoint_context"] = Endpoint()
    else:
        options["input_tokens"] = 0
    return transport.stream_responses(
        OpenAIResponsesRequest(model="requested", input="Return value 1", **fields),
        **options,
    )


async def consume_structured(raw, *, failed):
    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=raw.encode()
            )
        )
    )
    async with (
        AsyncOpenAI(
            api_key="fixture",
            base_url="https://consumer.invalid",
            max_retries=0,
            http_client=http,
        ) as client,
        client.responses.stream(
            model="fixture", input="Return value 1", text_format=Answer
        ) as stream,
    ):
        if failed:
            with pytest.raises(RuntimeError, match=r"response\.completed"):
                await stream.get_final_response()
        else:
            response = await stream.get_final_response()
            assert response.output_parsed == Answer(value=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol,kind",
    [
        ("responses", "json_schema"),
        ("responses", "json_object"),
        ("chat", "json_schema"),
        ("chat", "json_object"),
        ("messages", "json_schema"),
    ],
)
@pytest.mark.parametrize("stage", ["healthy", "hidden_retry", "delivered_failure"])
async def test_structured_request_keeps_healthy_output_and_failure_policy(
    protocol, kind, stage
):
    def reply(bodies):
        if len(bodies) == 1 and stage == "hidden_retry":
            return 200, []
        interrupted = len(bodies) == 1 and stage == "delivered_failure"
        # No echoed format: the request itself must protect this contract.
        return 200, text_events(
            protocol,
            '{"value":' if interrupted else '{"value":1}',
            complete=not interrupted,
        )

    async with _harness(protocol, reply, max_attempts=3) as (_, bodies, transport):
        raw = await delivered(
            responses_stream(transport, protocol, text={"format": output_format(kind)}),
            "responses",
        )
    events = parse_sse_text(raw)
    failed = stage == "delivered_failure"
    assert len(bodies) == (2 if stage == "hidden_retry" else 1)
    assert public_text(events, "responses") == (
        '{"value":' if failed else '{"value":1}'
    )
    if failed:
        assert events[-1].event == "response.failed"
        assert not any(event.event == "response.completed" for event in events)
        if protocol != "chat":
            assert not any(
                event.event == "response.output_text.done" for event in events
            )
    else:
        assert_completed(events, "responses")
    # Chat's existing failure writer closes partial text before reporting failure.
    # Its typed-JSON parsing limitation predates delivered-output recovery.
    if not failed or protocol != "chat":
        await consume_structured(raw, failed=failed)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_explicit_plain_text_format_can_continue(protocol):
    async with _harness(
        protocol,
        lambda bodies: (
            200,
            text_events(
                protocol,
                "Hello " if len(bodies) == 1 else "world.",
                complete=len(bodies) > 1,
            ),
        ),
    ) as (_, bodies, transport):
        raw = await delivered(
            responses_stream(transport, protocol, text={"format": {"type": "text"}}),
            "responses",
        )
    assert len(bodies) == 2
    assert_completed(parse_sse_text(raw), "responses")
    assert public_text(parse_sse_text(raw), "responses") == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["native_messages", "translated_messages", "chat_extra"]
)
async def test_messages_structured_request_does_not_resume_partial_json(path):
    bodies = []
    body = {
        "model": "requested",
        "messages": [{"role": "user", "content": "Return value 1"}],
        "max_tokens": 64,
    }
    schema = {"type": "json_schema", "schema": Answer.model_json_schema()}
    if path == "chat_extra":
        body["extra_body"] = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    key: value
                    for key, value in output_format("json_schema").items()
                    if key != "type"
                },
            }
        }
    else:
        body["output_config"] = {"format": schema}
    if path == "native_messages":

        def reply(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(*text_events("messages", '{"value":', complete=False)),
            )

        p = provider(reply)
        try:
            raw = await delivered(
                p.stream_native_messages(
                    NativeMessagesRequest({**native_body(True), **body})
                ),
                "messages",
            )
        finally:
            await p.cleanup()
    else:
        protocol = "chat" if path == "chat_extra" else "messages"
        async with _harness(
            protocol,
            lambda _: (200, text_events(protocol, '{"value":', complete=False)),
            max_attempts=2,
        ) as (_, bodies, transport):
            options = {
                "response_model": "public-alias",
                "request_id": "text-contract",
                "reasoning": ReasoningPolicy.provider_default(),
            }
            if protocol == "messages":
                options["endpoint_context"] = Endpoint()
            else:
                options["input_tokens"] = 0
            raw = await delivered(
                transport.stream_messages(
                    MessagesRequest.model_validate(body), **options
                ),
                "messages",
            )
    assert len(bodies) == 1
    events = parse_sse_text(raw)
    assert public_text(events, "messages") == '{"value":'
    assert events[-1].event == "error"


def scored_events(text, location, *, value=True):
    events = [deepcopy(event) for event in text_events("responses", text)]
    scores = (
        [
            {
                "token": text,
                "bytes": list(text.encode()),
                "logprob": -0.5,
                "top_logprobs": [],
            }
        ]
        if value is True
        else value
    )
    targets = {
        "delta": events[3],
        "text_done": events[4],
        "part": events[5]["part"],
        "item": events[6]["item"]["content"][0],
        "terminal": events[7]["response"]["output"][0]["content"][0],
    }
    for key, target in targets.items():
        if location == "all" or location == key:
            target["logprobs"] = deepcopy(scores)
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "continuing,location",
    [(False, location) for location in ("delta", "text_done", "part", "item")]
    + [
        (True, location)
        for location in ("delta", "text_done", "part", "item", "terminal")
    ],
)
async def test_preserved_logprobs_prevent_text_reconstruction(location, continuing):
    def reply(bodies):
        if continuing and len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        events = scored_events("Hello world." if continuing else "Hello ", location)
        if not continuing:
            events = events[
                : {"delta": 4, "text_done": 5, "part": 6, "item": 7}[location]
            ]
        return 200, events

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == (2 if continuing else 1)
    assert events[-1].event == "response.failed"
    assert public_text(events, "responses").startswith("Hello ")
    if not continuing:
        assert public_text(events, "responses") == "Hello "
    else:
        rejected_kind = {
            "delta": "response.output_text.delta",
            "text_done": "response.output_text.done",
            "part": "response.content_part.done",
            "item": "response.output_item.done",
            "terminal": "response.completed",
        }[location]
        assert not any(
            event.event == rejected_kind and event.data.get("output_index", 1) == 1
            for event in events
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, []])
async def test_empty_logprobs_do_not_block_recovery(value):
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ", complete=False)
            if len(bodies) == 1
            else scored_events("Hello world.", "all", value=value),
        ),
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
    assert len(bodies) == 2
    assert_completed(parse_sse_text(raw), "responses")
    assert public_text(parse_sse_text(raw), "responses") == "Hello world."


@pytest.mark.asyncio
async def test_healthy_logprob_payloads_are_preserved():
    source = scored_events("Hello world.", "all")
    async with _harness("responses", lambda _: (200, source)) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 1
    assert_completed(events, "responses")
    by_kind = {event.event: event.data for event in events}
    assert by_kind["response.output_text.delta"]["logprobs"] == source[3]["logprobs"]
    assert events[-1].data["response"]["output"] == source[-1]["response"]["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("continuing", [False, True])
async def test_echoed_structured_format_prevents_reconstruction(continuing):
    def reply(bodies):
        events = text_events(
            "responses",
            "Hello " if len(bodies) == 1 else "world.",
            complete=len(bodies) > 1,
        )
        if not continuing or len(bodies) > 1:
            events[0]["response"]["text"] = {"format": output_format("json_schema")}
        return 200, events

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
    assert len(bodies) == (2 if continuing else 1)
    assert parse_sse_text(raw)[-1].event == "response.failed"
    assert public_text(parse_sse_text(raw), "responses") == "Hello "


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", ["structured", "logprobs"])
@pytest.mark.parametrize("later_reasoning", [False, True])
async def test_complete_calls_do_not_bypass_unsupported_text_guard(
    constraint, later_reasoning
):
    prefix = text_events("responses", '{"value":', complete=False)
    if constraint == "logprobs":
        prefix = scored_events("Hello ", "delta")[:4]
    if later_reasoning:
        reasoning = deepcopy(_events_for("responses")[1:-1])
        for event in reasoning:
            event["output_index"] = 1
        prefix += reasoning
    calls = deepcopy(tool_events("responses", '{"path":"file"}')[1:-1])
    for event in calls:
        event["output_index"] += 2 if later_reasoning else 1
    async with _harness(
        "responses", lambda _: (200, prefix + calls), max_attempts=1
    ) as (_, bodies, transport):
        fields = {"tools": _tools("responses")}
        if constraint == "structured":
            fields["text"] = {"format": output_format("json_schema")}
        raw = await delivered(
            responses_stream(transport, "responses", **fields), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 1
    assert events[-1].event == "response.failed"
    assert not any(event.event == "response.output_text.done" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol,wire",
    [("responses", "messages"), ("chat", "messages"), ("chat", "responses")],
)
async def test_metadata_already_omitted_by_projection_does_not_block_recovery(
    protocol, wire
):
    def reply(bodies):
        if protocol == "responses":
            events = scored_events("Hello " if len(bodies) == 1 else "world.", "all")
            return 200, events[:4] if len(bodies) == 1 else events
        events = text_events(
            "chat", "Hello " if len(bodies) == 1 else "world.", complete=len(bodies) > 1
        )
        text = events[0]["choices"][0]["delta"]["content"]
        events[0]["choices"][0]["logprobs"] = {
            "content": [
                {
                    "token": text,
                    "logprob": -0.5,
                    "bytes": list(text.encode()),
                    "top_logprobs": [],
                }
            ]
        }
        return 200, events

    async with _harness(protocol, reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "hello"}]), wire)
    assert len(bodies) == 2
    assert_completed(parse_sse_text(raw), wire)
    assert public_text(parse_sse_text(raw), wire) == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", ["structured", "logprobs"])
async def test_failed_snapshot_keeps_its_error_category_with_text_metadata(constraint):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        failed = scored_events("unused", "terminal")[-1]
        if constraint == "structured":
            failed["response"]["output"] = []
            failed["response"]["text"] = {"format": output_format("json_schema")}
        failed["type"] = "response.failed"
        failed["response"]["status"] = "failed"
        failed["response"]["error"] = {
            "code": "context_length_exceeded",
            "message": "The context window is full.",
        }
        return 200, [failed]

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
    assert len(bodies) == 2
    events = parse_sse_text(raw)
    assert events[-1].event == "response.failed"
    assert events[-1].data["response"]["error"]["code"] == "context_length_exceeded"
    assert public_text(events, "responses") == "Hello "
