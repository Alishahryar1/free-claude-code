"""Recovery preserves content and closes each published lifecycle stage once."""

import asyncio
from collections import Counter
from copy import deepcopy
from typing import Any

import httpx2
import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered, partial_tool
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def release_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


@pytest.mark.asyncio
async def test_partial_closure_with_default_holdback(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        RecoveryHoldbackBuffer,
    )
    original_transport = httpx2.MockTransport

    class PacedStream(httpx2.AsyncByteStream):
        def __init__(self, raw):
            self.raw = raw

        async def __aiter__(self):
            first, _, rest = self.raw.partition(b"\n\n")
            yield first + b"\n\n"
            await asyncio.sleep(0.8)
            yield rest

    def transport(handler):
        async def handle(request):
            response = handler(request)
            raw = await response.aread()
            await response.aclose()
            return httpx2.Response(
                200, headers=response.headers, stream=PacedStream(raw)
            )

        return original_transport(handle)

    monkeypatch.setattr(httpx2, "MockTransport", transport)
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ")[:6]
            if len(bodies) == 1
            else text_events("responses", "world."),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert_completed(events, "responses")
    assert len(bodies) == 2
    assert "Hello " in str(bodies[1]["input"])
    assert public_text(events, "responses") == "Hello world."
    assert (
        sum(
            event.event == "response.content_part.done"
            and event.data["item_id"] == "msg_text"
            for event in events
        )
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("Hello world.", "Hello world."),
        ("lo world.", "Hello world."),
        ("Hel", "Hello Hel"),
    ],
)
async def test_reasoning_before_continuation_text_keeps_overlap_filter(
    wire, candidate, expected
):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("chat", "Hello ", complete=False)
        reasoning = text_events("chat", "", complete=False)[0]
        reasoning["choices"][0]["delta"] = {
            "reasoning_content": "Continue the greeting."
        }
        return 200, [reasoning, *text_events("chat", candidate)]

    async with _harness("chat", reply, max_attempts=2) as (send, _, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "greet"}]), wire)
    events = parse_sse_text(raw)
    assert_completed(events, wire)
    assert public_text(events, wire) == expected
    assert "Continue the greeting." in raw


def refusal_events(*, mixed=False):
    events = text_events("responses", "lo world." if mixed else "")
    part = {"type": "refusal", "refusal": "I cannot continue that request."}
    position = 1 if mixed else 0
    prefix = events[:-2] if mixed else events[:2]
    refusal: list[dict[str, Any]] = [
        {
            "type": "response.content_part.added",
            "part": {"type": "refusal", "refusal": ""},
        },
        {"type": "response.refusal.delta", "delta": part["refusal"]},
        {"type": "response.refusal.done", "refusal": part["refusal"]},
        {"type": "response.content_part.done", "part": part},
    ]
    for event in refusal:
        event.update(output_index=0, content_index=position, item_id="msg_text")
    item = events[-2]["item"]
    item["content"] = [*item["content"], part] if mixed else [part]
    item["phase"] = "final_answer"
    return [
        {**event, "sequence_number": index}
        for index, event in enumerate([*prefix, *refusal, *events[-2:]])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("limited", [False, True])
async def test_continuation_preserves_refusal_and_completed_metadata(
    monkeypatch, mixed, limited
):
    if limited:
        monkeypatch.setattr(
            "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 600
        )
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ", complete=False)
            if len(bodies) == 1
            else refusal_events(mixed=mixed),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert_completed(events, "responses")
    assert len(bodies) == 2
    output = events[-1].data["response"]["output"]
    assert output == [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ]
    assert output[-1]["phase"] == "final_answer"
    assert output[-1]["content"][-1] == {
        "type": "refusal",
        "refusal": "I cannot continue that request.",
    }
    assert public_text(events, "responses") == ("Hello world." if mixed else "Hello ")


@pytest.mark.asyncio
async def test_published_refusal_interruption_does_not_generate_again():
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ", complete=False)
            if len(bodies) == 1
            else refusal_events()[:-3],
        ),
        max_attempts=3,
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == "response.failed"
    assert (
        events[-1].data["response"]["output"][-1]["content"][-1]["refusal"]
        == "I cannot continue that request."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [4, 5, 6, 7])
async def test_recovery_emits_only_missing_text_completion_stages(cut):
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ")[:cut]
            if len(bodies) == 1
            else text_events("responses", "world."),
        ),
        max_attempts=2,
    ) as (send, _, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert_completed(events, "responses")
    counts = Counter(
        event.event
        for event in events
        if event.data.get("item_id", event.data.get("item", {}).get("id")) == "msg_text"
    )
    for kind in (
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ):
        assert counts[kind] == 1
    assert public_text(events, "responses") == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("continued", [False, True])
@pytest.mark.parametrize("cut", [4, 5, 6])
async def test_handoff_snapshot_matches_published_closures(
    monkeypatch, limited, continued, cut
):
    if limited:
        original = DeliveredResponse.closing_events

        def cap_during_boundary(self):
            events = original(self)
            if self.snapshot().has_calls:
                monkeypatch.setattr(
                    "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES",
                    self._bytes + 1,
                )
            return events

        monkeypatch.setattr(DeliveredResponse, "closing_events", cap_during_boundary)

    def reply(bodies):
        if continued and len(bodies) == 1:
            return 200, text_events("responses", "Before. ", complete=False)
        events = deepcopy(tool_events("responses", '{"path":"kept"}')[:-1])
        for event in text_events("responses", "Next ")[1:cut]:
            event["output_index"] = 1
            events.append(event)
        for event in partial_tool("responses")[1:]:
            if "output_index" in event:
                event["output_index"] = 2
            events.append(event)
        return 200, [
            {**event, "sequence_number": index} for index, event in enumerate(events)
        ]

    async with _harness("responses", reply, max_attempts=2 if continued else 1) as (
        send,
        bodies,
        _,
    ):
        raw = await delivered(
            send(
                "responses",
                [{"role": "user", "content": "read"}],
                tools=_tools("responses"),
            ),
            "responses",
        )
    events = parse_sse_text(raw)
    assert_completed(events, "responses")
    assert len(bodies) == (2 if continued else 1)
    assert "call_abandoned" not in raw
    output = events[-1].data["response"]["output"]
    assert output == [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ]
    assert all(item["status"] == "completed" for item in output)
    assert [
        item["arguments"] for item in output if item["type"] == "function_call"
    ] == ['{"path":"kept"}']
