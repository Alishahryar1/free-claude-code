"""One history policy applies to initial requests, continuation, and fallback."""

import asyncio
from copy import deepcopy

import pytest

from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.history_replay import readable_reasoning
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered, executed
from tests.api.test_initial_stream_holdback import GatedBody, encode
from tests.api.test_midstream_model_fallback import delivered_candidates
from tests.api.test_recovery_reasoning import followed_by_text, reasoning_events
from tests.api.test_response_streams import _serve
from tests.api.test_tool_call_buffer import _response
from tests.providers.test_history_transports import _harness


def history():
    return [
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Old thought"},
                {"type": "text", "text": "Old answer"},
            ],
        },
        {"role": "user", "content": "Current question"},
    ]


def interrupted():
    return followed_by_text(reasoning_events(field="content", closed=False), "Hello ")


def assert_modes(bodies):
    assert [
        ("Old thought" in str(body), "Find 17." in str(body)) for body in bodies
    ] == [(True, True), (False, True), (False, False)]
    for body in bodies:
        rows = body.get("input", body.get("messages"))
        assert "Find 17." not in str(rows[-1])


@pytest.mark.asyncio
async def test_continuation_negotiates_all_modes_then_next_request_uses_cache():
    def reply(bodies):
        if len(bodies) == 1:
            return 200, interrupted()
        if len(bodies) in {2, 3}:
            return 400, {"message": "Rejected"}
        return 200, text_events("responses", "world.")

    original = history()
    before = deepcopy(original)
    async with _harness("responses", reply, max_attempts=4) as (send, bodies, _):
        output = await delivered(send("messages", original), "messages")
        assert len(bodies) == 4
        assert_modes(bodies[1:])
        assert_completed(parse_sse_text(output), "messages")
        assert public_text(parse_sse_text(output), "messages") == "Hello world."
        await delivered(send("messages", original), "messages")
        assert len(bodies) == 5
        assert "Old thought" not in str(bodies[-1])
    assert original == before


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["chat", "messages", "responses"])
async def test_fallback_restarts_modes_from_original_history_and_delivered_prefix(
    target,
):
    def first_reply(bodies):
        if len(bodies) == 1:
            return 200, interrupted()
        if len(bodies) in {2, 3}:
            return 400, {"message": "Rejected"}
        return 200, text_events("responses", "world ", complete=False)

    def second_reply(bodies):
        if len(bodies) < 3:
            return 400, {"message": "Rejected"}
        return 200, text_events(target, "again.")

    original = history()
    before = deepcopy(original)
    async with (
        _harness("responses", first_reply, max_attempts=4) as (first, first_bodies, _),
        _harness(target, second_reply, max_attempts=3) as (second, second_bodies, _),
    ):
        output = await delivered_candidates(
            [first, second], "messages", history=original
        )
    assert len(first_bodies) == 4 and len(second_bodies) == 3
    assert_modes(first_bodies[1:])
    assert_modes(second_bodies)
    assert original == before
    events = parse_sse_text(output)
    assert_completed(events, "messages")
    assert public_text(events, "messages") == "Hello world again."


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("field", ["content", "summary"])
async def test_repeated_thinking_alone_does_not_complete_a_continuation(wire, field):
    async with _harness(
        "responses",
        lambda bodies: (200, reasoning_events(field=field, closed=len(bodies) > 1)),
        max_attempts=2,
    ) as (send, bodies, _):
        output = await delivered(
            send(wire, [{"role": "user", "content": "Think"}]), wire
        )
    assert len(bodies) == 2
    events = parse_sse_text(output)
    assert events[-1].event in {"error", "response.failed"}
    assert "without adding output" in output


def completed_thinking_and_text(field, suffix=""):
    reasoning = reasoning_events(field=field)
    for event in reasoning:
        if isinstance(event.get("delta"), str):
            event["delta"] = "Find 17." + suffix
        items = (
            [event["item"]]
            if "item" in event
            else event.get("response", {}).get("output", [])
        )
        for item in items:
            for part in item.get(field) or []:
                part["text"] = "Find 17." + suffix
    text = text_events("responses", "world.")
    for event in text:
        if "output_index" in event:
            event["output_index"] = 1
    text[-1]["response"]["output"].insert(0, reasoning[-1]["response"]["output"][0])
    return [*reasoning[:-1], *text[1:]]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("field", ["content", "summary"])
async def test_matched_thinking_releases_answer_before_upstream_completes(wire, field):
    first = followed_by_text(reasoning_events(field=field, closed=False), "Hello ")
    second = completed_thinking_and_text(field)
    gated = GatedBody(
        encode("responses", second[:-1]), encode("responses", second[-1:])
    )
    visible = asyncio.Event()
    sent = []

    async def capture(message):
        data = message.get("body", b"")
        sent.append(data)
        if b"world." in data:
            visible.set()

    async with _harness(
        "responses",
        lambda bodies: (200, first if len(bodies) == 1 else gated),
        max_attempts=2,
    ) as (send, bodies, _):

        async def serve():
            source = send(wire, [{"role": "user", "content": "hello"}])
            response = await _response(wire, executed(source, wire))
            await _serve(response, send=capture)

        serving = asyncio.create_task(serve())
        wait_visible = asyncio.create_task(visible.wait())
        try:
            await asyncio.wait_for(gated.waiting.wait(), 2)
            done, _ = await asyncio.wait({wait_visible, serving}, timeout=1)
            assert wait_visible in done, "Answer is still waiting for upstream"
            assert not gated.release.is_set()
            assert not serving.done()
            assert len(bodies) == 2
        finally:
            gated.release.set()
            wait_visible.cancel()
            await asyncio.gather(wait_visible, return_exceptions=True)
            await asyncio.wait_for(serving, 2)

    events = parse_sse_text(b"".join(sent).decode())
    assert public_text(events, wire) == "Hello world."
    assert_completed(events, wire)
    assert gated.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("field", ["content", "summary"])
async def test_second_cutoff_retains_answer_after_matched_thinking(wire, field):
    first = followed_by_text(reasoning_events(field=field, closed=False), "Hello ")
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else completed_thinking_and_text(field)[:-1],
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "hello"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event in {"error", "response.failed"}
    assert public_text(events, wire) == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("field", ["content", "summary"])
@pytest.mark.parametrize("suffix", ["", " Act"])
async def test_followup_does_not_restore_thinking_removed_by_overlap(
    wire, field, suffix
):
    first = followed_by_text(reasoning_events(field=field, closed=False), "Hello ")

    def reply(bodies):
        return 200, (
            first
            if len(bodies) == 1
            else completed_thinking_and_text(field, suffix)
            if len(bodies) == 2
            else text_events("responses", "Next")
        )

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "Think"}]), wire)
        events = parse_sse_text(raw)
        assert_completed(events, wire)
        if wire == "messages":

            async def stream():
                yield raw

            message, error, stopped = await aggregate_anthropic_sse_to_message(stream())
            assert error is None and stopped
            visible = "".join(block.get("thinking", "") for block in message["content"])
            saved = [{"role": "assistant", "content": deepcopy(message["content"])}]
        else:
            saved = events[-1].data["response"]["output"]
            visible = "".join(
                text
                for item in saved
                if item.get("type") == "reasoning"
                for text, _ in readable_reasoning(item)
            )
        assert visible == "Find 17." + suffix
        await delivered(send(wire, [*saved, {"role": "user", "content": "Next"}]), wire)
    replayed = "".join(
        text
        for item in bodies[-1]["input"]
        if item.get("type") == "reasoning"
        for text, _ in readable_reasoning(item)
    )
    assert replayed == visible
