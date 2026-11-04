"""One history policy applies to initial requests, continuation, and fallback."""

from copy import deepcopy

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_midstream_model_fallback import delivered_candidates
from tests.api.test_recovery_reasoning import followed_by_text, reasoning_events
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
