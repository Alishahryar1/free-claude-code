"""Attempt completion and public publication share one response lifecycle."""

from copy import deepcopy

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.providers.request_recovery import RequestCorrections
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import public_text, text_events
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_tool_call_buffer import _call, _frames, _response, _start
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def immediate_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preceding_text", [False, True])
@pytest.mark.parametrize("snapshot_only", [False, True])
async def test_continuation_preserves_call_completed_by_terminal_snapshot(
    preceding_text, snapshot_only
):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        tool = deepcopy(tool_events("responses", '{"path":"file"}'))
        if preceding_text:
            text = text_events("responses", "world.")
            for event in tool:
                if "output_index" in event:
                    event["output_index"] = 1
            tool[-1]["response"]["output"].insert(0, text[-2]["item"])
        events = [
            *(text[:-1] if preceding_text else tool[:1]),
            *(
                [e for e in tool[1:-1] if e["type"] != "response.output_item.done"]
                if not snapshot_only
                else []
            ),
            tool[-1],
        ]
        return 200, [{**event, "sequence_number": i} for i, event in enumerate(events)]

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send(
                "responses",
                [{"role": "user", "content": "greet and read"}],
                tools=_tools("responses"),
            ),
            "responses",
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == "response.completed"
    output = events[-1].data["response"]["output"]
    calls = [item for item in output if item["type"] == "function_call"]
    assert len(calls) == 1
    assert calls[0]["call_id"] == "call_probe"
    assert calls[0]["arguments"] == '{"path":"file"}'
    assert calls[0]["status"] == "completed"
    assert public_text(events, "responses") == "Hello " + (
        "world." if preceding_text else ""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["metadata", "withheld_tools", "text"])
async def test_body_correction_depends_on_current_attempt_publication(activity):
    visible = activity == "text"

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        if len(bodies) == 2:
            prefix = text_events("responses", "world ", complete=False)
            hidden = _frames(
                "responses", [_call("responses", 0)[0], _call("responses", 1)[0]]
            )
            return 200, [
                *(prefix if visible else prefix[:1]),
                *(
                    [event.data for frame in hidden for event in parse_sse_text(frame)]
                    if activity == "withheld_tools"
                    else []
                ),
                {
                    "type": "error",
                    "sequence_number": 4,
                    "code": "invalid_encrypted_content",
                    "param": "input[0].encrypted_content",
                    "message": "The encrypted content could not be verified.",
                },
            ]
        return 200, text_events("responses", "world again.")

    history = [
        {
            "type": "reasoning",
            "id": "rs_old",
            "encrypted_content": "opaque",
            "summary": [],
        },
        {"role": "user", "content": "greet"},
    ]
    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(send("responses", history), "responses")
    events = parse_sse_text(raw)
    assert len(bodies) == (2 if visible else 3)
    assert events[-1].event == ("response.failed" if visible else "response.completed")
    assert public_text(events, "responses") == (
        "Hello world " if visible else "Hello world again."
    )
    assert "call_0" not in raw and "call_1" not in raw


@pytest.mark.asyncio
async def test_released_tool_group_is_published_one_frame_at_a_time():
    resumed = []
    states = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        states.append(state)
        yield "".join(_frames("responses", [_start("responses")]))
        yield "".join(_frames("responses", _call("responses")))
        resumed.append(state.content_released)

    response = await _response("responses", source())
    try:
        iterator = aiter(response.body_iterator)
        assert parse_sse_text(str(await anext(iterator)))[0].event == "response.created"
        first = parse_sse_text(str(await anext(iterator)))
        assert len(first) == 1
        assert first[0].event == "response.output_item.added"
        assert not states[0].prefix.has_calls
        assert resumed == []
        assert len(parse_sse_text(str(await anext(iterator)))) == 1
        assert resumed == []
        await anext(iterator)
        await anext(iterator)
        assert resumed == []
        assert [part async for part in iterator] == []
        assert resumed == [True]
    finally:
        await response.aclose()


@pytest.mark.asyncio
async def test_public_transform_failure_preserves_returned_frames_and_emits_error(
    monkeypatch,
):
    original = ContinuationStream.prepare
    closed = []

    def prepare(self, payload):
        if payload["type"] == "response.output_text.done":
            raise ValueError("public transform failed")
        return original(self, payload)

    monkeypatch.setattr(ContinuationStream, "prepare", prepare)

    async def source():
        state = current_stream_delivery()
        assert state is not None
        try:
            for event in text_events("responses", "Hello ", complete=False):
                yield _frames("responses", [(event["type"], event)])[0]
            state.begin_continuation()
            yield "".join(
                _frames(
                    "responses",
                    [
                        (event["type"], event)
                        for event in text_events("responses", "world.")
                    ],
                )
            )
        finally:
            closed.append(True)

    raw = await delivered(source(), "responses")
    events = parse_sse_text(raw)
    assert public_text(events, "responses") == "Hello world."
    assert events[-1].event == "response.failed"
    assert sum(e.event == "response.failed" for e in events) == 1
    assert not any(e.event == "response.completed" for e in events)
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Hello world.", "world.", "Hello ", ""])
async def test_terminal_only_text_uses_the_same_overlap_and_progress_rule(text):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        events = text_events("responses", text)
        return 200, [events[0], events[-1]]

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == (
        "response.completed" if "world" in text else "response.failed"
    )
    if "world" in text:
        output = events[-1].data["response"]["output"]
        assert (
            "".join(part["text"] for item in output for part in item.get("content", []))
            == "Hello world."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["reasoning", "web_search_call", "message"])
async def test_final_only_native_state_is_checked_before_continuation_publication(kind):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        events = text_events("responses", "")
        item = {"id": "native_unreplayable", "type": kind, "status": "completed"}
        if kind == "reasoning":
            item.update(encrypted_content="opaque", summary=[])
        elif kind == "message":
            item.update(role="assistant", content=[{"type": "future_native_part"}])
        events[-1]["response"]["output"] = [item]
        return 200, [events[0], events[-1]]

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    assert len(bodies) == 2
    assert parse_sse_text(raw)[-1].event == "response.failed"
    assert "native_unreplayable" not in raw


@pytest.mark.asyncio
async def test_filtered_snapshot_does_not_change_overlap_target_identity():
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        text = text_events("responses", "Hello world.")
        for event in text:
            if "output_index" in event:
                event["output_index"] = 1
        call = _call("responses", 0)[0]
        terminal = text[-1]
        terminal["response"]["output"].insert(0, call[1]["item"])
        events = [*text[:-1], {"type": call[0], **call[1]}, terminal]
        return 200, [{**event, "sequence_number": i} for i, event in enumerate(events)]

    async with _harness("responses", reply, max_attempts=2) as (send, _, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert "call_0" not in raw
    assert public_text(events, "responses") == "Hello world."
    assert [
        item["content"][0]["text"] for item in events[-1].data["response"]["output"]
    ] == ["Hello ", "world."]


@pytest.mark.asyncio
async def test_attempt_resets_are_atomic_and_old_closures_do_not_commit_new_attempt():
    attempts = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        for event in text_events("responses", "Hello ", complete=False):
            yield _frames("responses", [(event["type"], event)])[0]
        state.begin_continuation()
        state.begin_attempt()
        yield _frames("responses", [_start("responses")])[0]
        attempts.append(state.attempt_content_released)
        yield "".join(_frames("responses", _call("responses")[:2]))
        yield 'event: response.output_item.added\ndata: {"type":'
        attempts.append(state.attempt_content_released)
        # Several invisible failures can occur in one read of the provider iterator.
        state.begin_attempt()
        state.begin_attempt()
        for event in text_events("responses", "world."):
            yield _frames("responses", [(event["type"], event)])[0]
        attempts.append(state.attempt_content_released)

    raw = await delivered(source(), "responses")
    events = parse_sse_text(raw)
    assert attempts == [False, False, True]
    assert public_text(events, "responses") == "Hello world."
    assert "call_0" not in raw
    assert events[-1].event == "response.completed"
    assert sum(e.event == "response.created" for e in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("visible", [False, True])
async def test_shared_correction_guard_uses_public_attempt_effects(
    monkeypatch, protocol, wire, visible
):
    marker = "injected request correction"
    proposals = []

    def correction(self, error, body, **kwargs):
        if marker not in str(error):
            return None
        proposals.append(True)
        return {**deepcopy(body), "temperature": 0}

    monkeypatch.setattr(RequestCorrections, "next_body", correction)

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events(protocol, "Hello ", complete=False)
        if len(bodies) == 2:
            events = text_events(protocol, "world " if visible else "", complete=False)
            if not visible and protocol != "chat":
                events = events[:1]
            error = {
                "type": "invalid_request_error",
                "message": marker,
                "code": "invalid_request",
            }
            events.append(
                {
                    "type": "error",
                    **{key: value for key, value in error.items() if key != "type"},
                }
                if protocol == "responses"
                else {"type": "error", "error": error}
            )
            return 200, events
        return 200, text_events(protocol, "world.")

    async with _harness(protocol, reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "greet"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == (2 if visible else 3)
    assert proposals == ([] if visible else [True])
    assert public_text(events, wire) == ("Hello world " if visible else "Hello world.")
    assert events[-1].event == (
        ("error" if wire == "messages" else "response.failed")
        if visible
        else ("message_stop" if wire == "messages" else "response.completed")
    )
