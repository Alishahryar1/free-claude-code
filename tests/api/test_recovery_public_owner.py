"""Attempt completion and public publication share one response lifecycle."""

import gc
from copy import deepcopy
from weakref import ref

import pytest

from free_claude_code.api.stream_delivery import PublicResponseStream
from free_claude_code.application.response_collection import WireProtocol
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    current_stream_delivery,
)
from free_claude_code.core.stream_output import completed_output
from free_claude_code.providers.request_correction import (
    Correction,
    RequestCorrector,
)
from free_claude_code.providers.request_correction.rejection import Rejection
from tests.api.test_delivered_stream_recovery import public_text, text_events
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_tool_call_buffer import _call, _frames, _response, _start
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events
from tests.wire_payload import (
    array_payload,
    object_array,
    object_payload,
    string_payload,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consume", [False, True], ids=["before-first-packet", "after-commit"]
)
async def test_retained_prefix_does_not_keep_stream_alive(consume: bool) -> None:
    async def source():
        for event in text_events("responses", "Hello"):
            yield completed_output(_frames("responses", [(event["type"], event)]))

    response = await _response("responses", source())
    stream = response.body_iterator
    assert isinstance(stream, PublicResponseStream)
    state = stream.state
    if consume:
        async for _ in stream:
            pass

    stream_ref = ref(stream)
    publication_ref = ref(stream._publication)
    decoder_ref = ref(stream._publication._decoder)
    del stream, response
    gc.collect()

    assert stream_ref() is None
    assert publication_ref() is None
    assert decoder_ref() is None
    prefix = state.prefix
    assert prefix is not None
    assert prefix.text == ("Hello" if consume else "")


@pytest.mark.asyncio
async def test_completion_proof_survives_replay_hold_until_later_text() -> None:
    observed = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        for value in text_events("responses", "Hello ", complete=False):
            yield completed_output(_frames("responses", [(value["type"], value)]))
        state.begin_continuation()
        state.begin_attempt()
        for value in text_events("responses", "He", complete=False):
            yield completed_output(_frames("responses", [(value["type"], value)]))
        yield completed_output(_frames("responses", _call("responses", index=1)))
        assert state.prefix is not None and not state.prefix.has_calls
        value = text_events("responses", "llo world.", complete=False)[-1]
        yield completed_output(_frames("responses", [(value["type"], value)]))
        assert state.prefix is not None
        observed.append(state.prefix.has_calls)

    response = await _response("responses", source())
    try:
        async for _ in response.body_iterator:
            pass
    finally:
        await response.aclose()
    assert observed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("preceding_text", [False, True])
@pytest.mark.parametrize("snapshot_only", [False, True])
async def test_continuation_preserves_call_completed_by_terminal_snapshot(
    preceding_text: bool, snapshot_only: bool
) -> None:
    def reply(bodies: list[JsonObject]):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        tool = deepcopy(tool_events("responses", '{"path":"file"}'))
        if preceding_text:
            text = text_events("responses", "world.")
            for event in tool:
                if "output_index" in event:
                    event["output_index"] = 1
            array_payload(object_payload(tool[-1]["response"])["output"]).insert(
                0, text[-2]["item"]
            )
            start = text[:-1]
        else:
            start = tool[:1]
        events = [
            *start,
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
    assert events[-1].event == "response.completed", events[-1].data
    output = object_payload(events[-1].data["response"])["output"]
    calls = [
        item
        for item in array_payload(output)
        if object_payload(item)["type"] == "function_call"
    ]
    assert len(calls) == 1
    assert object_payload(calls[0])["call_id"] == "call_probe"
    assert object_payload(calls[0])["arguments"] == '{"path":"file"}'
    assert object_payload(calls[0])["status"] == "completed"
    assert public_text(events, "responses") == "Hello " + (
        "world." if preceding_text else ""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["metadata", "withheld_tools", "text"])
async def test_body_correction_depends_on_current_attempt_publication(
    activity: str,
) -> None:
    visible = activity == "text"

    def reply(bodies: list[JsonObject]):
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

    history: list[JsonObject] = [
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
async def test_released_tool_packet_is_complete_before_upstream_resumes() -> None:
    resumed = []
    states = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        states.append(state)
        yield completed_output(("".join(_frames("responses", [_start("responses")])),))
        yield completed_output(("".join(_frames("responses", _call("responses"))),))
        resumed.append(state.content_released)

    response = await _response("responses", source())
    try:
        iterator = aiter(response.body_iterator)
        assert parse_sse_text(str(await anext(iterator)))[0].event == "response.created"
        packet = parse_sse_text(str(await anext(iterator)))
        assert packet[0].event == "response.output_item.added"
        assert packet[-1].event == "response.output_item.done"
        assert states[0].prefix is not None and states[0].prefix.has_calls
        assert resumed == []
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        assert resumed == [True]
    finally:
        await response.aclose()


@pytest.mark.asyncio
async def test_public_transform_failure_preserves_returned_packets_and_emits_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ContinuationStream.prepare
    closed = []
    delivery: StreamDeliveryState | None = None
    observed_prefixes: list[str] = []

    def prepare(self: ContinuationStream, payload: JsonObject):
        if payload["type"] == "response.output_text.done":
            assert delivery is not None
            prefix = delivery.prefix
            assert prefix is not None
            observed_prefixes.append(prefix.text)
            raise ValueError("public transform failed")
        return original(self, object_payload(payload))

    monkeypatch.setattr(ContinuationStream, "prepare", prepare)

    async def source():
        nonlocal delivery
        delivery = current_stream_delivery()
        assert delivery is not None
        try:
            for event in text_events("responses", "Hello ", complete=False):
                yield completed_output(
                    (_frames("responses", [(event["type"], event)])[0],)
                )
            delivery.begin_continuation()
            yield completed_output(
                (
                    "".join(
                        _frames(
                            "responses",
                            [
                                (event["type"], event)
                                for event in text_events("responses", "world.")
                            ],
                        )
                    ),
                )
            )
        finally:
            closed.append(True)

    raw = await delivered(source(), "responses")
    events = parse_sse_text(raw)
    assert public_text(events, "responses") == "Hello "
    assert events[-1].event == "response.failed"
    assert observed_prefixes == ["Hello "]
    assert sum(e.event == "response.failed" for e in events) == 1
    assert not any(e.event == "response.completed" for e in events)
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Hello world.", "world.", "Hello ", ""])
async def test_terminal_only_text_uses_the_same_overlap_and_progress_rule(
    text: str,
) -> None:
    def reply(bodies: list[JsonObject]):
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
        output = object_payload(events[-1].data["response"])["output"]
        assert (
            "".join(
                string_payload(part["text"])
                for item in array_payload(output)
                for part in object_array(object_payload(item).get("content", []))
            )
            == "Hello world."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["reasoning", "web_search_call", "message"])
async def test_final_only_native_state_is_checked_before_continuation_publication(
    kind: str,
) -> None:
    def reply(bodies: list[JsonObject]):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        events = text_events("responses", "")
        item: JsonObject = {
            "id": "native_unreplayable",
            "type": kind,
            "status": "completed",
        }
        if kind == "reasoning":
            item.update(encrypted_content="opaque", summary=[])
        elif kind == "message":
            item.update(role="assistant", content=[{"type": "future_native_part"}])
        object_payload(events[-1]["response"])["output"] = [item]
        return 200, [events[0], events[-1]]

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    assert len(bodies) == 2
    assert parse_sse_text(raw)[-1].event == "response.failed"
    assert "native_unreplayable" not in raw


@pytest.mark.asyncio
async def test_filtered_snapshot_does_not_change_overlap_target_identity() -> None:
    def reply(bodies: list[JsonObject]):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        text = text_events("responses", "Hello world.")
        for event in text:
            if "output_index" in event:
                event["output_index"] = 1
        call = _call("responses", 0)[0]
        terminal = text[-1]
        array_payload(object_payload(terminal["response"])["output"]).insert(
            0, call[1]["item"]
        )
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
        object_array(object_payload(item)["content"])[0]["text"]
        for item in array_payload(object_payload(events[-1].data["response"])["output"])
    ] == ["Hello ", "world."]


@pytest.mark.asyncio
async def test_attempt_resets_are_atomic_and_old_closures_do_not_commit_new_attempt() -> (
    None
):
    attempts = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        for event in text_events("responses", "Hello ", complete=False):
            yield completed_output((_frames("responses", [(event["type"], event)])[0],))
        state.begin_continuation()
        state.begin_attempt()
        yield completed_output((_frames("responses", [_start("responses")])[0],))
        attempts.append(state.attempt_content_released)

        yield completed_output(('event: response.output_item.added\ndata: {"type":',))
        attempts.append(state.attempt_content_released)
        # Several invisible failures can occur in one read of the provider iterator.
        state.begin_attempt()
        state.begin_attempt()
        for event in text_events("responses", "world."):
            yield completed_output((_frames("responses", [(event["type"], event)])[0],))
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
    monkeypatch: pytest.MonkeyPatch, protocol: str, wire: WireProtocol, visible: bool
) -> None:
    marker = "injected request correction"
    proposals = []

    def correction(self: object, rejection: Rejection):
        if marker not in str(rejection.error):
            return None
        proposals.append(True)
        return Correction({**deepcopy(rejection.body), "temperature": 0}, "injected")

    monkeypatch.setattr(RequestCorrector, "correct", correction)

    def reply(bodies: list[JsonObject]):
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
