"""Only the final candidate failure reaches the public Responses lifecycle."""

import asyncio
from copy import deepcopy
from itertools import pairwise

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    after_text,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import partial_tool
from tests.api.test_midstream_model_fallback import delivered_candidates
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def release_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def failed_response(text, label):
    events = text_events("responses", text)
    for event in events:
        if "response" in event:
            event["response"]["id"] = f"resp_{label}"
    final = events[-1]
    final["type"] = "response.failed"
    final["extension"] = {"candidate": label}
    final["response"].update(
        status="failed",
        created_at=12.0,
        completed_at=34.0,
        metadata={"candidate": label},
        error={
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": f"{label} rejected continuation",
            "param": None,
            "provider_extension": {"keep": True},
        },
    )
    return events


@pytest.mark.asyncio
async def test_final_native_failure_preserves_projected_error_output_usage_and_extensions():
    source = failed_response("Partial text", "only")
    expected = deepcopy(source[-1])
    async with _harness("responses", lambda _: (200, source), max_attempts=1) as (
        send,
        bodies,
        _,
    ):
        raw = await delivered_candidates([send], "responses")
    assert len(bodies) == 1
    events = parse_sse_text(raw)
    final = events[-1].data
    assert events[-1].event == "response.failed"
    assert sum(event.event == "response.failed" for event in events) == 1
    assert final["extension"] == expected["extension"]
    for field in ("id", "output", "usage", "error", "metadata", "status"):
        assert final["response"][field] == expected["response"][field]
    assert final["response"]["model"] == "public-alias"
    assert type(final["response"]["created_at"]) is int
    assert type(final["response"]["completed_at"]) is int
    assert source[-1] == expected


@pytest.mark.asyncio
async def test_final_native_snapshot_cannot_release_a_withheld_call():
    source = after_text("responses", partial_tool("responses"))
    final = failed_response("Before. ", "only")[-1]
    call = tool_events("responses", '{"path":"unreleased"}')[-1]["response"]["output"][
        0
    ]
    call["call_id"] = "call_abandoned"
    final["response"]["output"].append(call)
    final["sequence_number"] = len(source)
    source.append(final)
    async with _harness("responses", lambda _: (200, source), max_attempts=1) as (
        send,
        _,
        _,
    ):
        raw = await delivered_candidates([send], "responses", tools=_tools("responses"))
    events = parse_sse_text(raw)
    assert "call_abandoned" not in raw and "unreleased" not in raw
    assert events[-1].data["response"]["error"] == final["response"]["error"]
    assert [item["type"] for item in events[-1].data["response"]["output"]] == [
        "message"
    ]
    assert public_text(events, "responses") == "Before. "


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
async def test_zero_call_final_candidate_keeps_published_response_identity(protocol):
    admission = ProviderAdmissionController(
        provider_name="cooled",
        rate_limit=1000,
        max_attempts=1,
        base_delay=60,
        max_delay=60,
        jitter=0,
    )
    async with (
        _harness(
            "responses",
            lambda _: (200, text_events("responses", "")[:1]),
            max_attempts=1,
        ) as (first, first_bodies, _),
        _harness(
            protocol,
            lambda _: (503, {"type": "server_error", "message": "Unavailable"}),
            max_attempts=1,
        ) as (last, last_bodies, provider),
    ):
        transport = provider._chat if protocol == "chat" else provider
        transport._admission = admission
        warmup = last("responses", [{"role": "user", "content": "earlier request"}])
        try:
            with pytest.raises(ExecutionFailure):
                await anext(warmup)
        finally:
            await warmup.aclose()
        prior_calls = len(last_bodies)
        raw = await delivered_candidates([first, last], "responses")
        assert len(last_bodies) == prior_calls == 1
    assert len(first_bodies) == 1
    events = parse_sse_text(raw)
    assert [event.event for event in events] == ["response.created", "response.failed"]
    first_event, final = (event.data for event in events)
    assert final["response"]["id"] == first_event["response"]["id"]
    assert final["response"]["created_at"] == first_event["response"]["created_at"]
    assert final["sequence_number"] > first_event["sequence_number"]


def failed_overlapping_calls():
    template = tool_events("responses", '{"path":"finished"}')
    response = deepcopy(template[-1]["response"])
    finished = {
        **response["output"][0],
        "id": "fc_finished",
        "call_id": "call_finished",
        "status": "completed",
    }
    unfinished = {
        **finished,
        "id": "fc_unfinished",
        "call_id": "call_unfinished",
        "arguments": '{"path":',
        "status": "incomplete",
    }
    events = [
        {
            **template[0],
            "response": {**response, "status": "in_progress", "output": []},
        },
        *(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            }
            for index, item in enumerate([finished, unfinished])
        ),
        {"type": "response.output_item.done", "output_index": 0, "item": finished},
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "fc_unfinished",
            "delta": unfinished["arguments"],
        },
        {
            "type": "response.failed",
            "response": {
                **response,
                "output": [finished, unfinished],
                "status": "failed",
                "error": {"code": "server_error", "message": "cut"},
            },
        },
    ]
    return [{**event, "sequence_number": index} for index, event in enumerate(events)]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_candidate", [False, True])
@pytest.mark.parametrize("hold_final_attempt", [False, True])
async def test_failed_snapshot_cannot_release_an_overlapping_group_hidden_by_provider(
    monkeypatch, same_candidate, hold_final_attempt
):
    buffers = []

    def buffer():
        value = RecoveryHoldbackBuffer(
            max_bytes=1_000_000 if buffers and hold_final_attempt else 1,
            holdback_seconds=60,
        )
        buffers.append(value)
        return value

    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer", buffer
    )
    async with (
        _harness(
            "responses",
            lambda bodies: (
                200,
                failed_overlapping_calls()
                if same_candidate and len(bodies) > 1
                else text_events("responses", "")[:1],
            ),
            max_attempts=2 if same_candidate else 1,
        ) as (first, first_bodies, _),
        _harness(
            "responses", lambda _: (200, failed_overlapping_calls()), max_attempts=1
        ) as (last, last_bodies, _),
    ):
        raw = await delivered_candidates(
            [first] if same_candidate else [first, last],
            "responses",
            tools=_tools("responses"),
        )
    assert len(first_bodies) == (2 if same_candidate else 1)
    assert len(last_bodies) == (0 if same_candidate else 1)
    events = parse_sse_text(raw)
    assert [event.event for event in events] == ["response.created", "response.failed"]
    assert events[-1].data["response"]["output"] == []
    assert "call_finished" not in raw and "call_unfinished" not in raw


@pytest.mark.asyncio
async def test_last_model_error_keeps_combined_output_with_public_item_ids():
    first_events = failed_response("One ", "first")
    last_events = failed_response("two", "last")
    async with (
        _harness("responses", lambda _: (200, first_events), max_attempts=5) as (
            first,
            first_bodies,
            _,
        ),
        _harness("responses", lambda _: (200, last_events), max_attempts=5) as (
            last,
            last_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, last], "responses")
    assert len(first_bodies) == len(last_bodies) == 1
    events = parse_sse_text(raw)
    assert sum(event.event == "response.failed" for event in events) == 1
    assert sum(event.event == "response.created" for event in events) == 1
    final = events[-1].data
    assert final["response"]["id"] == "resp_first"
    assert final["response"]["created_at"] == events[0].data["response"]["created_at"]
    assert final["response"]["error"] == last_events[-1]["response"]["error"]
    assert final["response"]["metadata"] == {"candidate": "last"}
    assert final["extension"] == {"candidate": "last"}
    assert public_text(events, "responses") == "One two"
    assert (
        "".join(
            part["text"]
            for item in final["response"]["output"]
            for part in item.get("content", [])
        )
        == "One two"
    )
    assert {item["id"] for item in final["response"]["output"]} == {
        event.data["item_id"]
        for event in events
        if event.event == "response.output_text.delta"
    }
    assert all(
        a.data["sequence_number"] < b.data["sequence_number"]
        for a, b in pairwise(events)
    )
    assert final["response"]["usage"]["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_fallback_timeout_before_its_first_frame_retains_prefix_and_closes(wire):
    opened = []
    closed = []

    async def waiting(*args, **kwargs):
        opened.append(kwargs["continuation"])
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed.append(True)

    async def forbidden(*args, **kwargs):
        raise AssertionError("The application deadline cannot select another candidate")
        yield "unreachable"

    async with _harness(
        "chat",
        lambda _: (200, text_events("chat", "Before. ", complete=False)),
        max_attempts=1,
    ) as (first, _, _):
        raw = await delivered_candidates(
            [first, waiting, forbidden], wire, progress_timeout=0.1
        )
    assert len(opened) == 1 and opened[0].text == "Before. "
    assert closed == [True]
    events = parse_sse_text(raw)
    assert public_text(events, wire) == "Before. "
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")
    if wire == "responses":
        final = events[-1].data["response"]
        assert final["id"] == events[0].data["response"]["id"]
        assert (
            "".join(
                part["text"]
                for item in final["output"]
                for part in item.get("content", [])
            )
            == "Before. "
        )
        assert final["error"]["type"] == "timeout_error"
