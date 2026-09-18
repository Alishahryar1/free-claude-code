"""Private terminal snapshots must agree with the published web response."""

from copy import deepcopy

import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import ResponsesConversionError
from free_claude_code.core.openai_responses.web_request import prepare_web_request
from free_claude_code.core.openai_responses.web_stream import WebResponsePresenter
from free_claude_code.core.sse import parse_sse_text
from tests.application.test_responses_web_workflow import request


def _feed(presenter, kind, **data):
    from free_claude_code.core.openai_responses.events import format_response_sse_event

    frames = presenter.feed(
        parse_sse_text(format_response_sse_event(kind, {"type": kind, **data}))[0]
    )
    return parse_sse_text("".join(frames))


def _started():
    original = request()
    _, spec = prepare_web_request(original)
    assert spec is not None
    presenter = WebResponsePresenter(original, spec)
    _feed(
        presenter,
        "response.created",
        response={
            "id": "resp_source",
            "status": "in_progress",
            "output": [],
            "usage": None,
        },
    )
    return presenter


@pytest.mark.parametrize("status", ["failed", "incomplete"])
@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "partial answer", "annotations": []}
            ],
        },
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "partial thought"}],
            "encrypted_content": "opaque",
        },
        {
            "type": "function_call",
            "call_id": "call_client",
            "name": "read_file",
            "arguments": "",
        },
        {
            "type": "custom_tool_call",
            "call_id": "call_client",
            "name": "patch",
            "namespace": "editor",
            "input": "",
        },
        {
            "type": "tool_search_call",
            "call_id": "call_client",
            "execution": "client",
            "arguments": {},
        },
    ],
)
def test_terminal_only_incomplete_snapshot_closes_announced_item(item, status):
    presenter = _started()
    initial = {**item, "id": "source_item", "status": "in_progress"}
    if "content" in initial:
        initial["content"] = []
    initial.pop("encrypted_content", None)
    added = _feed(presenter, "response.output_item.added", output_index=0, item=initial)
    snapshot = {**item, "id": "source_item", "status": "incomplete"}
    events = _feed(
        presenter,
        f"response.{status}",
        response={
            "status": status,
            "output": [snapshot],
            "usage": None,
        },
    )
    final = parse_sse_text(presenter.finish())[0].data["response"]
    expected = {**snapshot, "id": added[0].data["item"]["id"]}
    assert final["output"] == [expected]
    assert [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ] == [expected]
    assert all("arguments.done" not in event.event for event in events)
    assert not presenter.current[0].private


def test_failed_snapshot_omitting_malformed_call_preserves_other_done_items():
    presenter = _started()
    complete = {
        "type": "message",
        "id": "source_message",
        "status": "completed",
        "role": "assistant",
        "content": [],
    }
    _feed(
        presenter,
        "response.output_item.added",
        output_index=0,
        item={**complete, "status": "in_progress"},
    )
    completed = _feed(
        presenter, "response.output_item.done", output_index=0, item=complete
    )
    malformed = {
        "type": "function_call",
        "id": "source_call",
        "call_id": "client_call",
        "name": "read_file",
        "arguments": "",
        "status": "in_progress",
    }
    added = _feed(
        presenter, "response.output_item.added", output_index=1, item=malformed
    )
    events = _feed(
        presenter,
        "response.failed",
        response={"status": "failed", "output": [complete], "usage": None},
    )
    final = parse_sse_text(presenter.finish())[0].data["response"]
    assert final["output"] == [
        completed[-1].data["item"],
        {**malformed, "id": added[0].data["item"]["id"], "status": "incomplete"},
    ]
    assert [
        event.data["output_index"]
        for event in events
        if event.event == "response.output_item.done"
    ] == [1]


def test_unannounced_function_is_omitted_without_a_public_index_hole():
    presenter = _started()
    assert (
        _feed(
            presenter,
            "response.output_item.added",
            output_index=0,
            item={
                "id": "unknown",
                "type": "function_call",
                "name": "",
                "arguments": "",
                "status": "in_progress",
            },
        )
        == []
    )
    added = _feed(
        presenter,
        "response.output_item.added",
        output_index=1,
        item={
            "id": "message",
            "type": "message",
            "role": "assistant",
            "content": [],
            "status": "in_progress",
        },
    )
    assert added[0].data["output_index"] == 0
    presenter.fail(ExecutionFailure(FailureKind.UPSTREAM, 502, "failed", False))
    final = parse_sse_text(presenter.finish())[0].data["response"]
    assert len(final["output"]) == 1
    assert final["output"][0]["type"] == "message"


def test_terminal_snapshot_cannot_substitute_for_provider_item_done():
    presenter = _started()
    item = {
        "type": "function_call",
        "id": "source_call",
        "call_id": "call",
        "name": "fcc_web",
        "arguments": '{"action":"search","query":"docs"}',
        "status": "completed",
    }
    _feed(
        presenter,
        "response.output_item.added",
        output_index=0,
        item={**item, "status": "in_progress"},
    )
    _feed(
        presenter,
        "response.completed",
        response={"status": "completed", "output": [item], "usage": None},
    )
    with pytest.raises(ResponsesConversionError, match="finish"):
        presenter.private_output()


def test_invalid_later_snapshot_does_not_lose_earlier_item_completion():
    presenter = _started()
    items = [
        {
            "id": name,
            "type": "message",
            "role": "assistant",
            "content": [],
            "status": "in_progress",
        }
        for name in ("first", "second")
    ]
    for index, item in enumerate(items):
        _feed(presenter, "response.output_item.added", output_index=index, item=item)
    with pytest.raises(ResponsesConversionError):
        _feed(
            presenter,
            "response.failed",
            response={
                "status": "failed",
                "output": [
                    {**items[0], "status": "incomplete"},
                    {**items[1], "type": "function_call", "status": "incomplete"},
                ],
            },
        )
    frames = presenter.fail(
        ExecutionFailure(FailureKind.UPSTREAM, 502, "invalid snapshot", False)
    )
    done = [
        event.data["output_index"]
        for event in parse_sse_text("".join(frames))
        if event.event == "response.output_item.done"
    ]
    assert done == [0, 1]


@pytest.mark.parametrize(
    "late_kind", ["response.output_item.done", "response.output_text.delta"]
)
def test_late_provider_data_cannot_mutate_a_published_done_snapshot(late_kind):
    presenter = _started()
    item = {
        "id": "source",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [],
    }
    _feed(
        presenter,
        "response.output_item.added",
        output_index=0,
        item={**item, "status": "in_progress"},
    )
    _feed(presenter, "response.output_item.done", output_index=0, item=item)
    before = deepcopy(presenter.payload("in_progress")["output"])
    with pytest.raises(ResponsesConversionError):
        _feed(
            presenter,
            late_kind,
            output_index=0,
            item=item,
            delta="late text",
            content_index=0,
        )
    assert presenter.payload("in_progress")["output"] == before
