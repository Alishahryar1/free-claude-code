"""Responses completion belongs to items, never argument or formatter ends."""

import pytest

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses.tool_calls import ResponsesToolCallBuffer


def item(index, status="completed", arguments="{}"):
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "read",
        "arguments": arguments,
        "status": status,
    }


def test_argument_done_does_not_release_call_and_incomplete_item_is_discarded():
    buffer = ResponsesToolCallBuffer()
    assert (
        buffer.feed(
            "response.output_item.added",
            {"output_index": 0, "item": item(0, "in_progress", "")},
        )
        == []
    )
    assert (
        buffer.feed(
            "response.function_call_arguments.done",
            {"output_index": 0, "item_id": "fc_0", "arguments": "{}"},
        )
        == []
    )
    assert (
        buffer.feed(
            "response.output_item.done",
            {"output_index": 0, "item": item(0, "incomplete")},
        )
        == []
    )
    output = buffer.feed(
        "response.incomplete",
        {"response": {"output": [item(0, "incomplete")], "status": "incomplete"}},
    )
    assert output[-1][-1][1]["response"]["output"] == []


def test_completed_call_survives_another_unfinished_call_on_failure():
    buffer = ResponsesToolCallBuffer()
    for index in (0, 1):
        assert (
            buffer.feed(
                "response.output_item.added",
                {"output_index": index, "item": item(index, "in_progress", "")},
            )
            == []
        )
    assert (
        buffer.feed("response.output_item.done", {"output_index": 1, "item": item(1)})
        == []
    )
    output = buffer.abandon()
    assert output[-1][-1][1]["item"] == item(1)


def test_terminal_snapshot_completes_missing_incremental_call_once():
    buffer = ResponsesToolCallBuffer()
    output = buffer.feed(
        "response.completed", {"response": {"status": "completed", "output": [item(0)]}}
    )
    done = [
        payload
        for group in output
        for kind, payload in group
        if kind == "response.output_item.done"
    ]
    assert [payload["item"] for payload in done] == [item(0)]


def test_fragment_metadata_keeps_its_event_and_does_not_overwrite_other_fragments():
    buffer = ResponsesToolCallBuffer()
    events = [
        (
            "response.output_item.added",
            {
                "output_index": 0,
                "item": item(0, "in_progress", ""),
                "extension": "start",
            },
        ),
        (
            "response.function_call_arguments.delta",
            {"item_id": "fc_0", "output_index": 0, "delta": "{", "extension": "first"},
        ),
        (
            "response.function_call_arguments.delta",
            {"item_id": "fc_0", "output_index": 0, "delta": "}", "extension": "second"},
        ),
        (
            "response.output_item.done",
            {"output_index": 0, "item": item(0), "extension": "end"},
        ),
    ]
    output = []
    for kind, data in events:
        output.extend(buffer.feed(kind, data))
    assert [
        (kind, data["extension"])
        for group in output
        for kind, data in group
        if "extension" in data
    ] == [(kind, data["extension"]) for kind, data in events]


def test_failed_snapshot_keeps_only_explicit_completed_calls():
    buffer = ResponsesToolCallBuffer()
    groups = buffer.feed(
        "response.failed",
        {"response": {"status": "failed", "output": [item(0, "incomplete"), item(1)]}},
    )
    assert [
        data["item"]["id"]
        for group in groups
        for kind, data in group
        if kind == "response.output_item.done"
    ] == ["fc_1"]
    assert groups[-1][-1][1]["response"]["output"] == [item(1)]


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize(
    "kind", ["response.output_item.done", "response.function_call_arguments.delta"]
)
def test_explicit_index_cannot_change_for_known_item(completed, kind):
    buffer = ResponsesToolCallBuffer()
    buffer.feed(
        "response.output_item.added",
        {"output_index": 0, "item": item(0, "in_progress", "")},
    )
    if completed:
        buffer.feed("response.output_item.done", {"output_index": 0, "item": item(0)})
    payload = (
        {"output_index": 1, "item": item(0)}
        if kind.endswith(".done")
        else {"output_index": 1, "item_id": "fc_0", "delta": "{}"}
    )
    with pytest.raises(ExecutionFailure):
        buffer.feed(kind, payload)


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function_call", "name": "f", "arguments": "{}"},
        {"type": "custom_tool_call", "name": "f", "input": "patch"},
        {"type": "computer_call", "action": {"type": "screenshot"}},
        {
            "type": "local_shell_call",
            "action": {"type": "exec", "command": ["echo", "test"]},
        },
        {
            "type": "apply_patch_call",
            "operation": {"type": "delete_file", "path": "test"},
        },
        {
            "type": "shell_call",
            "environment": {"type": "local"},
            "action": {"commands": ["echo test"]},
        },
        {
            "type": "tool_search_call",
            "execution": "client",
            "arguments": {"query": "test"},
        },
    ],
)
def test_all_client_call_types_wait_for_item_completion(tool):
    buffer = ResponsesToolCallBuffer()
    final = {"id": "tool", "call_id": "call", "status": "completed", **tool}
    initial = {**final, "status": "in_progress"}
    assert (
        buffer.feed("response.output_item.added", {"output_index": 0, "item": initial})
        == []
    )
    groups = buffer.feed(
        "response.output_item.done", {"output_index": 0, "item": final}
    )
    assert groups[-1][-1][1]["item"] == final


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search_call"},
        {"type": "file_search_call"},
        {"type": "tool_search_call", "execution": "server"},
        {
            "type": "shell_call",
            "environment": {"type": "container_reference", "container_id": "box"},
        },
    ],
)
def test_server_tools_are_not_gated_as_client_calls(tool):
    buffer = ResponsesToolCallBuffer()
    payload = {
        "output_index": 0,
        "item": {"id": "server", "status": "in_progress", **tool},
    }
    assert buffer.feed("response.output_item.added", payload)
