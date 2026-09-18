"""Incomplete client calls remain visible without becoming executable history."""

import json
from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.native import NativeMessagesOptions
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
    build_responses_chat_request,
    build_responses_messages_request,
)
from free_claude_code.core.openai_responses.native import build_native_responses_request
from free_claude_code.core.openai_responses.web_history import (
    prepare_web_history,
    replay_item,
)
from free_claude_code.core.openai_tool_names import encode_openai_chat_tool_names
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.core.sse import parse_sse_text
from free_claude_code.providers.openai_responses.presentation import (
    NativeResponsesPresenter,
)
from tests.core.openai_responses.test_client_tool_discovery import (
    SEARCH,
    _native_adapter,
)
from tests.core.openai_responses.test_web_history import record


def discovery(call_id, *, status="completed", arguments=None):
    return {
        "type": "tool_search_call",
        "execution": "client",
        "call_id": call_id,
        "status": status,
        "arguments": arguments,
    }


def result(call_id, name="lookup"):
    return {
        "type": "tool_search_output",
        "execution": "client",
        "call_id": call_id,
        "status": "completed",
        "tools": [{"type": "function", "name": name, "parameters": {"type": "object"}}],
    }


def adapted(request):
    return ResponsesToolAdapter(
        request,
        ResponsesToolPolicy(
            custom_tools_as_functions=True,
            flatten_namespaces=True,
            client_tool_search=True,
        ),
    )


@pytest.mark.parametrize("status", ["incomplete", "failed"])
@pytest.mark.parametrize("arguments", [None, "", '{"query":'])
def test_unfinished_discovery_retains_raw_arguments(status, arguments):
    adapter = _native_adapter([SEARCH])
    item = {
        "id": "fc_search",
        "type": "function_call",
        "call_id": "search",
        "name": "fcc_tool_search",
        "arguments": arguments,
        "status": status,
    }
    restored = adapter.restore_item(item)
    assert restored == {
        **discovery("search", status=status, arguments=arguments),
        "id": "fc_search",
    }


@pytest.mark.parametrize("status", [None, "completed"])
@pytest.mark.parametrize("arguments", ["", '{"query":', "[]", "null", '{"x":NaN}'])
def test_finished_discovery_still_requires_valid_object_arguments(status, arguments):
    item = {
        "id": "fc_search",
        "type": "function_call",
        "call_id": "search",
        "name": "fcc_tool_search",
        "arguments": arguments,
    }
    if status is not None:
        item["status"] = status
    with pytest.raises(ResponsesConversionError):
        _native_adapter([SEARCH]).restore_item(item)


@pytest.mark.parametrize("name", ["fcc_tool_search", "lookup", "patch"])
@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_adapted_native_failure_does_not_emit_executable_client_completion(
    name, status
):
    adapter = _native_adapter(
        [
            SEARCH,
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
            {"type": "custom", "name": "patch"},
        ]
    )
    presenter = NativeResponsesPresenter(
        public_model="example", tool_events=adapter.event_adapter()
    )
    item = {
        "id": "fc_partial",
        "type": "function_call",
        "call_id": "partial",
        "name": name,
        "arguments": '{"input":',
        "status": "in_progress",
    }
    list(presenter.feed("response.created", {"response": {"id": "resp", "output": []}}))
    list(
        presenter.feed("response.output_item.added", {"output_index": 1, "item": item})
    )
    item = {**item, "status": "incomplete"}
    done = parse_sse_text(
        "".join(
            presenter.feed(
                "response.output_item.done", {"output_index": 1, "item": item}
            )
        )
    )
    if name == "fcc_tool_search":
        assert len(done) == 1 and done[0].event == "response.output_item.done"
        assert done[0].data["item"]["status"] == "incomplete"
        assert done[0].data["item"]["arguments"] == item["arguments"]
    else:
        assert done == []
    text = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "Already finished.", "annotations": []}
        ],
    }
    response = {
        "id": "resp",
        "status": status,
        "output": [text, item],
        "error": {"code": "upstream_error", "message": "Original failure"},
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    events = parse_sse_text(
        "".join(presenter.feed(f"response.{status}", {"response": response}))
    )
    assert len(events) == 1 and events[0].event == f"response.{status}"
    final = events[0].data["response"]
    assert final["output"][0] == text
    assert final["output"][1]["status"] == "incomplete"
    assert final["error"] == response["error"]
    assert final["incomplete_details"] == response["incomplete_details"]


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("with_web", [False, True])
@pytest.mark.parametrize("with_output", [False, True])
def test_retry_omits_only_aborted_discovery_exchange(protocol, with_web, with_output):
    input_items = [
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "Completed text"},
        discovery("aborted", status="incomplete", arguments=""),
        *([result("aborted", "must_not_activate")] if with_output else []),
        *([replay_item([record()])] if with_web else []),
        discovery("valid", arguments={"query": "lookup"}),
        result("valid"),
    ]
    request = OpenAIResponsesRequest(model="example", tools=[SEARCH], input=input_items)
    original = request.model_dump()
    prepared = prepare_web_history(request).request
    adapter = adapted(prepared)
    if protocol == "chat":
        body = build_responses_chat_request(
            prepared, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        output = next(
            message
            for message in cast(list[dict[str, Any]], body["messages"])
            if message.get("tool_call_id") == "valid"
        )
        assert json.loads(output["content"]) == [
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ]
    elif protocol == "messages":
        body = build_responses_messages_request(
            adapter.request, options=NativeMessagesOptions("example", 512)
        ).body
    else:
        body = build_native_responses_request(
            adapter.request, model="example", reasoning=DEFAULT_REASONING_POLICY
        )
    encoded = json.dumps(body)
    assert "aborted" not in encoded and "must_not_activate" not in encoded
    assert "Completed text" in encoded and "lookup" in encoded
    assert ("Retained search evidence" in encoded) is with_web
    assert request.model_dump() == original


@pytest.mark.parametrize("first_incomplete", [False, True])
@pytest.mark.parametrize("superseded", [False, True])
def test_retry_correlates_reused_call_ids_by_occurrence(first_incomplete, superseded):
    calls = [
        discovery(
            "reused",
            status="incomplete" if first_incomplete else "completed",
            arguments="" if first_incomplete else {"query": "first"},
        ),
        discovery(
            "reused",
            status="completed" if first_incomplete else "incomplete",
            arguments={"query": "second"} if first_incomplete else "",
        ),
    ]
    items = [
        calls[0],
        *([] if superseded else [result("reused", "first")]),
        calls[1],
        result("reused", "second"),
    ]
    request = OpenAIResponsesRequest(model="example", tools=[SEARCH], input=items)
    adapter = adapted(request)
    outputs = [
        item for item in adapter.request.input if item["type"] == "function_call_output"
    ]
    names = [json.loads(item["output"])[0]["name"] for item in outputs]
    assert names == (
        ["second"] if first_incomplete else [] if superseded else ["first"]
    )
    assert (
        len([item for item in adapter.request.input if item["type"] == "function_call"])
        == 1
    )


def test_later_aborted_call_does_not_taint_standalone_earlier_output():
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            result("reused", "earlier"),
            discovery("reused", status="incomplete", arguments=""),
        ],
    )
    adapter = adapted(request)
    assert len(adapter.request.input) == 1
    assert json.loads(adapter.request.input[0]["output"])[0]["name"] == "earlier"


def test_unadapted_native_request_preserves_aborted_discovery():
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            discovery("aborted", status="incomplete", arguments=""),
            result("aborted"),
            replay_item([record()]),
        ],
    )
    before = deepcopy(request.input)
    prepared = prepare_web_history(request).request
    body = build_native_responses_request(
        prepared, model="example", reasoning=DEFAULT_REASONING_POLICY
    )
    wire_input = body["input"]
    assert isinstance(wire_input, list) and isinstance(before, list)
    assert wire_input[:2] == before[:2]
    assert request.input == before


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_aborted_exchange_between_other_calls_and_results_preserves_group(protocol):
    items = [
        {"role": "user", "content": "Continue"},
        {"type": "function_call", "name": "read", "call_id": "read", "arguments": "{}"},
        discovery("aborted", status="incomplete", arguments=""),
        {
            "type": "custom_tool_call",
            "name": "patch",
            "call_id": "patch",
            "input": "edit",
        },
        discovery("discover", arguments={"query": "lookup"}),
        {"type": "function_call_output", "call_id": "read", "output": "file contents"},
        result("aborted", "must_not_activate"),
        {"type": "custom_tool_call_output", "call_id": "patch", "output": "patched"},
        result("discover"),
    ]
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH, {"type": "custom", "name": "patch"}],
        input=items,
    )
    adapter = adapted(request)
    if protocol == "messages":
        body = build_responses_messages_request(
            adapter.request, options=NativeMessagesOptions("example", 512)
        ).body
        messages = cast(list[dict[str, Any]], body["messages"])
        assert [block["id"] for block in messages[1]["content"]] == [
            "read",
            "patch",
            "discover",
        ]
        assert [block["tool_use_id"] for block in messages[2]["content"]] == [
            "read",
            "patch",
            "discover",
        ]
    elif protocol == "chat":
        body = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        messages = cast(list[dict[str, Any]], body["messages"])
        assert [call["id"] for call in messages[1]["tool_calls"]] == [
            "read",
            "patch",
            "discover",
        ]
        assert [message["tool_call_id"] for message in messages[2:]] == [
            "read",
            "patch",
            "discover",
        ]
    else:
        body = build_native_responses_request(
            adapter.request, model="example", reasoning=DEFAULT_REASONING_POLICY
        )
        wire_input = cast(list[dict[str, Any]], body["input"])
        assert [item["call_id"] for item in wire_input[1:]] == [
            "read",
            "patch",
            "discover",
            "read",
            "patch",
            "discover",
        ]
    assert "aborted" not in json.dumps(body) and "must_not_activate" not in json.dumps(
        body
    )


def test_chat_keeps_custom_output_source_type_after_filtering():
    output = [
        {"type": "input_image", "image_url": "https://example.org/screenshot.png"}
    ]
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH, {"type": "custom", "name": "patch"}],
        input=[
            {"role": "user", "content": "Continue"},
            discovery("aborted", status="incomplete", arguments=""),
            result("aborted"),
            {
                "type": "custom_tool_call",
                "name": "patch",
                "call_id": "patch",
                "input": "edit",
            },
            {"type": "custom_tool_call_output", "call_id": "patch", "output": output},
        ],
    )
    body = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    ).body
    messages = cast(list[dict[str, Any]], body["messages"])
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert json.loads(messages[-1]["content"]) == output


def test_chat_encodes_discovered_names_after_filtering():
    discovered = result("valid", "tool with spaces")
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            {"role": "user", "content": "Continue"},
            discovery("aborted", status="incomplete", arguments=""),
            result("aborted"),
            discovery("valid", arguments={"query": "lookup"}),
            discovered,
        ],
    )
    prepared = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    )
    body = prepared.body
    encode_openai_chat_tool_names(body, prepared.tool_names)
    tools = cast(list[dict[str, Any]], body["tools"])
    messages = cast(list[dict[str, Any]], body["messages"])
    exposed = [tool["function"]["name"] for tool in tools]
    replayed = json.loads(messages[-1]["content"])
    assert replayed[0]["name"] in exposed
    assert replayed[0]["name"] != "tool with spaces"
