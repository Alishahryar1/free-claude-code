import json
from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
    build_responses_chat_request,
    responses_tool_identity_from_wire_name,
)

SEARCH: JsonObject = {
    "type": "tool_search",
    "execution": "client",
    "description": "Find callable tools.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}
AGENTS: JsonObject = {
    "type": "namespace",
    "name": "agents",
    "tools": [
        {
            "type": "function",
            "name": "spawn_agent",
            "defer_loading": True,
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }
    ],
}


def test_chat_exposes_client_search_as_a_callable_function() -> None:
    request = OpenAIResponsesRequest(
        model="example", input="Find an agent", tools=[SEARCH]
    )
    translated = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    )
    functions = cast(list[dict[str, Any]], translated.body.get("tools", []))
    assert len(functions) == 1
    assert functions[0]["function"]["parameters"] == SEARCH["parameters"]


def test_chat_discovery_preserves_pairing_and_activates_returned_tool() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            {"role": "user", "content": "Find an agent"},
            {
                "type": "tool_search_call",
                "call_id": "search",
                "execution": "client",
                "status": "completed",
                "arguments": {"query": "agent"},
            },
            {
                "type": "tool_search_output",
                "call_id": "search",
                "execution": "client",
                "status": "completed",
                "tools": [AGENTS],
            },
        ],
    )
    original = request.model_dump()
    body = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    ).body
    functions = {
        t["function"]["name"]: t["function"]
        for t in cast(list[dict[str, Any]], body.get("tools", []))
    }
    assert "agents__spawn_agent" in functions
    messages = cast(list[dict[str, Any]], body["messages"])
    assert messages[1]["tool_calls"][0]["id"] == "search"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "agent"
    }
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "search"
    assert "spawn_agent" in messages[2]["content"]
    assert request.model_dump() == original


def test_unique_bare_tool_name_restores_declared_namespace() -> None:
    identity = responses_tool_identity_from_wire_name([AGENTS], "spawn_agent")
    assert identity.name == "spawn_agent"
    assert identity.namespace == "agents"


def _native_adapter(tools: list[JsonObject]) -> ResponsesToolAdapter:
    return ResponsesToolAdapter(
        OpenAIResponsesRequest(model="example", input="hello", tools=tools),
        ResponsesToolPolicy(
            custom_tools_as_functions=True,
            client_tool_search=True,
            flatten_namespaces=True,
        ),
    )


def test_ordinary_request_definitions_remain_unchanged_without_discovery() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[{**AGENTS, "description": "Agent tools"}, {"type": "web_search"}],
    )
    adapter = ResponsesToolAdapter(
        request, ResponsesToolPolicy(client_tool_search=True)
    )
    assert adapter.request.tools == request.tools


def test_native_namespaced_argument_done_event_matches_completed_item() -> None:
    adapter = _native_adapter([AGENTS])
    events = adapter.event_adapter()
    assert events is not None
    item: JsonObject = {
        "type": "function_call",
        "id": "fc_one",
        "call_id": "one",
        "name": "agents__spawn_agent",
        "arguments": "",
        "status": "in_progress",
    }
    added = list(
        events.feed("response.output_item.added", {"item": item, "output_index": 0})
    )
    assert cast(dict[str, Any], added[-1][1]["item"])["name"] == "spawn_agent"
    done = list(
        events.feed(
            "response.function_call_arguments.done",
            {
                "item_id": "fc_one",
                "name": "agents__spawn_agent",
                "arguments": '{"message":"hello"}',
                "output_index": 0,
            },
        )
    )
    assert done == []
    completed = list(
        events.feed(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {
                    **item,
                    "status": "completed",
                    "arguments": '{"message":"hello"}',
                },
            },
        )
    )
    done_event = next(
        data
        for kind, data in completed
        if kind == "response.function_call_arguments.done"
    )
    assert done_event["name"] == "spawn_agent"
    assert done_event["namespace"] == "agents"


def test_native_rejects_ambiguous_bare_names() -> None:
    other = deepcopy(AGENTS)
    other["name"] = "other"
    adapter = _native_adapter([AGENTS, other])
    with pytest.raises(ResponsesConversionError, match="Ambiguous"):
        adapter.restore_item(
            {
                "type": "function_call",
                "name": "spawn_agent",
                "arguments": '{"message":"hello"}',
                "status": "completed",
            }
        )


def test_native_search_buffers_partial_added_arguments() -> None:
    adapter = _native_adapter([SEARCH])
    events = adapter.event_adapter()
    assert events is not None
    added = list(
        events.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_search",
                    "call_id": "search",
                    "name": "fcc_tool_search",
                    "status": "in_progress",
                    "arguments": '{"query":',
                }
            },
        )
    )
    assert cast(dict[str, Any], added[-1][1]["item"])["type"] == "tool_search_call"
    assert cast(dict[str, Any], added[-1][1]["item"])["arguments"] == {}
    assert (
        list(
            events.feed(
                "response.function_call_arguments.delta",
                {"item_id": "fc_search", "delta": '"agent"}'},
            )
        )
        == []
    )
    completed = list(
        events.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_search",
                    "call_id": "search",
                    "name": "fcc_tool_search",
                    "status": "completed",
                    "arguments": '{"query":"agent"}',
                }
            },
        )
    )
    assert cast(dict[str, Any], completed[-1][1]["item"])["arguments"] == {
        "query": "agent"
    }


def test_native_does_not_publish_empty_completed_arguments_as_success() -> None:
    adapter = _native_adapter([AGENTS])
    with pytest.raises(ResponsesConversionError, match="arguments"):
        adapter.restore_item(
            {
                "type": "function_call",
                "name": "agents__spawn_agent",
                "status": "completed",
                "arguments": "",
            }
        )


def test_search_lowering_does_not_shadow_a_real_function() -> None:
    adapter = _native_adapter(
        [
            SEARCH,
            {
                "type": "function",
                "name": "fcc_tool_search",
                "parameters": {"type": "object"},
            },
        ]
    )
    assert len({tool["name"] for tool in (adapter.request.tools or [])}) == 2
    result = adapter.restore_item(
        {
            "type": "function_call",
            "name": "fcc_tool_search",
            "status": "completed",
            "arguments": "{}",
        }
    )
    assert isinstance(result, dict)
    assert result["type"] == "function_call"


def test_flattened_function_collision_is_rejected() -> None:
    with pytest.raises(ResponsesConversionError, match="collide"):
        _native_adapter(
            [
                AGENTS,
                {
                    "type": "function",
                    "name": "agents__spawn_agent",
                    "parameters": {"type": "object"},
                },
            ]
        )


@pytest.mark.parametrize("search", [True, False])
def test_integral_json_numbers_are_accepted_by_native_codex(search: bool) -> None:
    adapter = _native_adapter([SEARCH, AGENTS])
    value = adapter.restore_item(
        {
            "type": "function_call",
            "name": "fcc_tool_search" if search else "agents__spawn_agent",
            "status": "completed",
            "arguments": '{"limit":8.0,"fraction":0.25,"nested":[30000.0]}',
        }
    )
    assert isinstance(value, dict)
    arguments = value["arguments"] if search else json.loads(str(value["arguments"]))
    assert isinstance(arguments, dict)
    assert type(arguments["limit"]) is int
    assert arguments["fraction"] == 0.25
    nested = arguments["nested"]
    assert isinstance(nested, list)
    assert type(nested[0]) is int


def test_server_search_choice_stays_native() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[{"type": "tool_search", "execution": "server"}],
        tool_choice={"type": "tool_search", "execution": "server"},
    )
    adapter = ResponsesToolAdapter(
        request, ResponsesToolPolicy(client_tool_search=True, flatten_namespaces=True)
    )
    assert adapter.request.tool_choice == request.tool_choice


def test_latest_discovery_and_explicit_definitions_have_stable_precedence() -> None:
    def declaration(description: str) -> JsonObject:
        return {
            "type": "function",
            "name": "lookup",
            "description": description,
            "parameters": {"type": "object"},
            "defer_loading": True,
        }

    history = [
        {
            "type": "tool_search_output",
            "call_id": str(i),
            "execution": "client",
            "status": "completed",
            "tools": tools,
        }
        for i, tools in enumerate([[declaration("old")], [declaration("new")], []])
    ]
    request = OpenAIResponsesRequest(model="example", input=history, tools=[SEARCH])
    adapter = ResponsesToolAdapter(
        request, ResponsesToolPolicy(client_tool_search=True)
    )
    definitions = adapter.request.tools or []
    assert len(definitions) == 2
    assert (
        next(t for t in definitions if t.get("name") == "lookup")["description"]
        == "new"
    )
    explicit = request.model_copy(update={"tools": [SEARCH, declaration("current")]})
    definitions = (
        ResponsesToolAdapter(
            explicit, ResponsesToolPolicy(client_tool_search=True)
        ).request.tools
        or []
    )
    assert (
        next(t for t in definitions if t.get("name") == "lookup")["description"]
        == "current"
    )
    assert all("defer_loading" not in tool for tool in definitions)


def test_conflicting_discovered_definitions_are_rejected() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input=[
            {
                "type": "tool_search_output",
                "execution": "client",
                "status": "completed",
                "call_id": "search",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "string"},
                    },
                ],
            }
        ],
        tools=[SEARCH],
    )
    with pytest.raises(ResponsesConversionError, match="Conflicting"):
        ResponsesToolAdapter(request, ResponsesToolPolicy(client_tool_search=True))


def test_native_custom_namespace_is_removed_from_provider_definition() -> None:
    adapter = _native_adapter(
        [
            {
                "type": "custom",
                "namespace": "editor",
                "name": "edit",
                "format": {"type": "text"},
            }
        ]
    )
    tool = (adapter.request.tools or [])[0]
    assert tool["name"] == "editor__edit"
    assert "namespace" not in tool


def test_real_unnamespaced_tool_wins_over_bare_namespace_alias() -> None:
    tools = [
        AGENTS,
        {"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}},
    ]
    identity = responses_tool_identity_from_wire_name(tools, "spawn_agent")
    assert identity.namespace is None


def test_native_canonical_arguments_agree_across_the_stream() -> None:
    adapter = _native_adapter([AGENTS])
    stream = adapter.event_adapter()
    assert stream is not None
    item: JsonObject = {
        "type": "function_call",
        "id": "fc_one",
        "call_id": "one",
        "name": "agents__spawn_agent",
        "status": "in_progress",
        "arguments": "",
    }
    events = list(
        stream.feed("response.output_item.added", {"item": item, "output_index": 0})
    )
    for fragment in ['{"limit":', "8.0}"]:
        events.extend(
            stream.feed(
                "response.function_call_arguments.delta",
                {"item_id": "fc_one", "output_index": 0, "delta": fragment},
            )
        )
    events.extend(
        stream.feed(
            "response.function_call_arguments.done",
            {
                "item_id": "fc_one",
                "output_index": 0,
                "name": "agents__spawn_agent",
                "arguments": '{"limit":8.0}',
            },
        )
    )
    events.extend(
        stream.feed(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {**item, "status": "completed", "arguments": '{"limit":8.0}'},
            },
        )
    )
    completed = cast(dict[str, Any], events[-1][1]["item"])
    delta = "".join(
        str(data["delta"])
        for kind, data in events
        if kind == "response.function_call_arguments.delta"
    )
    done = next(
        data for kind, data in events if kind == "response.function_call_arguments.done"
    )
    assert delta == done["arguments"] == completed["arguments"]


def test_unspecified_tool_metadata_keeps_native_defaults() -> None:
    adapter = _native_adapter([])
    events = adapter.event_adapter()
    assert events is not None
    result = list(
        events.feed(
            "response.completed",
            {"response": {"output": [], "tools": [], "tool_choice": "auto"}},
        )
    )
    response = cast(dict[str, Any], result[-1][1]["response"])
    assert response["tool_choice"] == "auto"
    assert response["tools"] == []


@pytest.mark.parametrize("discovery", [False, True])
def test_chat_forced_namespaced_custom_choice_is_lowered_once(discovery: bool) -> None:
    custom: JsonObject = {
        "type": "namespace",
        "name": "editor",
        "tools": [{"type": "custom", "name": "edit", "format": {"type": "text"}}],
    }
    request = OpenAIResponsesRequest(
        model="example",
        input="Edit the file",
        tools=[custom, SEARCH] if discovery else [custom],
        tool_choice={"type": "custom", "namespace": "editor", "name": "edit"},
    )
    body = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    ).body
    assert body.get("tool_choice") == {
        "type": "function",
        "function": {"name": "editor__edit"},
    }
    functions = cast(list[dict[str, Any]], body["tools"])
    assert any(tool["function"]["name"] == "editor__edit" for tool in functions)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "status", ["omitted", None, "completed", "in_progress", "incomplete", "failed"]
)
def test_discovery_optional_status_preserves_valid_tools(
    native: bool, status: str | None
) -> None:
    result: JsonObject = {
        "type": "tool_search_output",
        "execution": "client",
        "call_id": "search",
        "tools": [
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ],
    }
    if status != "omitted":
        result["status"] = status
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            {"role": "user", "content": "Find lookup"},
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "search",
                "arguments": {"query": "lookup"},
            },
            result,
        ],
    )
    original = request.model_dump()
    if native:
        adapter = ResponsesToolAdapter(
            request,
            ResponsesToolPolicy(client_tool_search=True, flatten_namespaces=True),
        )
        definitions = adapter.request.tools or []
        names = [tool.get("name") for tool in definitions]
        items = cast(list[dict[str, Any]], adapter.request.input)
        payload = json.loads(items[-1]["output"])
        assert items[-1]["call_id"] == "search"
    else:
        body = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        definitions = cast(list[dict[str, Any]], body["tools"])
        names = [tool["function"]["name"] for tool in definitions]
        messages = cast(list[dict[str, Any]], body["messages"])
        payload = json.loads(messages[-1]["content"])
        assert messages[-1]["tool_call_id"] == "search"
    accepted = status in {"omitted", None, "completed"}
    assert ("lookup" in names) is accepted
    assert payload == (
        [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]
        if accepted
        else []
    )
    assert request.model_dump() == original
