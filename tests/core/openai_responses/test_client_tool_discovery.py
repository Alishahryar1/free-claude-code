import json
from copy import deepcopy
from decimal import Decimal
from typing import Any, cast

import pytest
from openai import omit
from openai.lib.streaming.responses._responses import ResponseStreamState
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
    build_responses_chat_request,
)
from free_claude_code.providers.openai_chat.stream_output import (
    ChatStreamUsage,
    ResponsesChatStreamOutput,
)
from free_claude_code.providers.openai_responses.presentation import (
    NativeResponsesPresenter,
)
from tests.providers.test_opencode import _responses_event_stream

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
    item = cast(
        dict[str, Any],
        _native_adapter([AGENTS]).restore_item(
            {
                "type": "function_call",
                "name": "spawn_agent",
                "arguments": '{"message":"hello"}',
            }
        ),
    )
    assert item["name"] == "spawn_agent"
    assert item["namespace"] == "agents"


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
    tools: list[JsonObject] = [
        AGENTS,
        {"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}},
    ]
    item = cast(
        dict[str, Any],
        _native_adapter(tools).restore_item(
            {"type": "function_call", "name": "spawn_agent", "arguments": "{}"}
        ),
    )
    assert "namespace" not in item


@pytest.mark.parametrize("prefix", ["", '{"limit":', '{"limit":8.0}'])
def test_native_canonical_arguments_agree_across_the_stream(prefix: str) -> None:
    adapter = _native_adapter([{**AGENTS, "description": "Agent tools"}])
    stream = adapter.event_adapter()
    assert stream is not None
    item: JsonObject = {
        "type": "function_call",
        "id": "fc_one",
        "call_id": "one",
        "name": "agents__spawn_agent",
        "status": "in_progress",
        "arguments": prefix,
    }
    created = parse_sse_text(_responses_event_stream(""))[0].data
    events = list(stream.feed("response.created", created))
    events.extend(
        stream.feed("response.output_item.added", {"item": item, "output_index": 0})
    )
    for fragment in ['{"limit":8.0}'[len(prefix) :]]:
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
    sdk: ResponseStreamState[Any] = ResponseStreamState(
        input_tools=omit, text_format=omit
    )
    parser: TypeAdapter[ResponseStreamEvent] = TypeAdapter(ResponseStreamEvent)
    snapshots = [
        event.snapshot
        for _, data in events
        for event in sdk.handle_event(parser.validate_python(data))
        if event.type == "response.function_call_arguments.delta"
    ]
    assert snapshots == [completed["arguments"]]


@pytest.mark.parametrize(
    ("prefix", "arguments"),
    [
        ("", '{"input":"whole"}'),
        ('{"input":', '{"input":"whole"}'),
        ('{"input":"whole"}', '{"input":"whole"}'),
        ("wh", "whole"),
    ],
)
def test_native_custom_input_is_emitted_once_when_buffered(
    prefix: str, arguments: str
) -> None:
    adapter = _native_adapter([{"type": "custom", "name": "edit"}])
    stream = adapter.event_adapter()
    assert stream is not None
    item: JsonObject = {
        "type": "function_call",
        "id": "fc",
        "call_id": "call",
        "name": "edit",
        "status": "in_progress",
        "arguments": prefix,
    }
    events = list(
        stream.feed("response.output_item.added", {"output_index": 0, "item": item})
    )
    events.extend(
        stream.feed(
            "response.function_call_arguments.delta",
            {"output_index": 0, "item_id": "fc", "delta": arguments[len(prefix) :]},
        )
    )
    events.extend(
        stream.feed(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {**item, "status": "completed", "arguments": arguments},
            },
        )
    )
    added = cast(dict[str, Any], events[0][1]["item"])
    completed = cast(dict[str, Any], events[-1][1]["item"])
    assembled = added["input"] + "".join(
        str(data["delta"])
        for kind, data in events
        if kind == "response.custom_tool_call_input.delta"
    )
    assert assembled == completed["input"] == "whole"


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
@pytest.mark.parametrize("namespace", [None, "editor"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_generic_choice_uses_declared_tool_kind(
    native: bool, namespace: str | None, custom: bool, nested: bool
) -> None:
    definition: JsonObject = {
        "type": "custom" if custom else "function",
        "name": "edit",
    }
    if custom:
        definition["format"] = {"type": "text"}
    else:
        definition["parameters"] = {"type": "object"}
    choice: JsonObject = {"type": "tool", "name": "edit"}
    tools = [definition]
    if namespace:
        tools = [{"type": "namespace", "name": namespace, "tools": tools}]
        choice["namespace"] = namespace
    if nested:
        choice = {
            "type": "tool",
            "custom" if custom else "function": {
                key: value for key, value in choice.items() if key != "type"
            },
        }
    request = OpenAIResponsesRequest(
        model="example", input="Edit", tools=tools, tool_choice=choice
    )
    original = request.model_dump()
    wire_name = "editor__edit" if namespace else "edit"
    if native:
        prepared = ResponsesToolAdapter(
            request,
            ResponsesToolPolicy(
                custom_tools_as_functions=True, flatten_namespaces=True
            ),
        ).request
        assert prepared.tool_choice == {"type": "function", "name": wire_name}
        assert prepared.tools and prepared.tools[0]["name"] == wire_name
    else:
        body = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": wire_name},
        }
        assert (
            cast(list[dict[str, Any]], body["tools"])[0]["function"]["name"]
            == wire_name
        )
    assert request.model_dump() == original


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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("missing_side", ["call", "output"])
@pytest.mark.parametrize("missing", ["omitted", None])
def test_discovery_execution_is_inferred_from_the_matching_record(
    native: bool, missing_side: str, missing: str | None
) -> None:
    call: JsonObject = {
        "type": "tool_search_call",
        "execution": "client",
        "call_id": "search",
        "arguments": {"query": "agent"},
    }
    result: JsonObject = {
        "type": "tool_search_output",
        "execution": "client",
        "call_id": "search",
        "tools": [AGENTS],
    }
    item = call if missing_side == "call" else result
    if missing == "omitted":
        item.pop("execution")
    else:
        item["execution"] = missing
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[{"role": "user", "content": "Find agent"}, call, result],
    )
    original = request.model_dump()
    if native:
        prepared = ResponsesToolAdapter(
            request,
            ResponsesToolPolicy(client_tool_search=True, flatten_namespaces=True),
        ).request
        items = cast(list[dict[str, Any]], prepared.input)
        assert items[-2]["type"] == "function_call"
        assert items[-1]["type"] == "function_call_output"
        assert items[-2]["call_id"] == items[-1]["call_id"] == "search"
        assert "agents__spawn_agent" in [
            tool.get("name") for tool in prepared.tools or []
        ]
        assert "spawn_agent" in items[-1]["output"]
    else:
        body = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        items = cast(list[dict[str, Any]], body["messages"])
        assert items[-2]["tool_calls"][0]["id"] == "search"
        assert items[-1]["role"] == "tool"
        assert items[-1]["tool_call_id"] == "search"
        assert "spawn_agent" in items[-1]["content"]
        assert "agents__spawn_agent" in [
            tool["function"]["name"]
            for tool in cast(list[dict[str, Any]], body["tools"])
        ]
    assert request.model_dump() == original


@pytest.mark.parametrize("execution", ["server", "omitted"])
def test_discovery_call_id_does_not_imply_client_execution(execution: str) -> None:
    call: JsonObject = {
        "type": "tool_search_call",
        "call_id": "search",
        "arguments": {"query": "agent"},
    }
    if execution != "omitted":
        call["execution"] = execution
    result: JsonObject = {
        "type": "tool_search_output",
        "call_id": "search",
        "tools": [AGENTS],
    }
    request = OpenAIResponsesRequest(
        model="example", input=[call, result], tools=[SEARCH]
    )
    prepared = ResponsesToolAdapter(
        request, ResponsesToolPolicy(client_tool_search=True, flatten_namespaces=True)
    ).request
    items = cast(list[dict[str, Any]], prepared.input)
    assert [item["type"] for item in items] == [
        "tool_search_call",
        "tool_search_output",
    ]
    assert "agents__spawn_agent" not in [
        tool.get("name") for tool in prepared.tools or []
    ]


@pytest.mark.parametrize("native", [False, True])
def test_discovery_rejects_conflicting_execution_for_one_call(native: bool) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            {"role": "user", "content": "Find agent"},
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "search",
                "arguments": {"query": "agent"},
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": "search",
                "tools": [AGENTS],
            },
        ],
    )
    with pytest.raises(ResponsesConversionError, match="execution"):
        if native:
            ResponsesToolAdapter(request, ResponsesToolPolicy(client_tool_search=True))
        else:
            build_responses_chat_request(
                request, reasoning_replay=ReasoningReplayMode.DISABLED
            )


def _completed_tool_events(
    request: OpenAIResponsesRequest, *, native: bool, name: str, arguments: str
) -> list[dict[str, Any]]:
    if native:
        adapter = ResponsesToolAdapter(
            request,
            ResponsesToolPolicy(
                custom_tools_as_functions=True,
                client_tool_search=True,
                flatten_namespaces=True,
            ),
        )
        events = adapter.event_adapter()
        assert events is not None
        item: JsonObject = {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        completed: JsonObject = {**item, "arguments": arguments, "status": "completed"}
        payloads: list[tuple[str, JsonObject]] = [
            ("response.output_item.added", {"item": item, "output_index": 0}),
            ("response.output_item.done", {"item": completed, "output_index": 0}),
            (
                "response.completed",
                {"response": {"output": [completed], "status": "completed"}},
            ),
        ]
        return [
            data for kind, payload in payloads for _, data in events.feed(kind, payload)
        ]
    adapter = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    ).tool_adapter
    writer = ResponsesChatStreamOutput(adapter, input_tokens=1)
    frames = [*writer.start_events(), writer.start_tool_block(0, "call_test", name)]
    frames.append(writer.emit_tool_delta(0, arguments))
    frames.extend(
        writer.finish_success(
            stop_reason="tool_calls",
            usage=ChatStreamUsage(input_tokens=1, output_tokens=1),
        )
    )
    return [frame.data for frame in parse_sse_text("".join(frames))]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_namespaced_search_named_tool_keeps_its_identity_in_all_events(
    native: bool, custom: bool
) -> None:
    tool: JsonObject = {
        "type": "custom" if custom else "function",
        "name": "fcc_tool_search",
    }
    if not custom:
        tool["parameters"] = {"type": "object"}
    request = OpenAIResponsesRequest(
        model="example",
        input="Use ordinary tool",
        tools=[
            SEARCH,
            {"type": "namespace", "name": "ordinary", "tools": [tool]},
        ],
    )
    events = _completed_tool_events(
        request,
        native=native,
        name="ordinary__fcc_tool_search",
        arguments='{"input":"patch"}' if custom else '{"query":"ordinary"}',
    )
    expected_type = "custom_tool_call" if custom else "function_call"
    for event in events:
        if "item" in event:
            assert event["item"]["type"] == expected_type
            assert event["item"]["name"] == "fcc_tool_search"
            assert event["item"]["namespace"] == "ordinary"
    final = events[-1]["response"]["output"][0]
    assert final["type"] == expected_type
    assert final["namespace"] == "ordinary"
    assert final["call_id"] == "call_test"
    assert (
        final.get("input") == "patch"
        if custom
        else json.loads(final["arguments"]) == {"query": "ordinary"}
    )


def test_search_name_reserves_ordinary_calls_from_history() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            {
                "type": "function_call",
                "call_id": "old",
                "name": "fcc_tool_search",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "old", "output": "done"},
        ],
    )
    adapter = ResponsesToolAdapter(
        request,
        ResponsesToolPolicy(
            custom_tools_as_functions=True,
            client_tool_search=True,
            flatten_namespaces=True,
        ),
    )
    helper = (adapter.request.tools or [])[0]
    replay = cast(list[dict[str, Any]], adapter.request.input)[0]
    assert helper["name"] != replay["name"]
    restored = cast(
        dict[str, Any], adapter.restore_item({**replay, "status": "completed"})
    )
    assert restored["type"] == "function_call"


@pytest.mark.parametrize("native", [False, True])
def test_nested_function_definition_and_choice_use_one_provider_name(
    native: bool,
) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="Call lookup",
        tools=[
            {
                "type": "namespace",
                "name": "group",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    },
                ],
            },
        ],
        tool_choice={"type": "function", "name": "lookup", "namespace": "group"},
    )
    if native:
        prepared = ResponsesToolAdapter(
            request, ResponsesToolPolicy(flatten_namespaces=True)
        ).request
        definition = (prepared.tools or [])[0]
        assert "function" not in definition
        assert definition["name"] == "group__lookup"
        assert prepared.tool_choice == {"type": "function", "name": "group__lookup"}
    else:
        body = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        definition = cast(list[dict[str, Any]], body["tools"])[0]["function"]
        assert definition["name"] == "group__lookup"
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "group__lookup"},
        }


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("discovered", [False, True])
def test_discovery_preserves_outer_namespace_in_nested_tools(
    native: bool, custom: bool, discovered: bool
) -> None:
    kind = "custom" if custom else "function"
    definition: JsonObject = {"name": "run"}
    if custom:
        definition["format"] = {"type": "text"}
    else:
        definition["parameters"] = {"type": "object"}
    tool: JsonObject = {"type": kind, "namespace": "agents", kind: definition}
    tools = [SEARCH]
    items: list[JsonObject] = [{"role": "user", "content": "Run the agent tool"}]
    if discovered:
        items.extend(
            [
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
                    "tools": [tool],
                },
            ]
        )
    else:
        tools.append(tool)
    request = OpenAIResponsesRequest(
        model="example",
        input=items,
        tools=tools,
        tool_choice={"type": kind, "name": "run", "namespace": "agents"},
    )
    original = request.model_dump()
    if native:
        prepared = ResponsesToolAdapter(
            request,
            ResponsesToolPolicy(
                custom_tools_as_functions=True,
                client_tool_search=True,
                flatten_namespaces=True,
            ),
        ).request
        assert {tool["name"] for tool in prepared.tools or []} == {
            "fcc_tool_search",
            "agents__run",
        }
        assert prepared.tool_choice == {"type": "function", "name": "agents__run"}
    else:
        body = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).body
        assert {
            tool["function"]["name"]
            for tool in cast(list[dict[str, Any]], body["tools"])
        } == {"fcc_tool_search", "agents__run"}
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "agents__run"},
        }
    events = _completed_tool_events(
        request,
        native=native,
        name="agents__run",
        arguments='{"input":"hello"}' if custom else '{"message":"hello"}',
    )
    expected_type = "custom_tool_call" if custom else "function_call"
    for event in events:
        if "item" in event:
            assert event["item"]["type"] == expected_type
            assert event["item"]["name"] == "run"
            assert event["item"]["namespace"] == "agents"
    final = events[-1]["response"]["output"][0]
    assert final["type"] == expected_type
    assert final["name"] == "run"
    assert final["namespace"] == "agents"
    if custom:
        assert final["input"] == "hello"
    else:
        assert json.loads(final["arguments"]) == {"message": "hello"}
    assert request.model_dump() == original


def test_chat_does_not_turn_missing_search_arguments_into_a_successful_call() -> None:
    events = _completed_tool_events(
        OpenAIResponsesRequest(model="example", input="Search", tools=[SEARCH]),
        native=False,
        name="fcc_tool_search",
        arguments="",
    )
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["output"] == []


@pytest.mark.parametrize("native", [False, True])
def test_synthetic_search_wins_over_bare_aliases_of_real_tools(native: bool) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="Search",
        tools=[
            SEARCH,
            {
                "type": "namespace",
                "name": "editor",
                "tools": [
                    {"type": "custom", "name": "fcc_tool_search"},
                ],
            },
            {
                "type": "namespace",
                "name": "ordinary",
                "tools": [
                    {
                        "type": "function",
                        "name": "fcc_tool_search",
                        "parameters": {"type": "object"},
                    },
                ],
            },
        ],
    )
    events = _completed_tool_events(
        request, native=native, name="fcc_tool_search", arguments='{"query":"agent"}'
    )
    item = events[-1]["response"]["output"][0]
    assert item["type"] == "tool_search_call"
    assert item["arguments"] == {"query": "agent"}
    assert "namespace" not in item


@pytest.mark.parametrize("native", [False, True])
def test_malformed_namespace_is_not_silently_dropped(native: bool) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="Hello",
        tools=[
            {"type": "namespace", "name": "ordinary", "tools": {}},
        ],
    )
    with pytest.raises(ResponsesConversionError, match="list"):
        if native:
            ResponsesToolAdapter(request, ResponsesToolPolicy(flatten_namespaces=True))
        else:
            build_responses_chat_request(
                request, reasoning_replay=ReasoningReplayMode.DISABLED
            )


@pytest.mark.parametrize("native", [False, True])
def test_interleaved_tool_kinds_keep_arguments_and_terminal_output_consistent(
    native: bool,
) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="Use tools",
        tools=[
            SEARCH,
            {
                "type": "namespace",
                "name": "group",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                    {"type": "custom", "name": "edit"},
                ],
            },
        ],
    )
    names = ["fcc_tool_search", "group__lookup", "group__edit"]
    fragments = [['{"query":', '"agent"}'], ['{"n":', "8.0}"], ["plain ", "patch"]]
    events: list[dict[str, Any]]
    if native:
        adapter = _native_adapter(cast(list[JsonObject], request.tools))
        stream = adapter.event_adapter()
        assert stream is not None
        items: list[JsonObject] = [
            {
                "id": f"fc_{i}",
                "call_id": f"call_{i}",
                "type": "function_call",
                "status": "in_progress",
                "name": name,
                "arguments": "",
            }
            for i, name in enumerate(names)
        ]
        events = []
        for i, item in enumerate(items):
            events.extend(
                data
                for _, data in stream.feed(
                    "response.output_item.added", {"item": item, "output_index": i}
                )
            )
        for part in range(2):
            for i, item in enumerate(items):
                events.extend(
                    data
                    for _, data in stream.feed(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": item["id"],
                            "output_index": i,
                            "delta": fragments[i][part],
                        },
                    )
                )
        completed: list[JsonObject] = []
        for i, item in enumerate(items):
            finished: JsonObject = {
                **item,
                "status": "completed",
                "arguments": "".join(fragments[i]),
            }
            completed.append(finished)
            events.extend(
                data
                for _, data in stream.feed(
                    "response.output_item.done", {"item": finished, "output_index": i}
                )
            )
        events.extend(
            data
            for _, data in stream.feed(
                "response.completed",
                {"response": {"output": completed, "status": "completed"}},
            )
        )
    else:
        prepared = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        )
        writer = ResponsesChatStreamOutput(prepared.tool_adapter, input_tokens=1)
        frames = [*writer.start_events()]
        for i, name in enumerate(names):
            frames.append(writer.start_tool_block(i, f"call_{i}", name))
        for part in range(2):
            frames.extend(
                writer.emit_tool_delta(i, fragments[i][part]) for i in range(3)
            )
        frames.extend(
            writer.finish_success(
                stop_reason="tool_calls",
                usage=ChatStreamUsage(input_tokens=1, output_tokens=1),
            )
        )
        events = [frame.data for frame in parse_sse_text("".join(frames))]
    final = events[-1]["response"]["output"]
    assert final == [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]
    assert [item["call_id"] for item in final] == ["call_0", "call_1", "call_2"]
    assert final[0]["type"] == "tool_search_call"
    assert final[0]["arguments"] == {"query": "agent"}
    assert final[1]["type"] == "function_call"
    assert (final[1]["name"], final[1]["namespace"]) == ("lookup", "group")
    assert final[1]["arguments"] == '{"n":8}'
    assert final[2]["type"] == "custom_tool_call"
    assert (final[2]["name"], final[2]["namespace"], final[2]["input"]) == (
        "edit",
        "group",
        "plain patch",
    )
    argument_events = [
        event
        for event in events
        if event["type"].startswith("response.function_call_arguments.")
    ]
    assert {event["item_id"] for event in argument_events} == {final[1]["id"]}
    assert argument_events[-1]["arguments"] == final[1]["arguments"]
    sequences = [event["sequence_number"] for event in events]
    assert sequences == sorted(set(sequences))


def test_chat_preserves_custom_result_text_serialization() -> None:
    result = [{"type": "input_image", "image_url": "https://example.com/result.png"}]
    request = OpenAIResponsesRequest(
        model="example",
        tools=[{"type": "custom", "name": "edit"}],
        input=[
            {
                "type": "custom_tool_call",
                "call_id": "edit",
                "name": "edit",
                "input": "patch",
            },
            {"type": "custom_tool_call_output", "call_id": "edit", "output": result},
        ],
    )
    body = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    ).body
    messages = cast(list[dict[str, Any]], body["messages"])
    assert len(messages) == 2
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "edit",
        "content": json.dumps(result, separators=(",", ":")),
    }


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_arguments_cannot_complete_or_corrupt_sse(
    native: bool, search: bool, constant: str
) -> None:
    request = OpenAIResponsesRequest(
        model="example", input="Call a tool", tools=[SEARCH, AGENTS]
    )
    name = "fcc_tool_search" if search else "agents__spawn_agent"
    arguments = '{"nested":[{"limit":' + constant + "}]}"
    if native:
        with pytest.raises(ResponsesConversionError, match="arguments"):
            _completed_tool_events(request, native=True, name=name, arguments=arguments)
    else:
        events = _completed_tool_events(
            request, native=False, name=name, arguments=arguments
        )
        assert events[-1]["type"] == "response.failed"
        assert events[-1]["response"]["output"] == []
        assert not any(event["type"] == "response.output_item.done" for event in events)
        json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("native", [False, True])
def test_non_finite_spellings_remain_valid_custom_text(native: bool) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="Edit",
        tools=[
            {"type": "custom", "name": "edit"},
        ],
    )
    events = _completed_tool_events(
        request, native=native, name="edit", arguments="NaN Infinity -Infinity"
    )
    assert events[-1]["response"]["output"][0]["input"] == "NaN Infinity -Infinity"
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("search", [False, True])
def test_non_finite_spellings_in_json_strings_remain_valid(
    native: bool, search: bool
) -> None:
    request = OpenAIResponsesRequest(
        model="example", input="Call a tool", tools=[SEARCH, AGENTS]
    )
    events = _completed_tool_events(
        request,
        native=native,
        name="fcc_tool_search" if search else "agents__spawn_agent",
        arguments='{"text":"NaN Infinity -Infinity","limit":8.0}',
    )
    item = events[-1]["response"]["output"][0]
    arguments = item["arguments"] if search else json.loads(item["arguments"])
    assert arguments == {"text": "NaN Infinity -Infinity", "limit": 8}
    assert type(arguments["limit"]) is int
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize(
    "number", ["1e-400", "-1e-400", "0.12345678901234567890123456789", "1.25", "8.0"]
)
def test_argument_numbers_survive_sse_serialization(
    native: bool, search: bool, number: str
) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="Call a tool",
        tools=[
            SEARCH,
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
        ],
    )
    arguments = (
        '{"nested":{"values":[' + number + ',8.0],"text":' + json.dumps(number) + "}}"
    )
    name = "fcc_tool_search" if search else "lookup"
    if native:
        adapter = ResponsesToolAdapter(
            request,
            ResponsesToolPolicy(
                custom_tools_as_functions=True,
                flatten_namespaces=True,
                client_tool_search=True,
            ),
        )
        presenter = NativeResponsesPresenter(
            public_model="example", tool_events=adapter.event_adapter()
        )
        item: JsonObject = {
            "type": "function_call",
            "id": "fc",
            "call_id": "call",
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        completed: JsonObject = {**item, "arguments": arguments, "status": "completed"}
        frames = list(
            presenter.feed(
                "response.output_item.added", {"output_index": 0, "item": item}
            )
        )
        frames.extend(
            presenter.feed(
                "response.output_item.done", {"output_index": 0, "item": completed}
            )
        )
        frames.extend(
            presenter.feed(
                "response.completed",
                {
                    "response": {
                        "id": "resp",
                        "model": "example",
                        "status": "completed",
                        "output": [completed],
                    }
                },
            )
        )
    else:
        adapter = build_responses_chat_request(
            request, reasoning_replay=ReasoningReplayMode.DISABLED
        ).tool_adapter
        writer = ResponsesChatStreamOutput(adapter, input_tokens=1)
        frames = [
            *writer.start_events(),
            writer.start_tool_block(0, "call", name),
            writer.emit_tool_delta(0, arguments),
        ]
        frames.extend(
            writer.finish_success(
                stop_reason="tool_calls",
                usage=ChatStreamUsage(input_tokens=1, output_tokens=1),
            )
        )
    events = [
        json.loads(line[6:], parse_float=Decimal)
        for line in "".join(frames).splitlines()
        if line.startswith("data: ")
    ]
    items = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]
    items.extend(events[-1]["response"]["output"])
    assert len(items) == 2
    for item in items:
        actual = (
            item["arguments"]
            if search
            else json.loads(item["arguments"], parse_float=Decimal)
        )
        assert actual == json.loads(arguments, parse_float=Decimal)
        assert type(actual["nested"]["values"][1]) is int


@pytest.mark.parametrize("custom", [False, True])
def test_client_search_reserves_hosted_discovery_names_before_replay(
    custom: bool,
) -> None:
    tool: JsonObject = {
        "type": "custom" if custom else "function",
        "name": "fcc_tool_search",
    }
    if not custom:
        tool["parameters"] = {"type": "object"}
    request = OpenAIResponsesRequest(
        model="example",
        tools=[SEARCH],
        input=[
            {
                "type": "tool_search_call",
                "call_id": "server_search",
                "execution": "server",
                "status": "completed",
                "arguments": {},
            },
            {
                "type": "tool_search_output",
                "call_id": "server_search",
                "execution": "server",
                "status": "completed",
                "tools": [
                    tool,
                    {
                        "type": "function",
                        "name": "fcc_tool_search_1",
                        "parameters": {"type": "object"},
                    },
                ],
            },
        ],
    )
    original = request.model_dump()
    policy = ResponsesToolPolicy(
        custom_tools_as_functions=True, flatten_namespaces=True, client_tool_search=True
    )
    adapter = ResponsesToolAdapter(request, policy)
    assert adapter.request.tools and len(adapter.request.tools) == 1
    assert adapter.request.tools[0]["name"] == "fcc_tool_search_2"
    call = cast(
        dict[str, Any],
        adapter.restore_item(
            {
                "type": "function_call",
                "id": "fc",
                "call_id": "real",
                "status": "completed",
                "name": "fcc_tool_search",
                "arguments": '{"input":"text"}' if custom else "{}",
            }
        ),
    )
    assert call["type"] == ("custom_tool_call" if custom else "function_call")
    assert call["name"] == "fcc_tool_search"
    search = cast(
        dict[str, Any],
        adapter.restore_item(
            {
                "type": "function_call",
                "id": "fc_search",
                "call_id": "client_search",
                "status": "completed",
                "name": "fcc_tool_search_2",
                "arguments": '{"query":"lookup"}',
            }
        ),
    )
    assert search["type"] == "tool_search_call"
    assert search["execution"] == "client"
    continuation = request.model_copy(
        update={
            "input": [
                *cast(list[JsonObject], request.input),
                call,
                {
                    "type": "custom_tool_call_output"
                    if custom
                    else "function_call_output",
                    "call_id": "real",
                    "output": "done",
                },
            ]
        },
        deep=True,
    )
    replay = ResponsesToolAdapter(continuation, policy)
    assert (
        replay.request.tools and replay.request.tools[0]["name"] == "fcc_tool_search_2"
    )
    assert (
        cast(list[dict[str, Any]], replay.request.input)[-2]["name"]
        == "fcc_tool_search"
    )
    assert request.model_dump() == original
