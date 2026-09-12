import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesStreamFailure,
    ResponsesToolAdapter,
    ResponsesToolPolicy,
    build_responses_chat_request,
)
from free_claude_code.providers.openai_responses.presentation import (
    NativeResponsesPresenter,
)


def _adapter(request: OpenAIResponsesRequest) -> ResponsesToolAdapter:
    return ResponsesToolAdapter(request, ResponsesToolPolicy(True, True, True))


def _presenter(adapter: ResponsesToolAdapter) -> NativeResponsesPresenter:
    return NativeResponsesPresenter(
        public_model="example", tool_events=adapter.event_adapter()
    )


def test_custom_tool_identity_and_history_preserve_namespaces() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input=[
            {
                "type": "custom_tool_call",
                "id": "custom",
                "call_id": "call_custom",
                "namespace": "editor",
                "name": "edit",
                "input": "patch",
            },
            {"type": "reasoning", "id": "reasoning", "encrypted_content": "opaque"},
        ],
        tools=[
            {"type": "custom", "name": "edit", "format": {"type": "text"}},
            {
                "type": "namespace",
                "name": "editor",
                "tools": [
                    {"type": "custom", "name": "edit", "format": {"type": "text"}}
                ],
            },
            {
                "type": "namespace",
                "name": "ordinary",
                "tools": [
                    {
                        "type": "function",
                        "name": "edit",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ],
        tool_choice={"type": "custom", "namespace": "editor", "name": "edit"},
    )
    original = request.model_dump()
    adapter = _adapter(request)
    wire = adapter.request.model_dump()
    nested = wire["tools"][1]["tools"][0]
    assert nested["type"] == "function"
    assert wire["input"][0]["name"] == nested["name"]
    assert wire["input"][0]["namespace"] == "editor"
    assert wire["tool_choice"] == {
        "type": "function",
        "namespace": "editor",
        "name": nested["name"],
    }
    assert wire["input"][1] == original["input"][1]

    output = [
        {
            "type": "function_call",
            "name": nested["name"],
            "namespace": "editor",
            "call_id": "one",
            "arguments": '{"input":"patch"}',
        },
        {
            "type": "function_call",
            "name": "edit",
            "namespace": "ordinary",
            "call_id": "two",
            "arguments": "{}",
        },
    ]
    event: JsonObject = {
        "type": "response.completed",
        "sequence_number": 10,
        "response": {
            "id": "resp_one",
            "model": "example",
            "status": "completed",
            "output": output,
            "tools": wire["tools"],
        },
    }
    stream = "".join(_presenter(adapter).feed("response.completed", event))
    restored = parse_sse_text(stream)[0].data["response"]
    assert restored["output"][0]["type"] == "custom_tool_call"
    assert restored["output"][0]["name"] == "edit"
    assert restored["output"][0]["namespace"] == "editor"
    assert restored["output"][0]["input"] == "patch"
    assert restored["output"][1] == output[1]
    assert restored["tools"] == original["tools"]
    assert request.model_dump() == original


def test_image_only_web_search_is_not_silently_replaced_with_text_search() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[{"type": "web_search", "search_content_types": ["image"]}],
    )
    with pytest.raises(ResponsesConversionError, match="text web search only"):
        _adapter(request)


def test_custom_and_function_names_cannot_collide_after_conversion() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[
            {"type": "custom", "name": "edit"},
            {"type": "function", "name": "edit", "parameters": {"type": "object"}},
        ],
    )
    with pytest.raises(ResponsesConversionError, match="names collide"):
        _adapter(request)


@pytest.mark.parametrize(
    "field,value", [("defer_loading", True), ("allowed_callers", ["programmatic"])]
)
def test_custom_conversion_preserves_tool_settings(field: str, value: object) -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "example",
            "input": "hello",
            "tools": [
                {
                    "type": "custom",
                    "name": "edit",
                    "format": {"type": "text"},
                    field: value,
                }
            ],
        }
    )
    converted = _adapter(request).request.model_dump()["tools"][0]
    assert converted[field] == value


def test_client_discovery_converts_customs_and_preserves_each_definition() -> None:
    first = {
        "type": "namespace",
        "name": "editor",
        "description": "Original group",
        "tools": [
            {
                "type": "custom",
                "name": "edit",
                "description": "First definition",
                "defer_loading": True,
            }
        ],
    }
    second = {
        **first,
        "tools": [
            {
                **first["tools"][0],
                "description": "Current definition",
                "defer_loading": False,
            }
        ],
    }
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "example",
            "tools": [second],
            "input": [
                {
                    "type": "tool_search_output",
                    "call_id": "search_one",
                    "execution": "client",
                    "status": "completed",
                    "tools": [first],
                },
                {
                    "type": "tool_search_output",
                    "call_id": "search_two",
                    "execution": "client",
                    "status": "completed",
                    "tools": [second],
                },
            ],
        }
    )
    original = request.model_dump()
    adapter = _adapter(request)
    wire = adapter.request.model_dump()
    assert wire["input"][0]["tools"][0]["tools"][0]["type"] == "function"
    assert wire["input"][1]["tools"][0]["tools"][0]["type"] == "function"
    assert wire["input"][0]["tools"][0]["description"] == "Original group"
    event: JsonObject = {
        "type": "response.completed",
        "response": {
            "id": "response",
            "status": "completed",
            "model": "example",
            "output": wire["input"],
            "tools": wire["tools"],
        },
    }
    stream = "".join(_presenter(adapter).feed("response.completed", event))
    restored = parse_sse_text(stream)[0].data["response"]
    assert restored["output"] == original["input"]
    assert restored["tools"] == original["tools"]
    assert request.model_dump() == original


def test_flattened_custom_call_without_namespace_is_restored() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[
            {
                "type": "namespace",
                "name": "editor",
                "tools": [{"type": "custom", "name": "edit"}],
            }
        ],
    )
    adapter = _adapter(request)
    wire = adapter.request.model_dump()
    event: JsonObject = {
        "type": "response.completed",
        "response": {
            "id": "response",
            "status": "completed",
            "model": "example",
            "output": [
                {
                    "type": "function_call",
                    "id": "item_one",
                    "call_id": "call_one",
                    "name": wire["tools"][0]["tools"][0]["name"],
                    "arguments": '{"input":"patch"}',
                }
            ],
        },
    }
    stream = "".join(_presenter(adapter).feed("response.completed", event))
    item = parse_sse_text(stream)[0].data["response"]["output"][0]
    assert item == {
        "type": "custom_tool_call",
        "id": "item_one",
        "call_id": "call_one",
        "name": "edit",
        "namespace": "editor",
        "input": "patch",
    }


@pytest.mark.parametrize(
    "namespace,expected",
    [
        (None, "function_call"),
        ("other", "function_call"),
        ("editor", "custom_tool_call"),
    ],
)
def test_omitted_namespace_cannot_misidentify_an_ordinary_function(
    namespace: str | None, expected: str
) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[
            {
                "type": "namespace",
                "name": "editor",
                "tools": [{"type": "custom", "name": "edit"}],
            },
            {
                "type": "function",
                "name": "editor__edit",
                "parameters": {"type": "object"},
            },
        ],
    )
    adapter = _adapter(request)
    call: JsonObject = {
        "type": "function_call",
        "name": "editor__edit",
        "call_id": "one",
        "arguments": '{"input":"value"}',
    }
    if namespace is not None:
        call["namespace"] = namespace
    restored = adapter.restore_item(call)
    assert isinstance(restored, dict)
    assert restored["type"] == expected
    if expected == "function_call":
        assert restored == call


def test_default_policy_preserves_native_tools_and_history() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        tools=[
            {
                "type": "custom",
                "name": "edit",
                "defer_loading": True,
                "extra": {"x": 1},
            },
            {"type": "web_search", "search_content_types": ["image"]},
        ],
        input=[
            {
                "type": "custom_tool_call",
                "call_id": "one",
                "name": "edit",
                "input": "patch",
            }
        ],
    )
    adapter = ResponsesToolAdapter(request, ResponsesToolPolicy())
    assert adapter.request == request
    assert adapter.request is not request
    assert adapter.event_adapter() is None


def test_custom_conversion_keeps_unrelated_fields_and_restores_allowed_choices() -> (
    None
):
    tool: JsonObject = {
        "type": "custom",
        "name": "edit",
        "format": {"type": "text"},
        "defer_loading": False,
        "allowed_callers": ["direct"],
        "extension": {"a": [1, 2]},
    }
    choice: JsonObject = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "custom", "name": "edit"}],
    }
    request = OpenAIResponsesRequest(
        model="example", input="hello", tools=[tool], tool_choice=choice
    )
    adapter = _adapter(request)
    wire = adapter.request.model_dump()
    for field in ("defer_loading", "allowed_callers", "extension"):
        assert wire["tools"][0][field] == tool[field]
    assert adapter.restore_choice(wire["tool_choice"]) == choice
    echoed = [{**wire["tools"][0], "provider_extension": "keep"}]
    assert adapter.restore_tools(echoed) == [{**tool, "provider_extension": "keep"}]


def test_separate_attempts_do_not_share_custom_item_ids_or_sequences() -> None:
    adapter = _adapter(
        OpenAIResponsesRequest(
            model="example",
            input="hello",
            tools=[
                {"type": "custom", "name": "edit"},
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            ],
        )
    )
    first, second = adapter.event_adapter(), adapter.event_adapter()
    assert first is not None and second is not None
    list(
        first.feed(
            "response.output_item.added",
            {
                "sequence_number": 100,
                "item": {
                    "type": "function_call",
                    "id": "reused",
                    "name": "edit",
                    "call_id": "first",
                    "arguments": "",
                },
            },
        )
    )
    ordinary: JsonObject = {"sequence_number": 0, "item_id": "reused", "delta": "{}"}
    result = list(second.feed("response.function_call_arguments.delta", ordinary))
    assert result == [
        (
            "response.function_call_arguments.delta",
            {**ordinary, "type": "response.function_call_arguments.delta"},
        )
    ]


def test_already_native_custom_calls_pass_through_without_extra_input_events() -> None:
    adapter = _adapter(
        OpenAIResponsesRequest(
            model="example", input="hello", tools=[{"type": "custom", "name": "edit"}]
        )
    ).event_adapter()
    assert adapter is not None
    item: JsonObject = {
        "type": "custom_tool_call",
        "id": "native",
        "call_id": "call_native",
        "name": "native_tool",
        "input": "text",
        "status": "completed",
    }
    done: JsonObject = {
        "type": "response.output_item.done",
        "sequence_number": 4,
        "output_index": 0,
        "item": item,
    }
    assert list(adapter.feed("response.output_item.done", done)) == [
        ("response.output_item.done", done)
    ]


@pytest.mark.parametrize(
    "format_value,description",
    [
        ({"type": "text"}, "Custom tool input format: unconstrained text."),
        (
            {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
            "Custom tool input format: grammar (lark): start: /.+/",
        ),
        ({"type": "future"}, "Custom tool input format: future."),
    ],
)
def test_chat_and_native_adaptation_share_custom_input_and_description(
    format_value: JsonObject, description: str
) -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[{"type": "custom", "name": "edit", "format": format_value}],
    )
    native = _adapter(request).request.model_dump()["tools"][0]
    chat_tools = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    ).body["tools"]
    assert isinstance(chat_tools, list)
    chat = chat_tools[0]["function"]
    assert native["parameters"] == chat["parameters"]
    assert native["description"] == chat["description"] == description


@pytest.mark.parametrize("upstream_failure", [False, True])
def test_adapted_presenter_preserves_terminal_failure_and_original_tool_metadata(
    upstream_failure: bool,
) -> None:
    tool: JsonObject = {
        "type": "custom",
        "name": "edit",
        "format": {"type": "text"},
        "defer_loading": True,
    }
    adapter = _adapter(
        OpenAIResponsesRequest(model="example", input="hello", tools=[tool])
    )
    presenter = _presenter(adapter)
    response: JsonObject = {
        "id": "resp_one",
        "model": "upstream",
        "status": "in_progress",
        "tools": adapter.request.model_dump()["tools"],
        "output": [],
    }
    list(
        presenter.feed(
            "response.created",
            {"type": "response.created", "sequence_number": 0, "response": response},
        )
    )
    list(
        presenter.feed(
            "response.output_text.delta", {"sequence_number": 1, "delta": "partial"}
        )
    )
    failure = ExecutionFailure(FailureKind.UPSTREAM, 502, "failed", False)
    raw: Exception = RuntimeError("stream ended")
    if upstream_failure:
        raw = ResponsesStreamFailure(
            "failed",
            event_type="response.failed",
            payload={
                "type": "response.failed",
                "sequence_number": 2,
                "response": {
                    **response,
                    "status": "failed",
                    "error": {"message": "failed"},
                },
            },
        )
    events = parse_sse_text("".join(presenter.terminal_failure(raw, failure)))
    assert len(events) == 1 and events[0].event == "response.failed"
    assert events[0].data["sequence_number"] == 2
    assert events[0].data["response"]["tools"] == [tool]
    assert events[0].data["response"]["status"] == "failed"
    assert presenter.completed
