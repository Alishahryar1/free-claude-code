"""Every ingress/egress pair survives a tool turn with protected reasoning."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Never
from unittest.mock import patch

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.api.support import create_test_app, provider_manager_for_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.providers.support import (
    MockHandler,
    immediate_admission,
    make_provider_config,
    mock_response,
)
from tests.providers.test_custom_provider import definition, provider
from tests.providers.test_history_transports import _native
from tests.providers.test_native_tool_arguments import tool_events
from tests.wire_payload import (
    array_payload,
    decode_object,
    int_payload,
    object_array,
    object_payload,
    response_object,
    response_payload,
    string_payload,
)

SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}
GEMINI_PARTS = [
    {"text": "Find 17.", "thought": True},
    {
        "functionCall": {"id": "call_probe", "name": "read", "args": {"path": "kept"}},
        "thoughtSignature": "native-google-signature",
    },
]


@pytest.mark.parametrize("tool_type", [None, "custom", "web_search_20250305"])
def test_messages_custom_tool_type_is_translated_to_native_gemini(
    tool_type: str | None,
) -> None:
    from tests.providers.test_gemini import _sse

    bodies = []

    def reply(request: httpx.Request | httpx2.Request):
        bodies.append(decode_object(request.content))
        return _sse([{"text": "hello"}])

    tool = {"name": "read", "description": "Read a file", "input_schema": SCHEMA}
    if tool_type is not None:
        tool["type"] = tool_type
    with TestClient(matrix_app("gemini", native_reply=reply)) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 128,
                "tools": [tool],
            },
        )
    if tool_type == "web_search_20250305":
        assert response.status_code == 400
        assert not bodies
    else:
        assert response.status_code == 200, response.text
        assert bodies[0]["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "read",
                        "description": "Read a file",
                        "parametersJsonSchema": SCHEMA,
                    }
                ]
            }
        ]


def source_events(protocol: str):
    if protocol == "gemini":
        return [
            {
                "responseId": "native-response",
                "modelVersion": "upstream",
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": GEMINI_PARTS},
                        "finishReason": "STOP",
                    }
                ],
            }
        ]
    events = tool_events(protocol, '{"path":"kept"}')
    native = _native(protocol)
    if protocol == "chat":
        return [
            {
                **events[0],
                "choices": [{"index": 0, "delta": native, "finish_reason": None}],
            },
            *events,
        ]
    if protocol == "messages":
        tail = deepcopy(events[1:])
        for event in tail:
            if "index" in event:
                event["index"] = int_payload(event["index"]) + 1
        return [
            events[0],
            {"type": "content_block_start", "index": 0, "content_block": native},
            {"type": "content_block_stop", "index": 0},
            *tail,
        ]
    for event in events:
        if "output_index" in event:
            event["output_index"] = int_payload(event["output_index"]) + 1
    array_payload(object_payload(events[-1]["response"])["output"]).insert(0, native)
    events.insert(
        1, {"type": "response.output_item.done", "output_index": 0, "item": native}
    )
    return [{**event, "sequence_number": index} for index, event in enumerate(events)]


def matrix_reply(protocol: str, events: list[JsonObject] | None = None):
    raw = "".join(
        (f"event: {event['type']}\n" if protocol == "messages" else "")
        + f"data: {json.dumps(event)}\n\n"
        for event in (source_events(protocol) if events is None else events)
    )
    response = httpx.Response if protocol in {"messages", "gemini"} else httpx2.Response
    return response(200, headers={"content-type": "text/event-stream"}, text=raw)


def matrix_app(
    protocol: str,
    *,
    native_reply: MockHandler[httpx.Request] | None = None,
    sdk_reply: MockHandler[httpx2.Request] | None = None,
    max_attempts: int = 2,
    model_fallbacks: list[ModelFallback] | None = None,
):
    if protocol in {"gemini", "messages"}:
        if native_reply is None:
            raise ValueError(f"{protocol} requires native_reply")
    elif sdk_reply is None:
        raise ValueError(f"{protocol} requires sdk_reply")

    async def native_handler(request: httpx.Request) -> httpx.Response:
        assert native_reply is not None
        return await mock_response(native_reply(request), httpx.Response)

    async def sdk_handler(request: httpx2.Request) -> httpx2.Response:
        assert sdk_reply is not None
        return await mock_response(sdk_reply(request), httpx2.Response)

    if protocol == "gemini":
        instance = GeminiProvider(
            make_provider_config(api_key="upstream-key", base_url=GEMINI_DEFAULT_BASE),
            admission=immediate_admission(max_attempts=max_attempts),
            transport=httpx.MockTransport(native_handler),
        )
        provider_id = "gemini"
        settings = Settings(MODEL="gemini/models/backend")
    elif protocol == "chat":
        client = AsyncOpenAI(
            api_key="upstream-key",
            base_url="https://upstream.invalid/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(sdk_handler)),
        )
        with patch(
            "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
            return_value=client,
        ):
            instance = OpenRouterProvider(
                make_provider_config(
                    api_key="upstream-key", base_url="https://upstream.invalid/v1"
                ),
                admission=immediate_admission(max_attempts=max_attempts),
            )
        provider_id = "open_router"
        settings = Settings(
            MODEL="open_router/backend", MODEL_FALLBACKS=model_fallbacks
        )
    else:
        configured = definition(
            api_format="anthropic_messages"
            if protocol == "messages"
            else "openai_responses",
            api_keys=["upstream-key"],
        )
        instance = provider(
            configured, openai_handler=sdk_reply, messages_handler=native_reply
        )
        provider_id = configured.provider_id
        settings = Settings(
            MODEL=f"{provider_id}/backend", custom_providers=(configured,)
        )
    return create_test_app(settings, providers={provider_id: instance})


def translation_error(protocol: str, path: str, body: JsonObject) -> str:
    def unexpected(request: httpx.Request | httpx2.Request) -> Never:
        pytest.fail(f"An untranslatable request reached {request.url.path}")

    with TestClient(
        matrix_app(protocol, native_reply=unexpected, sdk_reply=unexpected)
    ) as client:
        result = client.post(path, json=body)
    assert result.status_code == 400, result.text
    return string_payload(object_payload(response_object(result)["error"])["message"])


@pytest.mark.parametrize(
    ("protocol", "required"),
    [
        ("gemini", "native_reply"),
        ("messages", "native_reply"),
        ("chat", "sdk_reply"),
        ("responses", "sdk_reply"),
    ],
)
def test_matrix_requires_the_handler_for_its_protocol(
    protocol: str, required: str
) -> None:
    def unexpected_request(request: httpx.Request | httpx2.Request) -> Never:
        raise AssertionError("A mismatched handler was invoked")

    with pytest.raises(ValueError, match=required):
        if required == "native_reply":
            matrix_app(protocol, sdk_reply=unexpected_request)
        else:
            matrix_app(protocol, native_reply=unexpected_request)


def ingress_request(
    wire: str, streaming: bool, history: Sequence[Mapping[str, JsonValue]] | None = None
) -> tuple[str, JsonObject]:
    tools = (
        [{"name": "read", "input_schema": SCHEMA}]
        if wire == "messages"
        else [
            {"type": "function", "function": {"name": "read", "parameters": SCHEMA}}
            if wire == "chat"
            else {"type": "function", "name": "read", "parameters": SCHEMA}
        ]
    )
    if wire == "gemini":
        method = "streamGenerateContent?alt=sse" if streaming else "generateContent"
        return f"/v1beta/models/public:{method}", {
            "contents": history or [{"role": "user", "parts": [{"text": "hello"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "read", "parametersJsonSchema": SCHEMA}
                    ]
                }
            ],
            "generationConfig": {"thinkingConfig": {"includeThoughts": True}},
        }
    body: JsonObject = {
        "model": "public",
        "stream": streaming,
        "tools": tools,
        "input" if wire == "responses" else "messages": history
        or [{"role": "user", "content": "hello"}],
    }
    if wire == "messages":
        body["max_tokens"] = 100
    return "/v1/chat/completions" if wire == "chat" else f"/v1/{wire}", body


@pytest.mark.parametrize("wire", ["chat", "messages", "responses", "gemini"])
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_request_extensions_follow_the_selected_wire(wire: str, protocol: str) -> None:
    bodies: list[JsonObject] = []

    def reply(request: httpx.Request | httpx2.Request):
        bodies.append(decode_object(request.content))
        return matrix_reply(protocol)

    parts_key = "parts" if wire == "gemini" else "content"
    turn_key = {"gemini": "contents", "responses": "input"}.get(wire, "messages")
    part: JsonObject = {"text": "hello"}
    if wire != "gemini":
        part["type"] = "input_text" if wire == "responses" else "text"
    path, baseline = ingress_request(
        wire, wire == "responses", [{"role": "user", parts_key: [part]}]
    )
    extended = deepcopy(baseline)
    extended.update(vendor_null=None, vendor_option={"enabled": False})
    turn = object_array(extended[turn_key])[0]
    turn["vendor_turn"] = 0
    if wire == "chat":
        turn["function_call"] = {"name": "client_metadata"}
    parts = array_payload(turn[parts_key])
    object_payload(parts[0])["vendor_part"] = True
    future: JsonObject = {"futureData": {"text": "opaque"}}
    if wire != "gemini":
        future["type"] = "future_data"
    parts.append(future)
    tool = object_array(extended["tools"])[0]
    if wire == "gemini":
        tool = object_array(tool["functionDeclarations"])[0]
    elif wire == "chat":
        tool = object_payload(tool["function"])
    tool["vendor_declaration"] = 17
    if wire == "messages":
        array_payload(extended["tools"]).append({"type": "future_hosted"})
    if wire == "responses":
        extended["messages"] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "fcc:history:invalid"}
                ],
            }
        ]
    if wire in {"chat", "responses", "messages"}:
        extended["prompt_cache_key"] = "cache-group"
    with TestClient(
        matrix_app(protocol, native_reply=reply, sdk_reply=reply)
    ) as client:
        for body in (baseline, extended):
            result = client.post(path, json=body)
            assert result.status_code == 200, result.text
    original, sent = bodies
    if wire == protocol:
        assert sent["vendor_null"] is None
        assert sent["vendor_option"] == {"enabled": False}
        sent_turn = object_array(sent[turn_key])[0]
        assert sent_turn["vendor_turn"] == 0
        sent_parts = object_array(sent_turn[parts_key])
        assert sent_parts[0]["vendor_part"] is True
        assert sent_parts[1] == future
        declaration = object_array(sent["tools"])[0]
        if wire == "gemini":
            declaration = object_array(declaration["functionDeclarations"])[0]
        elif wire == "chat":
            declaration = object_payload(declaration["function"])
        assert declaration["vendor_declaration"] == 17
        if wire == "messages":
            assert object_array(sent["tools"])[1] == {"type": "future_hosted"}
        if wire == "responses":
            assert sent["messages"] == extended["messages"]
        if wire != "gemini":
            assert sent["prompt_cache_key"] == "cache-group"
    elif {wire, protocol} == {"chat", "responses"}:
        assert sent == {**original, "prompt_cache_key": "cache-group"}
    else:
        assert sent == original


def response_value(response: httpx.Response, wire: str, streaming: bool):
    assert response.status_code == 200, response.text
    if not streaming:
        return response_payload(response)
    events = parse_sse_text(response.text)
    if wire == "responses":
        assert events[-1].event == "response.completed", response.text
        return events[-1].data["response"]
    if wire == "messages":

        async def stream():
            yield response.text

        message, error, stopped = asyncio.run(
            aggregate_anthropic_sse_to_message(stream())
        )
        assert error is None and stopped, response.text
        return message
    accumulator = (
        ChatResponseAccumulator() if wire == "chat" else GeminiResponseAccumulator()
    )
    for event in events:
        accumulator.feed(event.data)
    if wire == "chat":
        assert response.text.endswith("data: [DONE]\n\n"), response.text
        assert isinstance(accumulator, ChatResponseAccumulator)
        accumulator.finish()
    return accumulator.result()


@pytest.mark.parametrize(
    "text_controls",
    [
        "unsupported",
        ["unsupported"],
        "",
        [],
        False,
        0,
        {"format": 123},
        {"format": []},
        {"format": ""},
        {"format": False},
        {"format": {}},
        {"format": {"type": "json_schema"}},
        {"format": {"type": "json_schema", "schema": "invalid"}},
    ],
)
def test_gemini_rejects_invalid_responses_text_controls(
    text_controls: JsonValue,
) -> None:
    def unexpected_request(_: httpx.Request) -> Never:
        raise AssertionError("Invalid text controls must be rejected before inference")

    app = matrix_app("gemini", native_reply=unexpected_request)
    path, body = ingress_request("responses", True)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(path, json={**body, "text": text_controls})
    assert response.status_code == 400, response.text
    assert (
        object_payload(response_object(response)["error"])["type"]
        == "invalid_request_error"
    )


@pytest.mark.parametrize(
    ("text_controls", "expected_config"),
    [
        (None, {}),
        ({}, {}),
        ({"format": None}, {}),
        ({"format": {"type": "text"}}, {}),
        ({"format": {"type": "json_object"}}, {"responseMimeType": "application/json"}),
        (
            {"format": {"type": "json_schema", "name": "answer", "schema": SCHEMA}},
            {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA},
        ),
    ],
)
def test_gemini_preserves_valid_responses_text_controls(
    text_controls: JsonValue, expected_config: JsonObject
) -> None:
    from tests.providers.test_gemini import _sse

    sent: list[JsonObject] = []

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(decode_object(request.content))
        return _sse([{"text": '{"answer":"yes"}'}])

    app = matrix_app("gemini", native_reply=reply)
    path, body = ingress_request("responses", True)
    with TestClient(app) as client:
        response = client.post(path, json={**body, "text": text_controls})
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    config = object_payload(sent[0]["generationConfig"])
    assert {
        field: config[field]
        for field in ("responseMimeType", "responseJsonSchema")
        if field in config
    } == expected_config


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish_reason", ["STOP", 1, 1.0])
def test_gemini_tool_call_survives_compatible_candidate_fields(
    streaming: bool, finish_reason: str | int | float
) -> None:
    events = source_events("gemini")
    candidate = object_array(events[0]["candidates"])[0]
    candidate["index"] = 0.0
    candidate["finishReason"] = finish_reason

    def reply(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(events[0])}\n\n",
        )

    with TestClient(matrix_app("gemini", native_reply=reply)) as client:
        path, body = ingress_request("gemini", streaming)
        result = object_payload(
            response_value(client.post(path, json=body), "gemini", streaming)
        )
    received = object_array(result["candidates"])[0]
    assert received["finishReason"] == finish_reason
    assert type(received["finishReason"]) is type(finish_reason)
    content = object_payload(received["content"])
    assert any(
        part.get("functionCall") == GEMINI_PARTS[1]["functionCall"]
        for part in object_array(content["parts"])
    )


def test_gemini_backed_chat_rejects_nonstring_tool_result_id() -> None:
    def unexpected_request(_: httpx.Request) -> Never:
        raise AssertionError("Invalid tool history reached the upstream provider")

    history = [
        {"role": "user", "content": "read"},
        {"role": "tool", "tool_call_id": 1, "content": "tool result"},
    ]
    with TestClient(matrix_app("gemini", native_reply=unexpected_request)) as client:
        path, body = ingress_request("chat", False, history)
        response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    assert (
        object_payload(response_object(response)["error"])["type"]
        == "invalid_request_error"
    )


@pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning:pydantic.main"
)
@pytest.mark.parametrize("wire", ["chat", "gemini"])
@pytest.mark.parametrize(
    ("index_case", "index_value"),
    [
        ("output_index", 0.0),
        ("content_index", 0.0),
        ("output_index", "0"),
        ("content_index", "0"),
        ("output_index", "+0"),
        ("content_index", "0.0"),
        ("output_index", None),
        ("content_index", None),
        ("unused", None),
        ("unused_summary_index", None),
    ],
)
def test_responses_text_translation_preserves_provider_index_encodings(
    wire: str, index_case: str, index_value: JsonValue
) -> None:
    events = text_events("responses", "Hello")
    if index_case == "unused":
        events[0]["output_index"] = None
        events[-1]["output_index"] = None
    elif index_case == "unused_summary_index":
        for event in events:
            if event["type"] in {
                "response.content_part.added",
                "response.content_part.done",
            }:
                event["summary_index"] = None
    else:
        for event in events:
            if index_case in event:
                event[index_case] = index_value

    def reply(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("responses", sdk_reply=reply)) as client:
        path, body = ingress_request(wire, False)
        result = object_payload(
            response_value(client.post(path, json=body), wire, False)
        )

    if wire == "chat":
        choice = object_array(result["choices"])[0]
        assert object_payload(choice["message"])["content"] == "Hello"
        assert object_payload(result["usage"])["total_tokens"] == 10
    else:
        candidate = object_array(result["candidates"])[0]
        parts = object_array(object_payload(candidate["content"])["parts"])
        assert (
            "".join(string_payload(part.get("text", "")) for part in parts) == "Hello"
        )
        assert object_payload(result["usageMetadata"])["totalTokenCount"] == 10


@pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning:pydantic.main"
)
@pytest.mark.parametrize(
    "snapshot", ["response.created", "response.completed", "response.failed"]
)
@pytest.mark.parametrize("empty_output", [{}, ""])
def test_native_responses_preserves_lifecycle_with_empty_output(
    snapshot: str, empty_output: JsonValue
) -> None:
    events = text_events("responses", "Hello")
    if snapshot == "response.failed":
        events[-1]["type"] = snapshot
        failure = object_payload(events[-1]["response"])
        failure.update(
            status="failed",
            error={"code": "server_error", "message": "Upstream ended the request"},
        )
    event = next(event for event in events if event["type"] == snapshot)
    object_payload(event["response"])["output"] = empty_output

    def reply(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("responses", sdk_reply=reply)) as client:
        path, body = ingress_request("responses", True)
        response = client.post(path, json=body)

    assert response.status_code == 200, response.text
    streamed = parse_sse_text(response.text)
    assert streamed[0].event == "response.created"
    assert object_payload(streamed[0].data["response"])["output"] == []
    assert [
        frame.data["delta"]
        for frame in streamed
        if frame.event == "response.output_text.delta"
    ] == ["Hello"]
    final = streamed[-1]
    if snapshot == "response.failed":
        assert final.event == snapshot
        error = object_payload(object_payload(final.data["response"])["error"])
        assert "Upstream ended the request" in string_payload(error["message"])
    else:
        assert final.event == "response.completed"


@pytest.mark.parametrize("wire", ["chat", "gemini", "messages", "responses"])
@pytest.mark.parametrize("empty_parts", [{}, ""])
def test_gemini_finishes_with_empty_provider_parts(
    wire: str, empty_parts: JsonValue
) -> None:
    events = [
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "Hello"}]}}
            ]
        },
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": empty_parts},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 2,
                "totalTokenCount": 10,
            },
        },
    ]

    def reply(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("gemini", native_reply=reply)) as client:
        path, body = ingress_request(wire, True)
        result = response_value(client.post(path, json=body), wire, True)
    assert "Hello" in json.dumps(result)


@pytest.mark.parametrize("wire", ["chat", "gemini", "messages", "responses"])
@pytest.mark.parametrize(
    "event_type", ["message_start", "message_delta", "message_stop"]
)
def test_messages_lifecycle_ignores_unused_block_index(
    wire: str, event_type: str
) -> None:
    events = text_events("messages", "Hello")
    event = next(event for event in events if event["type"] == event_type)
    event["index"] = None

    def reply(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ),
        )

    with TestClient(matrix_app("messages", native_reply=reply)) as client:
        path, body = ingress_request(wire, True)
        result = response_value(client.post(path, json=body), wire, True)
    assert "Hello" in json.dumps(result)


def reply_history(value: JsonObject, wire: str):
    initial = {"role": "user", "content": "hello"}
    if wire == "messages":
        call = next(
            block
            for block in object_array(value["content"])
            if object_payload(block)["type"] == "tool_use"
        )
        assert object_payload(call)["input"] == {"path": "kept"}
        return [
            initial,
            {"role": "assistant", "content": value["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": object_payload(call)["id"],
                        "content": "tool-value",
                    }
                ],
            },
        ]
    if wire == "responses":
        call = next(
            item
            for item in object_array(value["output"])
            if object_payload(item)["type"] == "function_call"
        )
        assert json.loads(string_payload(object_payload(call)["arguments"])) == {
            "path": "kept"
        }
        return [
            initial,
            *object_array(value["output"]),
            {
                "type": "function_call_output",
                "call_id": object_payload(call)["call_id"],
                "output": "tool-value",
            },
        ]
    if wire == "chat":
        message = object_payload(array_payload(value["choices"])[0])["message"]
        call = array_payload(object_payload(message)["tool_calls"])[0]
        assert json.loads(
            string_payload(
                object_payload(object_payload(call)["function"])["arguments"]
            )
        ) == {"path": "kept"}
        return [
            initial,
            message,
            {
                "role": "tool",
                "tool_call_id": object_payload(call)["id"],
                "content": "tool-value",
            },
        ]
    content = object_payload(array_payload(value["candidates"])[0])["content"]
    call = next(
        object_payload(part)["functionCall"]
        for part in object_array(object_payload(content)["parts"])
        if "functionCall" in part
    )
    assert object_payload(call)["args"] == {"path": "kept"}
    return [
        {"role": "user", "parts": [{"text": "hello"}]},
        content,
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": object_payload(call)["id"],
                        "name": "read",
                        "response": {"output": "tool-value"},
                    }
                }
            ],
        },
    ]


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
@pytest.mark.parametrize(
    ("wire", "streaming"),
    [
        (wire, stream)
        for wire in ("messages", "responses", "chat", "gemini")
        for stream in (False, True)
        if wire != "responses" or stream
    ],
)
def test_all_sixteen_paths_replay_native_state_and_tool_results(
    protocol: str, wire: str, streaming: bool
) -> None:
    bodies = []

    def reply(request: httpx.Request | httpx2.Request):
        bodies.append(decode_object(request.content))
        return matrix_reply(protocol)

    app = matrix_app(protocol, native_reply=reply, sdk_reply=reply)
    with TestClient(app) as client:
        path, body = ingress_request(wire, streaming)
        first = response_value(client.post(path, json=body), wire, streaming)
        assert "Find 17." in json.dumps(first)
        path, body = ingress_request(wire, streaming, reply_history(first, wire))
        response_value(client.post(path, json=body), wire, streaming)
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(bodies) == 2
    assert "tool-value" in json.dumps(bodies[1])
    if protocol == "gemini":
        assert array_payload(bodies[1]["contents"])[1] == {
            "role": "model",
            "parts": GEMINI_PARTS,
        }
    elif protocol == "messages":
        replayed = next(
            block
            for block in array_payload(
                object_array(bodies[1]["messages"])[1]["content"]
            )
            if object_payload(block).get("type") == "thinking"
        )
        expected = _native(protocol)
        # The existing Messages -> Responses codec retains signed fields, but
        # has never retained arbitrary block extensions. New paths retain both.
        if wire == "responses":
            expected.pop("extension")
        assert replayed == expected
    elif protocol == "responses":
        assert _native(protocol) in array_payload(bodies[1]["input"])
    else:
        assistant = next(
            message
            for message in array_payload(bodies[1]["messages"])
            if object_payload(message)["role"] == "assistant"
        )
        if wire in {"chat", "gemini"}:
            assert (
                object_payload(assistant)["reasoning_details"]
                == _native(protocol)["reasoning_details"]
            )
        else:
            # Existing Messages/Responses presentations expose Chat's opaque
            # bytes separately from its readable scalar representation.
            assert any(
                object_payload(detail).get("data") == "opaque-original"
                for detail in array_payload(
                    object_payload(assistant)["reasoning_details"]
                )
            )
            assert json.dumps(assistant).count("Find 17.") == 1


@pytest.mark.parametrize(
    ("wire", "protocol"),
    [
        (wire, protocol)
        for wire in ("chat", "messages", "responses", "gemini")
        for protocol in ("chat", "messages", "responses", "gemini")
        if wire != protocol
    ],
)
def test_unsupported_latest_user_turn_never_replays_an_older_prompt(
    wire: str, protocol: str
) -> None:
    key = "parts" if wire == "gemini" else "content"
    future: JsonObject = (
        {"futureMedia": {"text": "opaque"}}
        if wire == "gemini"
        else {"type": "future_media", "text": "opaque"}
    )
    known: JsonObject = {"text": "earlier"}
    if wire != "gemini":
        known["type"] = "input_text" if wire == "responses" else "text"
    path, body = ingress_request(
        wire,
        wire == "responses",
        [
            {"role": "user", key: [known]},
            {
                "role": "user",
                key: [future],
                **({"reasoning_content": "metadata"} if wire == "chat" else {}),
            },
        ],
    )
    assert "latest user" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "gemini"])
def test_gemini_saved_input_and_literal_sdk_fields_follow_the_wire(
    protocol: str,
) -> None:
    requests = []

    def upstream(request: httpx.Request | httpx2.Request):
        requests.append(request)
        return matrix_reply(protocol)

    path, body = ingress_request("gemini", False)
    extras: JsonObject = {
        "model": "other",
        "extra_headers": {"x-goog-api-key": "client-value"},
    }
    with TestClient(
        matrix_app(protocol, native_reply=upstream, sdk_reply=upstream)
    ) as client:
        result = client.post(
            path,
            json={
                **body,
                "extra_body": extras,
                "cachedContent": "cachedContents/context",
            },
        )
    if protocol == "gemini":
        assert result.status_code == 200, result.text
        (request,) = requests
        assert request.url.path.endswith("/models/backend:streamGenerateContent")
        assert request.headers["x-goog-api-key"] == "upstream-key"
        sent = decode_object(request.content)
        assert sent["cachedContent"] == "cachedContents/context"
        assert sent["extra_body"] == extras
    else:
        assert result.status_code == 400, result.text
        assert result.json()["error"]["status"] == "INVALID_ARGUMENT"
        assert "cachedContent" in result.text
        assert not requests


@pytest.mark.parametrize("protocol", ["chat", "responses"])
def test_messages_required_output_schema_survives_translation(protocol: str) -> None:
    bodies = []

    def upstream(request: httpx.Request | httpx2.Request):
        bodies.append(decode_object(request.content))
        return matrix_reply(protocol)

    path, body = ingress_request("messages", False)
    schema = {
        "type": "object",
        "properties": {"future_field": {"type": "null"}},
        "additionalProperties": False,
    }
    with TestClient(matrix_app(protocol, sdk_reply=upstream)) as client:
        result = client.post(
            path,
            json={
                **body,
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            },
        )
    assert result.status_code == 200, result.text
    (sent,) = bodies
    fmt = (
        object_payload(object_payload(sent["response_format"])["json_schema"])
        if protocol == "chat"
        else object_payload(object_payload(sent["text"])["format"])
    )
    assert fmt["schema"] == schema
    assert fmt["strict"] is True


@pytest.mark.parametrize("protocol", ["chat", "responses", "gemini"])
def test_forcing_omitted_hosted_tool_fails_before_upstream(protocol: str) -> None:
    requests = []

    def upstream(request: httpx.Request | httpx2.Request):
        requests.append(request)
        return matrix_reply(protocol)

    path, body = ingress_request("messages", False)
    tools = [
        {"type": "future_hosted", "name": "upstream_only"},
        *array_payload(body["tools"]),
    ]
    with TestClient(
        matrix_app(protocol, native_reply=upstream, sdk_reply=upstream)
    ) as client:
        result = client.post(
            path,
            json={
                **body,
                "tools": tools,
                "tool_choice": {"type": "tool", "name": "upstream_only"},
            },
        )
    assert result.status_code == 400, result.text
    assert (
        object_payload(response_object(result)["error"])["type"]
        == "invalid_request_error"
    )
    assert not requests


@pytest.mark.parametrize("wire", ["gemini", "chat"])
def test_partial_gemini_upstream_call_never_becomes_a_complete_tool_call(
    wire: str,
) -> None:
    from tests.providers.test_gemini import _sse

    def reply(request: httpx.Request):
        return _sse(
            [
                {
                    "functionCall": {
                        "name": "read",
                        "partialArgs": [
                            {"jsonPath": "$.path", "stringValue": "unfinished"}
                        ],
                        "willContinue": True,
                    }
                }
            ]
        )

    path, body = ingress_request(wire, False)
    with TestClient(matrix_app("gemini", native_reply=reply, max_attempts=1)) as client:
        result = client.post(path, json=body)
    assert result.status_code >= 400, result.text
    assert "error" in response_object(result)


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini"])
@pytest.mark.parametrize(
    "saved_state",
    [
        {"conversation": "conv_1"},
        {"prompt": {"id": "pmpt_1"}},
        {"previous_response_id": "resp_1"},
        {"type": "compaction", "encrypted_content": "opaque"},
        {"type": "item_reference", "id": "msg_1"},
        {"type": "mcp_approval_request", "id": "approval_1", "name": "read"},
        {
            "type": "mcp_approval_response",
            "approval_request_id": "approval_1",
            "approve": False,
        },
    ],
)
def test_foreign_provider_cannot_resolve_responses_context(
    protocol: str, saved_state: JsonObject
) -> None:
    path, body = ingress_request("responses", True)
    if "type" in saved_state:
        array_payload(body["input"]).insert(0, saved_state)
    else:
        body.update(saved_state)

    assert "context" in translation_error(protocol, path, body).lower()


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini"])
def test_programmatic_only_function_is_unavailable_to_direct_callers(
    protocol: str,
) -> None:
    bodies: list[JsonObject] = []

    def reply(request: httpx.Request | httpx2.Request):
        bodies.append(decode_object(request.content))
        return matrix_reply(protocol)

    path, body = ingress_request("responses", True)
    array_payload(body["tools"]).append(
        {"type": "function", "name": "estimate", "allowed_callers": ["programmatic"]}
    )
    array_payload(body["tools"]).append(
        {
            "type": "namespace",
            "name": "math",
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "allowed_callers": ["programmatic"],
                }
            ],
        }
    )
    with TestClient(
        matrix_app(protocol, native_reply=reply, sdk_reply=reply)
    ) as client:
        result = client.post(path, json=body)
        assert result.status_code == 200, result.text
        for choice in (
            {"type": "function", "name": "estimate"},
            {"type": "function", "name": "read", "namespace": "math"},
        ):
            denied = client.post(path, json={**body, "tool_choice": choice})
            assert denied.status_code == 400, denied.text
    (sent,) = bodies
    declarations = json.dumps(sent["tools"])
    assert '"read"' in declarations
    assert '"estimate"' not in declarations


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_chat_discovery_respects_the_current_responses_allowlist(mode: str) -> None:
    bodies: list[JsonObject] = []

    def reply(request: httpx2.Request):
        bodies.append(decode_object(request.content))
        return matrix_reply("chat", tool_events("chat", "{}", name="count"))

    choice: JsonObject = {
        "type": "allowed_tools",
        "mode": mode,
        "tools": [{"type": "function", "name": "count"}],
    }
    path, body = ingress_request(
        "responses",
        True,
        [
            {"role": "user", "content": "Find a tool"},
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "search",
                "arguments": {"query": "estimate"},
            },
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": "search",
                "tools": [
                    {"type": "function", "name": "estimate", "parameters": SCHEMA}
                ],
            },
            {"role": "user", "content": "Count"},
        ],
    )
    body.update(
        tools=[{"type": "function", "name": "count", "parameters": SCHEMA}],
        tool_choice=choice,
    )
    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        result = client.post(path, json=body)
    assert result.status_code == 200, result.text
    (sent,) = bodies
    assert [
        object_payload(tool["function"])["name"] for tool in object_array(sent["tools"])
    ] == ["count"]
    assert sent["tool_choice"] == mode
    messages = object_array(sent["messages"])
    call = next(
        object_array(message["tool_calls"])[0]
        for message in messages
        if message.get("tool_calls")
    )
    output = next(message for message in messages if message["role"] == "tool")
    assert call["id"] == output["tool_call_id"] == "search"
    assert "estimate" in string_payload(output["content"])
    echoed = {
        event.event: object_payload(event.data["response"])["tool_choice"]
        for event in parse_sse_text(result.text)
        if event.event in ("response.created", "response.completed")
    }
    assert echoed == {"response.created": choice, "response.completed": choice}


@pytest.mark.parametrize("protocol", ["messages", "responses", "gemini"])
@pytest.mark.parametrize(
    "fields",
    [
        {"modalities": ["text", "audio"]},
        {"function_call": {"name": "read"}},
        {"function_call": "none", "tool_choice": "required"},
        {
            "tool_choice": {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "function", "function": {"name": ["read"]}}],
                },
            }
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "extra_content": {"google": {"thought_signature": "opaque"}},
                },
                {"role": "user", "content": "continue"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Read"},
                {
                    "role": "assistant",
                    "function_call": {"name": "read", "arguments": "{}"},
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Read"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                            "extra_content": {
                                "google": {"thought_signature": "opaque"}
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c", "content": "done"},
            ]
        },
    ],
)
def test_chat_requirements_survive_or_fail_before_inference(
    protocol: str, fields: JsonObject
) -> None:
    path, body = ingress_request("chat", False)
    translation_error(protocol, path, {**body, **fields})


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize(
    "fragment",
    [
        {"willContinue": True},
        {
            "willContinue": False,
            "partialArgs": [{"jsonPath": "$.count", "numberValue": 1}],
        },
    ],
)
def test_gemini_history_requires_a_complete_function_call(
    protocol: str, fragment: JsonObject
) -> None:
    path, body = ingress_request(
        "gemini",
        False,
        [
            {"role": "user", "parts": [{"text": "Read"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "read", **fragment}}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read",
                            "response": {"output": "done"},
                        }
                    }
                ],
            },
        ],
    )
    assert "functionCall" in translation_error(protocol, path, body)


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_nonblocking_gemini_result_cannot_be_closed_early(protocol: str) -> None:
    path, body = ingress_request(
        "gemini",
        False,
        [
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "read", "args": {}}}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read",
                            "response": {"value": 1},
                            "willContinue": True,
                        }
                    }
                ],
            },
        ],
    )
    object_array(object_array(body["tools"])[0]["functionDeclarations"])[0][
        "behavior"
    ] = "NON_BLOCKING"
    assert "functionResponse" in translation_error(protocol, path, body)


def sent_request(protocol: str, path: str, body: JsonObject) -> JsonObject:
    sent: list[JsonObject] = []

    def reply(request: httpx.Request | httpx2.Request):
        sent.append(decode_object(request.content))
        return matrix_reply(protocol)

    with TestClient(
        matrix_app(protocol, native_reply=reply, sdk_reply=reply)
    ) as client:
        response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    (upstream,) = sent
    return upstream


@pytest.mark.parametrize("protocol", ["messages", "chat", "responses", "gemini"])
def test_messages_caller_permissions_follow_the_tool(protocol: str) -> None:
    path, body = ingress_request("messages", True)
    declarations = [
        {
            "name": "read",
            "input_schema": SCHEMA,
            "allowed_callers": ["direct", "code_execution_20260120"],
        },
        {
            "name": "container_only",
            "input_schema": SCHEMA,
            "allowed_callers": ["code_execution_20260120"],
        },
    ]
    body["tools"] = declarations
    upstream = sent_request(protocol, path, body)
    if protocol == "messages":
        assert upstream["tools"] == declarations
    else:
        assert '"read"' in json.dumps(upstream["tools"])
        assert "container_only" not in json.dumps(upstream["tools"])
        translation_error(
            protocol,
            path,
            {**body, "tool_choice": {"type": "tool", "name": "container_only"}},
        )


@pytest.mark.parametrize("protocol", ["chat", "messages", "gemini"])
@pytest.mark.parametrize("execution", ["client", "server"])
def test_deferred_function_is_callable_only_after_discovery(
    protocol: str, execution: str
) -> None:
    path, body = ingress_request("responses", True)
    deferred: JsonObject = {
        "type": "function",
        "name": "lookup",
        "description": "current schema",
        "parameters": SCHEMA,
        "defer_loading": True,
    }
    search: JsonObject = {
        "type": "tool_search",
        "execution": execution,
        "parameters": SCHEMA,
    }
    body["tools"] = [*array_payload(body["tools"]), deferred, search]
    choice: JsonObject = {"type": "function", "name": "lookup"}
    before = sent_request(protocol, path, body)
    assert '"lookup"' not in json.dumps(before["tools"])
    translation_error(protocol, path, {**body, "tool_choice": choice})
    call_id = "search_1" if execution == "client" else None
    history: list[JsonValue] = [
        {"role": "user", "content": "Find a tool"},
        {
            "type": "tool_search_call",
            "execution": execution,
            "status": "completed",
            "call_id": call_id,
            "arguments": {"query": "lookup"},
        },
        {
            "type": "tool_search_output",
            "execution": execution,
            "status": "completed",
            "call_id": call_id,
            "tools": [{**deferred, "description": "old schema"}],
        },
        {"role": "user", "content": "Use it"},
    ]
    body.update(input=history, tool_choice=choice)
    after = sent_request(protocol, path, body)
    assert "current schema" in json.dumps(after["tools"])
    assert "old schema" not in json.dumps(after["tools"])
    object_payload(history[2])["status"] = "failed"
    translation_error(protocol, path, body)


def test_forcing_a_deferred_function_cannot_select_the_private_search_tool() -> None:
    path, body = ingress_request("responses", True)
    body.update(
        tools=[
            {
                "type": "function",
                "name": "fcc_tool_search",
                "parameters": SCHEMA,
                "defer_loading": True,
            },
            {"type": "tool_search", "execution": "client", "parameters": SCHEMA},
        ],
        tool_choice={"type": "function", "name": "fcc_tool_search"},
    )
    translation_error("chat", path, body)


@pytest.mark.parametrize("protocol", ["chat", "responses", "gemini"])
def test_completed_messages_code_execution_survives_intervening_client_tool(
    protocol: str,
) -> None:
    call: JsonObject = {
        "type": "server_tool_use",
        "id": "srv_1",
        "name": "bash_code_execution",
        "input": {"command": "historical_command"},
    }
    result: JsonObject = {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": "srv_1",
        "content": {
            "type": "bash_code_execution_result",
            "stdout": "historical_output",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
    path, body = ingress_request(
        "messages",
        True,
        [
            {"role": "user", "content": "Read"},
            {
                "role": "assistant",
                "content": [
                    call,
                    {"type": "tool_use", "id": "tool_1", "name": "read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "client result",
                    }
                ],
            },
            {"role": "assistant", "content": [result]},
            {"role": "user", "content": "Summarize"},
        ],
    )
    upstream = json.dumps(sent_request(protocol, path, body))
    assert (
        upstream.index("historical_command")
        < upstream.index("client result")
        < upstream.index("historical_output")
    )
    assert "[Earlier tool record]" in upstream
    result.update(type="web_search_tool_result", content=[])
    translation_error(protocol, path, body)
