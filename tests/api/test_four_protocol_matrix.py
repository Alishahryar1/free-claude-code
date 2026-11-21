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
):
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


@pytest.mark.parametrize(
    ("protocol", "required"),
    [
        ("gemini", "native_reply"),
        ("messages", "native_reply"),
        ("chat", "sdk_reply"),
        ("responses", "sdk_reply"),
    ],
)
def test_matrix_requires_the_handler_for_its_protocol(protocol: str, required: str):
    def unexpected_request(request: httpx.Request | httpx2.Request) -> Never:
        raise AssertionError("A mismatched handler was invoked")

    with pytest.raises(ValueError, match=required):
        if required == "native_reply":
            matrix_app(protocol, sdk_reply=unexpected_request)
        else:
            matrix_app(protocol, native_reply=unexpected_request)


def ingress_request(
    wire: str, streaming: bool, history: Sequence[Mapping[str, JsonValue]] | None = None
):
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


@pytest.mark.parametrize("streaming", [False, True])
def test_gemini_tool_call_survives_whole_valued_candidate_index(streaming: bool):
    events = source_events("gemini")
    candidate = object_array(events[0]["candidates"])[0]
    candidate["index"] = 0.0

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
    content = object_payload(object_array(result["candidates"])[0]["content"])
    assert any(
        part.get("functionCall") == GEMINI_PARTS[1]["functionCall"]
        for part in object_array(content["parts"])
    )


def test_gemini_backed_chat_rejects_nonstring_tool_result_id():
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
@pytest.mark.parametrize("index_case", ["output_index", "content_index", "unused"])
def test_responses_text_translation_preserves_whole_valued_indexes(
    wire: str, index_case: str
):
    events = text_events("responses", "Hello")
    if index_case == "unused":
        events[0]["output_index"] = None
        events[-1]["output_index"] = None
    else:
        delta = next(
            event for event in events if event["type"] == "response.output_text.delta"
        )
        delta[index_case] = 0.0

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


@pytest.mark.parametrize("wire", ["chat", "gemini", "messages", "responses"])
@pytest.mark.parametrize(
    "event_type", ["message_start", "message_delta", "message_stop"]
)
def test_messages_lifecycle_ignores_unused_block_index(wire: str, event_type: str):
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
):
    bodies = []

    def reply(request: httpx.Request | httpx2.Request):
        bodies.append(decode_object(request.content))
        events = source_events(protocol)
        raw = "".join(
            (f"event: {event['type']}\n" if protocol == "messages" else "")
            + f"data: {json.dumps(event)}\n\n"
            for event in events
        )
        response = (
            httpx.Response if protocol in {"messages", "gemini"} else httpx2.Response
        )
        return response(200, headers={"content-type": "text/event-stream"}, text=raw)

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
