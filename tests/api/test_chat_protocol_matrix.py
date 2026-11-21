"""The Chat endpoint uses each upstream through the real HTTP boundary."""

import json
from typing import Any

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject, JsonValue, load_json_value
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.test_custom_provider import completion, definition, provider
from tests.wire_payload import (
    decode_object,
    object_array,
    object_payload,
    response_object,
    string_payload,
)


@pytest.mark.parametrize(
    "wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
@pytest.mark.parametrize("streaming", [False, True])
def test_chat_http_dispatch_preserves_controls_credentials_and_public_model(
    wire: str, streaming: bool
) -> None:
    sent = []

    def reply(request: httpx.Request | httpx2.Request):
        sent.append((decode_object(request.content), dict(request.headers)))
        module = httpx if wire == "anthropic_messages" else httpx2
        return module.Response(
            200, text=completion(wire), headers={"content-type": "text/event-stream"}
        )

    configured = definition(api_format=wire, api_keys=["upstream-key"])
    instance = provider(configured, openai_handler=reply, messages_handler=reply)
    app = create_test_app(
        Settings(
            MODEL=f"{configured.provider_id}/backend",
            custom_providers=(configured,),
            PROXY_AUTH_ENABLED=True,
            proxy_auth_token="ingress-key",
        ),
        providers={configured.provider_id: instance},
    )
    payload = {
        "model": "public",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "max_completion_tokens": 37,
        "stream": streaming,
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"authorization": "Bearer ingress-key"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-request-id"]
        if streaming:
            assert response.text.endswith("data: [DONE]\n\n")
            values = [
                load_json_value(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert (
                "".join(
                    string_payload(
                        object_payload(object_payload(choice).get("delta", {})).get(
                            "content"
                        )
                        or ""
                    )
                    for value in values
                    for choice in object_array(object_payload(value).get("choices", []))
                )
                == "Hello"
            )
            assert all(object_payload(value)["model"] == "public" for value in values)
        else:
            assert response_object(response)["model"] == "public"
            assert (
                object_payload(
                    object_array(response_object(response)["choices"])[0]["message"]
                )["content"]
                == "Hello"
            )
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(sent) == 1
    body, headers = sent[0]
    assert body["model"] == "backend"
    assert body["temperature"] == 0
    limit = {
        "openai_chat": "max_completion_tokens",
        "openai_responses": "max_output_tokens",
        "anthropic_messages": "max_tokens",
    }[wire]
    assert body[limit] == 37
    assert "ingress-key" not in json.dumps(headers)
    assert headers[
        "x-api-key" if wire == "anthropic_messages" else "authorization"
    ].endswith("upstream-key")


@pytest.mark.parametrize("wire", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize(
    "strict", [{}, {"strict": None}, {"strict": False}, {"strict": True}]
)
def test_chat_function_strictness_survives_upstream_translation(
    wire: str, strict: JsonObject
) -> None:
    sent = []

    def reply(request: httpx2.Request):
        sent.append(decode_object(request.content))
        return httpx2.Response(
            200, text=completion(wire), headers={"content-type": "text/event-stream"}
        )

    configured = definition(api_format=wire)
    instance = provider(configured, openai_handler=reply)
    app = create_test_app(
        Settings(
            MODEL=f"{configured.provider_id}/backend",
            custom_providers=(configured,),
        ),
        providers={configured.provider_id: instance},
    )
    function = {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"] if strict.get("strict") else [],
            "additionalProperties": False,
        },
        **strict,
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": [{"type": "function", "function": function}],
            },
        )
        assert response.status_code == 200, response.text
    assert len(sent) == 1
    if wire == "openai_chat":
        assert sent[0]["tools"] == [{"type": "function", "function": function}]
    else:
        assert sent[0]["tools"] == [
            {"type": "function", **function, "strict": strict.get("strict") is True}
        ]


def test_chat_validation_and_auth_errors_use_openai_shape() -> None:
    app = create_test_app(
        Settings(PROXY_AUTH_ENABLED=True, proxy_auth_token="ingress-key")
    )
    with TestClient(app) as client:
        unauthenticated = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauthenticated.status_code == 401
        assert "error" in unauthenticated.json()
        invalid = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
            headers={"authorization": "Bearer ingress-key"},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("with_tool", [False, True])
def test_native_chat_null_tool_delta_preserves_text_and_later_calls(
    streaming: bool, with_tool: bool
) -> None:
    from tests.api.test_four_protocol_matrix import matrix_app, response_value

    call = {
        "index": 0,
        "id": "call_read",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    deltas: list[JsonObject] = [
        {"role": "assistant", "content": "hello", "tool_calls": None},
    ]
    if with_tool:
        deltas.append({"tool_calls": [{**call, "function": None}]})
    deltas.append({"tool_calls": [call] if with_tool else None})

    def reply(_):
        events = [
            {
                "id": "chatcmpl-a",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "backend",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": ("tool_calls" if with_tool else "stop")
                        if index == len(deltas) - 1
                        else None,
                    }
                ],
            }
            for index, delta in enumerate(deltas)
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": streaming,
            },
        )
        value = response_value(response, "chat", streaming)
    message = value["choices"][0]["message"]
    assert message["content"] == "hello"
    assert message["tool_calls"] == (
        [{key: item for key, item in call.items() if key != "index"}]
        if with_tool
        else None
    )
    if streaming:
        first = decode_object(response.text.splitlines()[0][6:])
        assert (
            object_payload(object_array(first["choices"])[0]["delta"])["tool_calls"]
            is None
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"n": 0},
        {"n": "two"},
        {"max_completion_tokens": "many"},
        {"stream_options": ["usage"]},
        {"tools": [1]},
        {"tools": [{"type": "function", "function": {}}]},
        {"tool_choice": "bogus"},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "id": "x", "function": {"arguments": "{}"}}
                    ],
                }
            ]
        },
        {"messages": [{"role": "assistant", "reasoning_details": ["bad"]}]},
    ],
)
def test_chat_malformed_fields_fail_before_provider_execution(
    changes: JsonObject,
) -> None:
    app = create_test_app(Settings())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hi"}],
                **changes,
            },
        )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("field", "updates", "expected"),
    [
        (
            "function_call",
            [
                {"name": None, "arguments": None},
                {"name": "read", "arguments": "{"},
                {"name": None, "arguments": "}"},
            ],
            {"name": "read", "arguments": "{}"},
        ),
        (
            "audio",
            [
                {"id": "audio_a", "data": None, "transcript": None},
                {"data": "YQ==", "transcript": "hello"},
                {"id": None, "data": None, "transcript": None, "expires_at": 123},
            ],
            {"id": "audio_a", "data": "YQ==", "transcript": "hello", "expires_at": 123},
        ),
        (
            "reasoning_details",
            [
                [{"type": "reasoning.text", "text": "one"}],
                [{"type": "reasoning.text", "text": " two"}],
            ],
            [{"type": "reasoning.text", "text": "one two"}],
        ),
        ("citations", [["source-a"], ["source-b"]], ["source-a", "source-b"]),
    ],
)
def test_native_chat_nullable_fields_accumulate_later_values(
    streaming: bool, field: str, updates: list[JsonValue], expected: JsonValue
) -> None:
    from tests.api.test_four_protocol_matrix import matrix_app, response_value

    def reply(_):
        deltas = [
            {"role": "assistant", field: None},
            *[{field: value} for value in updates],
            {field: None, "content": "done"},
        ]
        events = [
            {
                "id": "chatcmpl-a",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "backend",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": "stop" if index == len(deltas) - 1 else None,
                    }
                ],
            }
            for index, delta in enumerate(deltas)
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": streaming,
            },
        )
        value = response_value(response, "chat", streaming)
    assert value["choices"][0]["message"][field] == expected
    if streaming:
        first = decode_object(response.text.splitlines()[0][6:])
        assert object_payload(object_array(first["choices"])[0]["delta"])[field] is None


@pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning:pydantic.main"
)
@pytest.mark.parametrize("choices", [[], {}])
def test_native_chat_usage_trailer_completes_stream(choices: JsonValue) -> None:
    from tests.api.test_four_protocol_matrix import matrix_app
    from tests.providers.test_openai_chat_transport import _success

    usage = {"prompt_tokens": 22, "completion_tokens": 4, "total_tokens": 26}

    def reply(_: httpx2.Request) -> httpx2.Response:
        return _success(usage_trailer={"choices": choices, "usage": usage})

    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert response.status_code == 200, response.text
    trailer, done = response.text.strip().split("\n\n")[-2:]
    assert done == "data: [DONE]"
    payload = decode_object(trailer.removeprefix("data: "))
    assert payload["choices"] == []
    assert payload["usage"] == usage


@pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning:pydantic.main"
)
@pytest.mark.parametrize("choices", [{}, ""])
@pytest.mark.parametrize("metadata_first", [False, True])
def test_native_chat_keeps_metadata_only_chunks(
    choices: JsonValue, metadata_first: bool
) -> None:
    from tests.api.test_four_protocol_matrix import matrix_app

    envelope = {
        "id": "chatcmpl-metadata",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "backend",
    }
    filter_results = [{"prompt_index": 0, "content_filter_results": {}}]
    metadata = {
        **envelope,
        "choices": choices,
        "prompt_filter_results": filter_results,
    }
    content = {
        **envelope,
        "choices": [
            {"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}
        ],
    }
    events = [metadata, content] if metadata_first else [content, metadata]

    def reply(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n",
        )

    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
    assert response.status_code == 200, response.text
    assert response.text.endswith("data: [DONE]\n\n"), response.text
    frames = response.text.strip().split("\n\n")
    payload = decode_object(frames[0 if metadata_first else 1].removeprefix("data: "))
    assert payload["choices"] == []
    assert payload["prompt_filter_results"] == filter_results


@pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning:pydantic.main"
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("empty_array", [{}, ""])
def test_native_chat_preserves_text_with_empty_delta_arrays(
    streaming: bool, empty_array: JsonValue
) -> None:
    from tests.api.test_delivered_stream_recovery import text_events
    from tests.api.test_four_protocol_matrix import matrix_app, response_value

    events = text_events("chat", "hello")
    delta = object_payload(object_array(events[0]["choices"])[0]["delta"])
    delta.update(tool_calls=empty_array, reasoning_details=empty_array)

    def reply(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n",
        )

    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": streaming,
            },
        )
        result = object_payload(response_value(response, "chat", streaming))
    message = object_payload(object_array(result["choices"])[0]["message"])
    assert message["content"] == "hello"
    assert message["tool_calls"] == []
    assert message["reasoning_details"] == []


@pytest.mark.parametrize("streaming", [False, True])
def test_native_custom_tool_fragments_survive_public_chat_response(
    streaming: bool,
) -> None:
    from tests.api.test_four_protocol_matrix import matrix_app, response_value

    chunks = [
        {
            "index": 0,
            "id": "call_custom",
            "type": "custom",
            "custom": {"name": "code_exec", "input": "print("},
        },
        {"index": 0, "custom": {"input": "1)"}},
    ]

    def reply(_):
        events: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-a",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "backend",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [chunk]},
                        "finish_reason": None,
                    }
                ],
            }
            for chunk in chunks
        ]
        events.append(
            {
                "id": "chatcmpl-a",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "backend",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("chat", sdk_reply=reply)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "run"}],
                "stream": streaming,
                "tools": [{"type": "custom", "custom": {"name": "code_exec"}}],
            },
        )
        value = response_value(response, "chat", streaming)
    assert value["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_custom",
            "type": "custom",
            "custom": {"name": "code_exec", "input": "print(1)"},
        }
    ]
