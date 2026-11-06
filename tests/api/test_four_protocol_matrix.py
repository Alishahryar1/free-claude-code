"""Every ingress/egress pair survives a tool turn with protected reasoning."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import patch

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_custom_provider import definition, provider
from tests.providers.test_history_transports import _native
from tests.providers.test_native_tool_arguments import tool_events

SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}
GEMINI_PARTS = [
    {"text": "Find 17.", "thought": True},
    {
        "functionCall": {"id": "call_probe", "name": "read", "args": {"path": "kept"}},
        "thoughtSignature": "native-google-signature",
    },
]


@pytest.mark.parametrize("tool_type", [None, "custom", "web_search_20250305"])
def test_messages_custom_tool_type_is_translated_to_native_gemini(tool_type):
    from tests.providers.test_gemini import _sse

    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return _sse([{"text": "hello"}])

    tool = {"name": "read", "description": "Read a file", "input_schema": SCHEMA}
    if tool_type is not None:
        tool["type"] = tool_type
    with TestClient(matrix_app("gemini", reply)) as client:
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


def source_events(protocol):
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
                event["index"] += 1
        return [
            events[0],
            {"type": "content_block_start", "index": 0, "content_block": native},
            {"type": "content_block_stop", "index": 0},
            *tail,
        ]
    for event in events:
        if "output_index" in event:
            event["output_index"] += 1
    events[-1]["response"]["output"].insert(0, native)
    events.insert(
        1, {"type": "response.output_item.done", "output_index": 0, "item": native}
    )
    return [{**event, "sequence_number": index} for index, event in enumerate(events)]


def matrix_app(protocol, reply, *, max_attempts=2):
    if protocol == "gemini":
        instance = GeminiProvider(
            make_provider_config(api_key="upstream-key", base_url=GEMINI_DEFAULT_BASE),
            admission=immediate_admission(max_attempts=max_attempts),
            transport=httpx.MockTransport(reply),
        )
        provider_id = "gemini"
        settings = Settings(MODEL="gemini/models/backend")
    elif protocol == "chat":
        client = AsyncOpenAI(
            api_key="upstream-key",
            base_url="https://upstream.invalid/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
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
        settings = Settings(MODEL="open_router/backend")
    else:
        configured = definition(
            api_format="anthropic_messages"
            if protocol == "messages"
            else "openai_responses",
            api_keys=["upstream-key"],
        )
        instance = provider(configured, openai_handler=reply, messages_handler=reply)
        provider_id = configured.provider_id
        settings = Settings(
            MODEL=f"{provider_id}/backend", custom_providers=(configured,)
        )
    return create_test_app(settings, providers={provider_id: instance})


def ingress_request(wire, streaming, history=None):
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
    body = {
        "model": "public",
        "stream": streaming,
        "tools": tools,
        "input" if wire == "responses" else "messages": history
        or [{"role": "user", "content": "hello"}],
    }
    if wire == "messages":
        body["max_tokens"] = 100
    return "/v1/chat/completions" if wire == "chat" else f"/v1/{wire}", body


def response_value(response, wire, streaming):
    assert response.status_code == 200, response.text
    if not streaming:
        return response.json()
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


def reply_history(value, wire):
    initial = {"role": "user", "content": "hello"}
    if wire == "messages":
        call = next(block for block in value["content"] if block["type"] == "tool_use")
        assert call["input"] == {"path": "kept"}
        return [
            initial,
            {"role": "assistant", "content": value["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": "tool-value",
                    }
                ],
            },
        ]
    if wire == "responses":
        call = next(item for item in value["output"] if item["type"] == "function_call")
        assert json.loads(call["arguments"]) == {"path": "kept"}
        return [
            initial,
            *value["output"],
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "tool-value",
            },
        ]
    if wire == "chat":
        message = value["choices"][0]["message"]
        call = message["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {"path": "kept"}
        return [
            initial,
            message,
            {"role": "tool", "tool_call_id": call["id"], "content": "tool-value"},
        ]
    content = value["candidates"][0]["content"]
    call = next(
        part["functionCall"] for part in content["parts"] if "functionCall" in part
    )
    assert call["args"] == {"path": "kept"}
    return [
        {"role": "user", "parts": [{"text": "hello"}]},
        content,
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": call["id"],
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
    protocol, wire, streaming
):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
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

    app = matrix_app(protocol, reply)
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
        assert bodies[1]["contents"][1] == {"role": "model", "parts": GEMINI_PARTS}
    elif protocol == "messages":
        replayed = next(
            block
            for block in bodies[1]["messages"][1]["content"]
            if block.get("type") == "thinking"
        )
        expected = _native(protocol)
        # The existing Messages -> Responses codec retains signed fields, but
        # has never retained arbitrary block extensions. New paths retain both.
        if wire == "responses":
            expected.pop("extension")
        assert replayed == expected
    elif protocol == "responses":
        assert _native(protocol) in bodies[1]["input"]
    else:
        assistant = next(
            message
            for message in bodies[1]["messages"]
            if message["role"] == "assistant"
        )
        if wire in {"chat", "gemini"}:
            assert (
                assistant["reasoning_details"] == _native(protocol)["reasoning_details"]
            )
        else:
            # Existing Messages/Responses presentations expose Chat's opaque
            # bytes separately from its readable scalar representation.
            assert any(
                detail.get("data") == "opaque-original"
                for detail in assistant["reasoning_details"]
            )
            assert json.dumps(assistant).count("Find 17.") == 1
