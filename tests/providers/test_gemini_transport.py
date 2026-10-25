"""Direct Gemini ingress through the real upstream transport adapters."""

import json

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.gemini.client import _PROFILE as GOOGLE_PROFILE
from free_claude_code.providers.google_openai.provider import GoogleChatBehavior
from free_claude_code.providers.openai_chat import OpenAIChatTransport
from tests.providers.support import immediate_admission
from tests.providers.test_anthropic_messages_transport import (
    Endpoint,
)
from tests.providers.test_anthropic_messages_transport import (
    _events as messages_events,
)
from tests.providers.test_anthropic_messages_transport import (
    _sse as messages_sse,
)
from tests.providers.test_anthropic_messages_transport import (
    _transport as messages_transport,
)
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events
from tests.providers.test_openai_chat_transport import _transport
from tests.providers.test_openai_responses_transport import (
    _sse as responses_sse,
)
from tests.providers.test_openai_responses_transport import (
    _transport as responses_transport,
)


def gemini_request(**values):
    return GeminiRequest.model_validate(
        {"model": "model", "contents": [{"parts": [{"text": "hello"}]}], **values}
    )


def chat_response(deltas, *, finish="stop", usage=None):
    chunks = [
        {
            "id": "upstream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    chunks.append(
        {
            "id": "upstream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            **({"usage": usage} if usage is not None else {}),
        }
    )
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n",
    )


async def collect(stream):
    aggregate = GeminiResponseAccumulator()
    chunks = [chunk async for chunk in stream]
    for event in parse_sse_text("".join(chunks)):
        aggregate.feed(event.data)
    return aggregate.result(), chunks


@pytest.mark.asyncio
async def test_chat_gemini_stream_preserves_text_identity_and_reported_usage():
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return chat_response(
            [{"content": "hello "}, {"content": "world"}],
            usage={
                "prompt_tokens": 4,
                "completion_tokens": 8,
                "total_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        )

    async with AsyncOpenAI(
        api_key="synthetic",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        value, chunks = await collect(
            _transport(client).stream_gemini(gemini_request(), response_model="public")
        )
    assert value["modelVersion"] == "public"
    assert value["candidates"][0]["content"]["parts"] == [{"text": "hello world"}]
    assert value["usageMetadata"] == {
        "promptTokenCount": 4,
        "candidatesTokenCount": 5,
        "thoughtsTokenCount": 3,
        "totalTokenCount": 12,
    }
    assert sent[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert all("event:" not in chunk for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "[]", "null"])
async def test_chat_gemini_never_releases_invalid_executable_arguments(arguments):
    def reply(request):
        return chat_response(
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "read", "arguments": arguments},
                        }
                    ]
                }
            ],
            finish="tool_calls",
        )

    async with AsyncOpenAI(
        api_key="synthetic",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        request = gemini_request(tools=[{"functionDeclarations": [{"name": "read"}]}])
        with pytest.raises(ExecutionFailure):
            async for chunk in _transport(client).stream_gemini(request):
                assert "functionCall" not in chunk


@pytest.mark.asyncio
async def test_messages_gemini_text_and_native_thinking_replay():
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        events = messages_events()
        events[1:1] = [
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "thinking_delta", "thinking": "private thought"},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "signature_delta", "signature": "signed-native"},
            },
            {"type": "content_block_stop", "index": 2},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=messages_sse(*events),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        value, _ = await collect(
            messages_transport(client).stream_gemini(
                gemini_request(), endpoint_context=Endpoint(), response_model="public"
            )
        )
    parts = value["candidates"][0]["content"]["parts"]
    assert parts[0] == {"text": "hello"}
    assert not any(part.get("thought") for part in parts)
    bundle = decode_bundle(parts[-1]["thoughtSignature"])
    assert bundle is not None
    assert bundle.records[0].native == {
        "type": "thinking",
        "thinking": "private thought",
        "signature": "signed-native",
    }
    assert bundle.records[0].origin.protocol == "messages"
    assert sent[0]["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_responses_gemini_text_complete_call_and_late_reasoning_replay():
    sent = []
    reasoning = {
        "id": "rs_one",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thought"}],
        "encrypted_content": "opaque",
    }
    message = {
        "id": "msg_one",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "answer", "annotations": []}],
    }
    call = {
        "id": "fc_one",
        "type": "function_call",
        "call_id": "call-one",
        "status": "completed",
        "name": "read",
        "arguments": '{"path":"a"}',
    }

    def reply(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=responses_sse(
                {
                    "type": "response.output_text.delta",
                    "output_index": 1,
                    "content_index": 0,
                    "item_id": "msg_one",
                    "delta": "answer",
                    "sequence_number": 0,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 2,
                    "item": call,
                    "sequence_number": 1,
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "upstream",
                        "status": "completed",
                        "output": [reasoning, message, call],
                        "usage": None,
                    },
                    "sequence_number": 2,
                },
            ),
        )

    async with AsyncOpenAI(
        api_key="synthetic",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        request = gemini_request(tools=[{"functionDeclarations": [{"name": "read"}]}])
        value, _ = await collect(
            responses_transport(client).stream_gemini(
                request,
                input_tokens=20,
                request_id="test",
                response_model="public",
                reasoning=ReasoningPolicy.provider_default(),
            )
        )
    parts = value["candidates"][0]["content"]["parts"]
    assert [part for part in parts if "text" in part] == [{"text": "answer"}]
    assert [part["functionCall"] for part in parts if "functionCall" in part] == [
        {"id": "call-one", "name": "read", "args": {"path": "a"}}
    ]
    bundle = decode_bundle(parts[-1]["thoughtSignature"])
    assert bundle is not None and bundle.records[0].native == reasoning
    assert "usageMetadata" not in value
    assert sent[0]["input"][0]["role"] == "user"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("same_connection", [False, True])
async def test_google_replay_survives_client_replacement_and_preserves_edited_args(
    tool_call, same_connection
):
    sent = []
    opaque = {"google": {"thought_signature": "native-signed-bytes"}}

    def reply(request):
        sent.append(json.loads(request.content))
        if len(sent) > 1:
            return chat_response([{"content": "continued"}])
        if tool_call:
            return chat_response(
                [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-original",
                                "type": "function",
                                "extra_content": opaque,
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"original"}',
                                },
                            }
                        ]
                    }
                ],
                finish="tool_calls",
            )
        return chat_response([{"content": "answer", "extra_content": opaque}])

    async def execute(request, key):
        async with AsyncOpenAI(
            api_key=key,
            base_url="https://provider.invalid/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
        ) as client:
            transport = OpenAIChatTransport(
                client=client,
                admission=immediate_admission(),
                behavior=GoogleChatBehavior(GOOGLE_PROFILE),
                read_timeout_s=2,
                log_raw_sse_events=False,
                log_api_error_tracebacks=False,
            )
            return (await collect(transport.stream_gemini(request)))[0]

    tools = [{"functionDeclarations": [{"name": "read"}]}]
    value = await execute(gemini_request(tools=tools), "first-account")
    parts = value["candidates"][0]["content"]["parts"]
    # Real clients relocate a late signature to an earlier content part.
    signature = parts.pop()["thoughtSignature"]
    parts[0]["thoughtSignature"] = signature
    if tool_call:
        parts[0]["functionCall"]["args"] = {"path": "edited"}
        followup = {
            "role": "model",
            "parts": [
                {
                    "functionResponse": {
                        "id": parts[0]["functionCall"]["id"],
                        "name": "read",
                        "response": {"output": "result"},
                    }
                }
            ],
        }
    else:
        followup = {"role": "user", "parts": [{"text": "continue"}]}
    history = [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": parts},
        followup,
    ]
    await execute(
        gemini_request(tools=tools, contents=history),
        "first-account" if same_connection else "different-account",
    )
    assert any(item["role"] == "assistant" for item in sent[1]["messages"]), sent[1]
    assistant = next(
        item for item in sent[1]["messages"] if item["role"] == "assistant"
    )
    target = assistant["tool_calls"][0] if tool_call else assistant
    if same_connection:
        assert target["extra_content"] == opaque
    else:
        assert "native-signed-bytes" not in json.dumps(sent[1])
    if tool_call:
        assert json.loads(target["function"]["arguments"]) == {"path": "edited"}
    assert "fcc:history" not in json.dumps(sent[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize(
    "arguments", ['{"path":42,"extra":null}', '{"path":', "[]", "null", '{"path":NaN}']
)
async def test_fragmented_tools_require_object_json_without_schema_coercion(
    protocol, arguments
):
    options = (
        {"endpoint_context": Endpoint()}
        if protocol == "messages"
        else {"input_tokens": 0}
    )
    request = gemini_request(
        tools=[
            {
                "functionDeclarations": [
                    {
                        "name": "read",
                        "parametersJsonSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        ]
    )
    chunks = []
    async with _harness(
        protocol, lambda _: (200, tool_events(protocol, arguments)), max_attempts=1
    ) as (_, bodies, transport):
        stream = transport.stream_gemini(
            request,
            request_id="fragmented",
            response_model="public",
            reasoning=ReasoningPolicy.provider_default(),
            **options,
        )
        if arguments.startswith('{"path":42'):
            result, chunks = await collect(stream)
            calls = [
                part["functionCall"]
                for part in result["candidates"][0]["content"]["parts"]
                if "functionCall" in part
            ]
            assert calls == [
                {
                    "id": "call_probe",
                    "name": "read",
                    "args": {"path": 42, "extra": None},
                }
            ]
        else:
            with pytest.raises(ExecutionFailure):
                async for chunk in stream:
                    chunks.append(chunk)
            assert not any("functionCall" in chunk for chunk in chunks)
        assert len(bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
async def test_unknown_upstream_function_is_rejected(protocol):
    options = (
        {"endpoint_context": Endpoint()}
        if protocol == "messages"
        else {"input_tokens": 0}
    )
    async with _harness(
        protocol, lambda _: (200, tool_events(protocol, "{}")), max_attempts=1
    ) as (_, bodies, transport):
        with pytest.raises(ExecutionFailure, match="undeclared"):
            async for chunk in transport.stream_gemini(
                gemini_request(),
                request_id="unknown",
                response_model="public",
                reasoning=ReasoningPolicy.provider_default(),
                **options,
            ):
                assert "functionCall" not in chunk
        assert len(bodies) == 1
