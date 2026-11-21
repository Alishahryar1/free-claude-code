"""Behavioral reproductions for the Gemini ingress review findings."""

import json
from dataclasses import replace
from typing import Literal

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini.replay import (
    ReplayBundle,
    encode_bundle,
)
from free_claude_code.core.history_replay import ReplayOrigin, ReplayRecord
from free_claude_code.providers.deepseek.client import _PROFILE, DeepSeekChatBehavior
from free_claude_code.providers.google_openai.provider import GoogleChatBehavior
from free_claude_code.providers.openai_chat import OpenAIChatTransport
from free_claude_code.providers.openai_chat.client import OpenAIAsyncCredentialProvider
from free_claude_code.providers.vertex.client import _PROFILE as GOOGLE_PROFILE
from tests.providers.support import immediate_admission
from tests.providers.test_gemini_transport import chat_response, collect, gemini_request
from tests.providers.test_lmstudio import (
    completion_response,
    metadata_response,
    provider_with_http,
)
from tests.wire_payload import array_payload, decode_object, object_payload


def transport(
    client: AsyncOpenAI,
    behavior: DeepSeekChatBehavior | GoogleChatBehavior,
    *,
    api_key_provider: OpenAIAsyncCredentialProvider | None = None,
    endpoint_transport: httpx2.AsyncBaseTransport | None = None,
):
    return OpenAIChatTransport(
        client=client,
        behavior=behavior,
        admission=immediate_admission(),
        read_timeout_s=2,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
        api_key_provider=api_key_provider,
        endpoint_transport=endpoint_transport,
    )


TOOLS = [{"functionDeclarations": [{"name": "read"}]}]
CALL = {
    "index": 0,
    "id": "call_a",
    "type": "function",
    "function": {"name": "read", "arguments": "{}"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize("with_call", [False, True])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_plain_reasoning_survives_gemini_round_trip(
    visible: bool, with_call: bool, field: Literal["reasoning_content", "reasoning"]
):
    sent = []

    def reply(request: httpx2.Request):
        sent.append(decode_object(request.content))
        return chat_response(
            [
                {field: "original "},
                {field: "thought"},
                {"tool_calls": [CALL]} if with_call else {"content": "answer"},
            ],
            finish="tool_calls" if with_call else "stop",
        )

    async with AsyncOpenAI(
        api_key="synthetic",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        adapter = transport(
            client, DeepSeekChatBehavior(replace(_PROFILE, reasoning_delta_field=field))
        )
        request = gemini_request(
            tools=TOOLS,
            generationConfig={"thinkingConfig": {"includeThoughts": visible}},
        )
        value, _ = await collect(adapter.stream_gemini(request))
        content = value["candidates"][0]["content"]
        signatures = [
            p["thoughtSignature"] for p in content["parts"] if "thoughtSignature" in p
        ]
        assert not signatures
        result = (
            {
                "functionResponse": {
                    "id": "call_a",
                    "name": "read",
                    "response": {"ok": True},
                }
            }
            if with_call
            else {"text": "continue"}
        )
        await collect(
            adapter.stream_gemini(
                gemini_request(
                    tools=TOOLS,
                    contents=[
                        {"role": "user", "parts": [{"text": "hello"}]},
                        content,
                        {"role": "user", "parts": [result]},
                    ],
                )
            )
        )
    assistant = next(
        m
        for m in array_payload(sent[1]["messages"])
        if object_payload(m)["role"] == "assistant"
    )
    assert object_payload(assistant)["reasoning_content"] == (
        "original thought" if visible else ""
    )
    assert "original thought" not in json.dumps(
        object_payload(assistant).get("content")
    )


@pytest.mark.asyncio
async def test_google_signature_cache_does_not_cross_credentials():
    sent = []
    call = {
        **CALL,
        "extra_content": {"google": {"thought_signature": "account-a-signature"}},
    }

    def reply(request: httpx2.Request):
        sent.append(decode_object(request.content))
        return chat_response([{"tool_calls": [call]}], finish="tool_calls")

    async with AsyncOpenAI(
        api_key="account-a",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        adapter = transport(client, GoogleChatBehavior(GOOGLE_PROFILE))
        value, _ = await collect(adapter.stream_gemini(gemini_request(tools=TOOLS)))
        client.api_key = "account-b"
        await collect(
            adapter.stream_gemini(
                gemini_request(
                    tools=TOOLS,
                    contents=[
                        {"role": "user", "parts": [{"text": "hello"}]},
                        value["candidates"][0]["content"],
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "id": "call_a",
                                        "name": "read",
                                        "response": {"ok": True},
                                    }
                                }
                            ],
                        },
                    ],
                )
            )
        )
    assert "account-a-signature" not in json.dumps(sent[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["replay", "image"])
async def test_lmstudio_counts_projected_replay_and_images(kind: str):
    sent = []

    def reply(request: httpx2.Request):
        if request.method == "GET":
            return metadata_response(100 if kind == "image" else 1024)
        sent.append(decode_object(request.content))
        return completion_response()

    record = ReplayRecord(
        ReplayOrigin("foreign", "messages", "", "", "model"),
        {
            "type": "thinking",
            "thinking": "replay reasoning " * 6000,
            "signature": "opaque",
        },
    )
    parts = (
        [{"thoughtSignature": encode_bundle(ReplayBundle("response", (record,)))}]
        if kind == "replay"
        else [{"inlineData": {"mimeType": "image/png", "data": "AAAA" * 90000}}]
    )
    request = gemini_request(
        contents=[
            {"role": "model" if kind == "replay" else "user", "parts": parts},
            {"role": "user", "parts": [{"text": "continue"}]},
        ]
    )
    async with provider_with_http(reply) as provider:
        with pytest.raises(ExecutionFailure) as caught:
            await collect(provider.stream_gemini(request))
        assert caught.value.kind == FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert not sent
