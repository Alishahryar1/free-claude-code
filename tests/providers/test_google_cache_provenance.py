"""Cached Google metadata is scoped to the credential used by each attempt."""

import asyncio
import json
from dataclasses import replace
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.google_openai.provider import GoogleChatBehavior
from free_claude_code.providers.google_openai.thought_signatures import (
    GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
)
from free_claude_code.providers.history_replay import replay_origin
from tests.providers.test_gemini_review_regressions import (
    CALL,
    GOOGLE_PROFILE,
    TOOLS,
    transport,
)
from tests.providers.test_gemini_transport import chat_response, collect, gemini_request


def history_request(wire, *, model="model", replay=False):
    if wire == "gemini":
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": "hello"}]}
        ]
        if replay:
            contents += [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"id": "call_a", "name": "read", "args": {}}}
                    ],
                },
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
            ]
        return gemini_request(model=model, contents=contents, tools=TOOLS)
    if wire == "messages":
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        if replay:
            messages += [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_a",
                            "name": "read",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_a",
                            "content": "ok",
                        }
                    ],
                },
            ]
        return MessagesRequest.model_validate(
            {
                "model": model,
                "max_tokens": 64,
                "messages": messages,
                "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            }
        )
    items = [{"role": "user", "content": "hello"}]
    if replay:
        items += [
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "read",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
        ]
    return OpenAIResponsesRequest.model_validate(
        {
            "model": model,
            "input": items,
            "tools": [
                {"type": "function", "name": "read", "parameters": {"type": "object"}}
            ],
        }
    )


async def consume(adapter, wire, **kwargs):
    return [
        event
        async for event in getattr(adapter, f"stream_{wire}")(
            history_request(wire, **kwargs)
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
@pytest.mark.parametrize(
    "change", ["none", "model", "credential", "endpoint", "client"]
)
async def test_cache_restoration_uses_actual_connection_for_every_ingress(wire, change):
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return chat_response(
            [
                {
                    "tool_calls": [
                        {
                            **CALL,
                            "extra_content": {
                                "google": {"thought_signature": "source-signature"}
                            },
                        }
                    ]
                }
            ],
            finish="tool_calls",
        )

    async with AsyncOpenAI(
        api_key="source-key",
        base_url="https://source.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        behavior = GoogleChatBehavior(GOOGLE_PROFILE)
        adapter = transport(client, behavior)
        await consume(adapter, wire)
        if change == "credential":
            client.api_key = "different-key"
        elif change == "endpoint":
            client.base_url = "https://different.invalid/v1"
        elif change == "client":
            adapter = transport(client.with_options(), behavior)
        await consume(
            adapter,
            wire,
            replay=True,
            model="changed" if change == "model" else "model",
        )
    call = next(m for m in sent[1]["messages"] if m.get("tool_calls"))["tool_calls"][0]
    expected = (
        GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR
        if change in {"credential", "endpoint"}
        else "source-signature"
    )
    assert call["extra_content"]["google"]["thought_signature"] == expected


@pytest.mark.asyncio
async def test_renewable_callback_is_resolved_once_per_attempt_and_pins_provenance():
    resolutions = 0
    sent = []

    async def credential():
        nonlocal resolutions
        resolutions += 1
        return f"synthetic-{resolutions}"

    def reply(request):
        sent.append(request.headers["authorization"])
        return chat_response(
            [
                {
                    "tool_calls": [
                        {
                            **CALL,
                            "extra_content": {
                                "google": {"thought_signature": "signature"}
                            },
                        }
                    ]
                }
            ],
            finish="tool_calls",
        )

    async with AsyncOpenAI(
        api_key=credential,
        base_url="https://source.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        adapter = transport(
            client, GoogleChatBehavior(GOOGLE_PROFILE), api_key_provider=credential
        )
        for index in range(1, 3):
            value, _ = await collect(adapter.stream_gemini(gemini_request(tools=TOOLS)))
            bundle = decode_bundle(
                value["candidates"][0]["content"]["parts"][-1]["thoughtSignature"]
            )
            assert bundle is not None
            expected = replay_origin(
                "GEMINI",
                "chat",
                "model",
                client=client.with_options(api_key=f"synthetic-{index}"),
            )
            assert bundle.calls[0].origin == expected
            assert resolutions == index
            assert not client.is_closed()
    assert sent == ["Bearer synthetic-1", "Bearer synthetic-2"]


@pytest.mark.asyncio
async def test_endpoint_snapshot_precedes_renewable_callback():
    class Endpoint:
        async def endpoint(self, *, force_refresh=False):
            return HttpEndpoint(
                "https://endpoint.invalid/v1", {}, "endpoint-key", "stable-account"
            )

    async def unexpected_callback():
        raise AssertionError("explicit endpoint owns credentials")

    requests = []

    def reply(request):
        requests.append(request)
        return chat_response([{"content": "answer"}])

    wire = httpx2.MockTransport(reply)
    async with AsyncOpenAI(
        api_key="base-key",
        base_url="https://base.invalid/v1",
        http_client=httpx2.AsyncClient(transport=wire),
    ) as client:
        adapter = transport(
            client,
            GoogleChatBehavior(GOOGLE_PROFILE),
            api_key_provider=unexpected_callback,
            endpoint_transport=wire,
        )
        await collect(
            adapter.stream_gemini(gemini_request(), endpoint_context=Endpoint())
        )
        assert not client.is_closed()
    assert requests[0].headers["authorization"] == "Bearer endpoint-key"
    assert requests[0].url.host == "endpoint.invalid"


def test_global_cache_bound_and_native_metadata_precedence():
    behavior = GoogleChatBehavior(GOOGLE_PROFILE)
    source = ReplayOrigin("GEMINI", "chat", "endpoint", "account", "model")
    for index in range(4097):
        behavior.record_tool_call_extra_content(
            str(index),
            {"google": {"thought_signature": str(index)}},
            origin=replace(source, connection=str(index % 2)),
        )
    assert len(behavior._tool_call_extra_content_by_id) == 4096
    assert not any(key[1] == "0" for key in behavior._tool_call_extra_content_by_id)
    origin = replace(source, connection="0")
    explicit = {
        "id": "4096",
        "extra_content": {"google": {"thought_signature": "explicit"}},
    }
    body = {"messages": [{"role": "assistant", "tool_calls": [explicit]}]}
    behavior.prepare_history_body(body, origin)
    assert explicit["extra_content"]["google"]["thought_signature"] == "explicit"


@pytest.mark.asyncio
async def test_concurrent_attempts_record_their_own_origin():
    barrier = asyncio.Event()
    arrivals = 0

    async def reply(request):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            barrier.set()
        await barrier.wait()
        identity = request.headers["authorization"].removeprefix("Bearer ")
        return chat_response(
            [
                {
                    "tool_calls": [
                        {
                            **CALL,
                            "extra_content": {
                                "google": {"thought_signature": identity}
                            },
                        }
                    ]
                }
            ],
            finish="tool_calls",
        )

    behavior = GoogleChatBehavior(GOOGLE_PROFILE)
    wire = httpx2.MockTransport(reply)
    async with AsyncOpenAI(
        api_key="one",
        base_url="https://source.invalid/v1",
        http_client=httpx2.AsyncClient(transport=wire),
    ) as client:
        other = client.with_options(api_key="two")
        await asyncio.gather(
            consume(transport(client, behavior), "gemini"),
            consume(transport(other, behavior), "gemini"),
        )
        for selected, identity in ((client, "one"), (other, "two")):
            origin = replay_origin("GEMINI", "chat", "model", client=selected)
            body: dict[str, Any] = {
                "messages": [{"role": "assistant", "tool_calls": [{"id": "call_a"}]}]
            }
            prepared = behavior.prepare_history_body(body, origin)
            assert (
                prepared["messages"][0]["tool_calls"][0]["extra_content"]["google"][
                    "thought_signature"
                ]
                == identity
            )


@pytest.mark.asyncio
async def test_buffered_recovery_metadata_uses_replacement_credentials():
    resolutions = 0
    requests = []

    async def credential():
        nonlocal resolutions
        resolutions += 1
        return f"key-{resolutions}"

    def reply(request):
        requests.append(request.headers["authorization"])
        if len(requests) == 1:
            return chat_response(
                [
                    {"reasoning_content": "original thought"},
                    {"content": "visible " * 10000},
                ],
                finish=None,
            )
        return chat_response(
            [
                {"reasoning_content": "replacement thought"},
                {
                    "tool_calls": [
                        {
                            **CALL,
                            "extra_content": {
                                "google": {"thought_signature": "recovered-signature"}
                            },
                        }
                    ]
                },
            ],
            finish="tool_calls",
        )

    async with AsyncOpenAI(
        api_key=credential,
        base_url="https://source.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        behavior = GoogleChatBehavior(GOOGLE_PROFILE)
        adapter = transport(client, behavior, api_key_provider=credential)
        value, _ = await collect(adapter.stream_gemini(gemini_request(tools=TOOLS)))
        bundle = decode_bundle(
            value["candidates"][0]["content"]["parts"][-1]["thoughtSignature"]
        )
        assert bundle is not None
        source = replay_origin(
            "GEMINI", "chat", "model", client=client.with_options(api_key="key-1")
        )
        replacement = replay_origin(
            "GEMINI", "chat", "model", client=client.with_options(api_key="key-2")
        )
        assert requests == ["Bearer key-1", "Bearer key-2"]
        assert bundle.records[0].origin == source
        assert bundle.records[0].native == {"reasoning_content": "original thought"}
        assert bundle.records[1].origin == replacement
        assert bundle.records[1].native == {"reasoning_content": "replacement thought"}
        assert bundle.calls[0].origin == replacement
        assert (
            replacement,
            bundle.calls[0].call_id,
        ) in behavior._tool_call_extra_content_by_id
        assert (
            source,
            bundle.calls[0].call_id,
        ) not in behavior._tool_call_extra_content_by_id
