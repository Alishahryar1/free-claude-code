"""Tests for the xKiro OpenAI-chat gateway profile and catalog."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import XKIRO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    OpenAIChatProvider,
)
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
    stream_messages,
    stream_responses,
)

_MODEL = "qwen/qwen3.7-flash:free"


@pytest_asyncio.fixture
async def xkiro_provider():
    provider = profiled_provider(
        "xkiro",
        make_provider_config(api_key="test-xkiro-key", base_url=XKIRO_DEFAULT_BASE),
        admission=immediate_admission(provider_name="xkiro", max_attempts=1),
    )
    try:
        yield provider
    finally:
        await provider.cleanup()


@asynccontextmanager
async def _wire_provider(handler):
    async with AsyncOpenAI(
        api_key="wire-xkiro-key",
        base_url=XKIRO_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        yield OpenAIChatProvider(
            make_provider_config(api_key="wire-xkiro-key", base_url=XKIRO_DEFAULT_BASE),
            profile=OPENAI_CHAT_PROFILES["xkiro"],
            admission=immediate_admission(provider_name="xkiro", max_attempts=1),
            client=client,
        )


def _request(**overrides: JsonValue) -> MessagesRequest:
    payload: JsonObject = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Inspect the file."}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def test_constructs_standard_openai_chat_provider(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(xkiro_provider, OpenAIChatProvider)
    assert xkiro_provider._provider_name == "XKIRO"
    assert xkiro_provider._api_key == "test-xkiro-key"
    assert xkiro_provider._base_url == XKIRO_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_OFF, "none"),
        (REASONING_ON, "adaptive"),
        (ReasoningPolicy.on(budget_tokens=8000), "adaptive"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "minimal"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "xhigh"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "max"),
    ],
)
def test_encodes_only_documented_reasoning_efforts(
    xkiro_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str | None,
) -> None:
    body = xkiro_provider._chat._build_request_body(
        _request(),
        reasoning=reasoning,
    )

    assert body.get("reasoning_effort") == expected
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "thinking" not in body
    assert "extra_body" not in body


def test_replays_reasoning_content_with_tool_history(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "Inspect the file."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Read it first."},
                    {"type": "text", "text": "I will inspect it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "example.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "print('hello')",
                    }
                ],
            },
        ]
    )

    body = xkiro_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "I will inspect it.",
        "reasoning_content": "Read it first.",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "example.py"}',
                },
            }
        ],
    }
    assert body["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }
    assert (
        xkiro_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_extracts_bare_model_ids(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    xkiro_provider._client.get = AsyncMock(
        return_value={"data": [{"id": _MODEL}, {"id": "xkiro/free"}]}
    )

    model_infos = await xkiro_provider.list_model_infos()

    assert model_infos == frozenset(
        {ProviderModelInfo(_MODEL), ProviderModelInfo("xkiro/free")}
    )
    xkiro_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{"id": _MODEL}, {}]}, "include id"),
        ({"data": []}, "did not include any model ids"),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    xkiro_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    xkiro_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await xkiro_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL}]})

    async with _wire_provider(handler) as provider:
        model_infos = await provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.xkiro.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-xkiro-key"


@pytest.mark.asyncio
async def test_full_catalog_retains_tiers_and_maps_capabilities(xkiro_provider):
    xkiro_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": _MODEL,
                    "access_tier": "free",
                    "capabilities": {"reasoning": True, "vision": True, "tools": True},
                    "context_length": 128000,
                    "max_output_tokens": 8192,
                },
                {
                    "id": "paid/model:free",
                    "access_tier": "paid",
                    "capabilities": {
                        "reasoning": False,
                        "vision": False,
                        "tools": False,
                    },
                },
                {"id": "premium/model", "access_tier": "premium"},
                {"id": _MODEL},
            ]
        }
    )
    infos = {info.model_id: info for info in await xkiro_provider.list_model_infos()}
    assert set(infos) == {_MODEL, "paid/model:free", "premium/model"}
    assert infos[_MODEL].supports_thinking is True
    assert infos[_MODEL].input_modalities == frozenset(
        {ModelInputModality.TEXT, ModelInputModality.IMAGE}
    )
    assert infos[_MODEL].context_window_tokens == 128000
    assert infos[_MODEL].max_output_tokens == 8192
    assert infos["paid/model:free"].supports_thinking is False
    assert infos["paid/model:free"].input_modalities == frozenset(
        {ModelInputModality.TEXT}
    )
    assert infos["premium/model"].supports_thinking is None
    assert infos["premium/model"].input_modalities is None
    assert infos["premium/model"].context_window_tokens is None


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_image_and_named_effort_translation(xkiro_provider, wire):
    image = "https://example.test/image.png"
    if wire == "messages":
        request = _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": image}},
                        {"type": "text", "text": "Describe"},
                    ],
                }
            ]
        )
        body = xkiro_provider._chat._build_request_body(
            request, reasoning=ReasoningPolicy.on(effort=ReasoningEffort.XHIGH)
        )
    else:
        request = OpenAIResponsesRequest(
            model=_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image},
                        {"type": "input_text", "text": "Describe"},
                    ],
                }
            ],
        )
        body = xkiro_provider._chat._build_responses_request_body(
            request, reasoning=ReasoningPolicy.on(effort=ReasoningEffort.XHIGH)
        ).body
    assert body["reasoning_effort"] == "xhigh"
    assert body["model"] == _MODEL
    assert body["messages"][0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": image},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_chat_stream_preserves_reasoning_tools_and_usage(wire):
    requests = []
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Inspecting."}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "inspect",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"README.md"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
        },
    ]

    def handler(request):
        requests.append(request)
        frames = [
            {
                "id": "chat_1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": _MODEL,
                **chunk,
            }
            for chunk in chunks
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
            + "data: [DONE]\n\n",
        )

    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    async with _wire_provider(handler) as provider:
        stream = (
            stream_messages(
                provider,
                MessagesRequest(
                    model=_MODEL,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": "Inspect the README."}],
                    tools=[{"name": "inspect", "input_schema": schema}],
                ),
            )
            if wire == "messages"
            else stream_responses(
                provider,
                OpenAIResponsesRequest(
                    model=_MODEL,
                    input="Inspect the README.",
                    max_output_tokens=2048,
                    tools=[
                        {"type": "function", "name": "inspect", "parameters": schema}
                    ],
                ),
            )
        )
        output = "".join([event async for event in stream])

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer wire-xkiro-key"
    body = json.loads(requests[0].content)
    assert body["model"] == _MODEL
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 2048
    assert body["tools"][0]["function"]["parameters"] == schema
    assert "reasoning_effort" not in body
    assert "Inspecting." in output
    assert "inspect" in output
    assert "README.md" in output
    events = [
        json.loads(line[6:])
        for line in output.splitlines()
        if line.startswith("data: {")
    ]
    if wire == "messages":
        terminal = next(event for event in events if event["type"] == "message_delta")
        assert terminal["delta"]["stop_reason"] == "tool_use"
        assert terminal["usage"]["output_tokens"] == 10
    else:
        terminal = next(
            event for event in events if event["type"] == "response.completed"
        )
        assert terminal["response"]["usage"]["input_tokens"] == 25
        assert terminal["response"]["usage"]["output_tokens"] == 10
