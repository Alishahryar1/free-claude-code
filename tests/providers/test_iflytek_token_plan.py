"""Token Plan discovery, configuration, and Chat Completions wire tests."""

import json
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.provider_catalog import (
    IFLYTEK_TOKEN_PLAN_DEFAULT_BASE,
    PROVIDER_CATALOG,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningCapability, ReasoningPolicy
from free_claude_code.providers.iflytek_token_plan import IFlytekTokenPlanProvider
from free_claude_code.providers.runtime.config import build_provider_config
from tests.providers.support import immediate_admission, make_provider_config


def test_subscription_credentials_use_token_plan_endpoint() -> None:
    settings = compose_settings_snapshot(
        {"IFLYTEK_TOKEN_PLAN_API_KEY": "subscription-key"},
        {"IFLYTEK_TOKEN_PLAN_PROXY": "http://proxy.test:8080"},
    ).settings
    config = build_provider_config(PROVIDER_CATALOG["iflytek_token_plan"], settings)

    assert config.api_key == "subscription-key"
    assert config.base_url == "https://maas-token-api.cn-huabei-1.xf-yun.com/v2"
    assert config.proxy == "http://proxy.test:8080"


@pytest.mark.asyncio
async def test_static_models_do_not_call_models_endpoint() -> None:
    client = AsyncMock(spec=AsyncOpenAI)
    provider = IFlytekTokenPlanProvider(
        make_provider_config("test-key", IFLYTEK_TOKEN_PLAN_DEFAULT_BASE),
        admission=immediate_admission(),
        client=client,
    )

    model_infos = await provider.list_model_infos()

    assert {info.model_id for info in model_infos} == {"spark-x2.5", "xsparkx2flash"}
    assert all(info.supports_thinking for info in model_infos)
    assert all(
        info.reasoning_capability is ReasoningCapability.OPTIONAL
        for info in model_infos
    )
    assert client.mock_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["spark-x2.5", "xsparkx2flash"])
@pytest.mark.parametrize(
    "reasoning, expected_thinking",
    [
        (ReasoningPolicy.provider_default(), None),
        (ReasoningPolicy.off(), {"type": "disabled"}),
        (ReasoningPolicy.on(), {"type": "enabled"}),
    ],
)
async def test_wire_preserves_tools_and_encodes_thinking(
    model: str,
    reasoning: ReasoningPolicy,
    expected_thinking: dict[str, str] | None,
) -> None:
    captured: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: [DONE]\n\n",
        )

    client = AsyncOpenAI(
        api_key="test-token-plan-key",
        base_url=IFLYTEK_TOKEN_PLAN_DEFAULT_BASE,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        max_retries=0,
    )
    provider = IFlytekTokenPlanProvider(
        make_provider_config("test-token-plan-key", IFLYTEK_TOKEN_PLAN_DEFAULT_BASE),
        admission=immediate_admission(),
        client=client,
    )
    request = MessagesRequest.model_validate(
        {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Look up the weather."}],
            "tools": [
                {
                    "name": "weather",
                    "description": "Fetch the weather for a city.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        }
    )
    body = provider._chat._build_request_body(request, reasoning=reasoning)
    try:
        stream = await client.chat.completions.create(**body, stream=True)
        await stream.close()
    finally:
        await client.close()

    assert len(captured) == 1
    wire = captured[0]
    payload = json.loads(wire.content)
    assert str(wire.url) == f"{IFLYTEK_TOKEN_PLAN_DEFAULT_BASE}/chat/completions"
    assert wire.headers["authorization"] == "Bearer test-token-plan-key"
    assert payload["model"] == model
    assert payload["max_tokens"] == 4096
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "weather"
    assert payload["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert payload.get("thinking") == expected_thinking


def _sse(chunks: list[dict]) -> httpx2.Response:
    frames = [
        {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "spark-x2.5",
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


@pytest.mark.asyncio
async def test_streamed_tool_turn_replays_reasoning_after_tool_result() -> None:
    captured: list[httpx2.Request] = []
    tool_turn = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Need the city."}}]},
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
                                    "name": "weather",
                                    "arguments": '{"city":',
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
                            {"index": 0, "function": {"arguments": '"Hefei"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        },
    ]
    answer_turn = [
        {"choices": [{"index": 0, "delta": {"content": "Sunny in Hefei."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]

    def respond(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return _sse(tool_turn if len(captured) == 1 else answer_turn)

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    tools = [{"name": "weather", "input_schema": schema}]
    question = {"role": "user", "content": "What is the weather in Hefei?"}
    async with AsyncOpenAI(
        api_key="test-token-plan-key",
        base_url=IFLYTEK_TOKEN_PLAN_DEFAULT_BASE,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        max_retries=0,
    ) as client:
        provider = IFlytekTokenPlanProvider(
            make_provider_config(
                "test-token-plan-key", IFLYTEK_TOKEN_PLAN_DEFAULT_BASE
            ),
            admission=immediate_admission(),
            client=client,
        )
        model_info = next(
            info
            for info in await provider.list_model_infos()
            if info.model_id == "spark-x2.5"
        )
        first = "".join(
            [
                event
                async for event in provider.stream_messages(
                    MessagesRequest.model_validate(
                        {
                            "model": "spark-x2.5",
                            "max_tokens": 2048,
                            "messages": [question],
                            "tools": tools,
                        }
                    ),
                    reasoning=ReasoningPolicy.on(),
                    model_info=model_info,
                )
            ]
        )
        second = "".join(
            [
                event
                async for event in provider.stream_messages(
                    MessagesRequest.model_validate(
                        {
                            "model": "spark-x2.5",
                            "max_tokens": 2048,
                            "messages": [
                                question,
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "thinking",
                                            "thinking": "Need the city.",
                                        },
                                        {
                                            "type": "tool_use",
                                            "id": "call_1",
                                            "name": "weather",
                                            "input": {"city": "Hefei"},
                                        },
                                    ],
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": "call_1",
                                            "content": "Sunny, 24C",
                                        }
                                    ],
                                },
                            ],
                            "tools": tools,
                        }
                    ),
                    reasoning=ReasoningPolicy.on(),
                    model_info=model_info,
                )
            ]
        )
        await provider.cleanup()

    events = [
        json.loads(line[6:])
        for line in first.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [
        event["delta"] for event in events if event["type"] == "content_block_delta"
    ]
    assert {"type": "thinking_delta", "thinking": "Need the city."} in deltas
    tool_start = next(
        event["content_block"]
        for event in events
        if event["type"] == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    )
    assert tool_start["id"] == "call_1"
    assert tool_start["name"] == "weather"
    assert (
        "".join(
            delta["partial_json"]
            for delta in deltas
            if delta["type"] == "input_json_delta"
        )
        == '{"city":"Hefei"}'
    )
    terminal = next(event for event in events if event["type"] == "message_delta")
    assert terminal["delta"]["stop_reason"] == "tool_use"
    assert terminal["usage"]["output_tokens"] == 12
    assert "Sunny in Hefei." in second

    assert len(captured) == 2
    first_payload = json.loads(captured[0].content)
    assert first_payload["thinking"] == {"type": "enabled"}
    assert first_payload["stream"] is True
    replay = json.loads(captured[1].content)
    assert replay["thinking"] == {"type": "enabled"}
    assistant, tool_message = replay["messages"][1], replay["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "Need the city."
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Hefei"
    }
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Sunny, 24C",
    }
