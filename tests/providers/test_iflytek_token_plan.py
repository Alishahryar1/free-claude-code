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
