"""Regional settings and real SDK serialization for MiniMax text requests."""

import json

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.loader import ConfigSource, compose_settings_snapshot
from free_claude_code.config.provider_catalog import (
    MINIMAX_DEFAULT_BASE,
    PROVIDER_CATALOG,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    OpenAIChatProvider,
)
from free_claude_code.providers.runtime import build_provider_config
from tests.providers.support import immediate_admission


@pytest.mark.parametrize("source", [ConfigSource.MANAGED, ConfigSource.PROCESS])
@pytest.mark.parametrize(
    "base_url",
    [MINIMAX_DEFAULT_BASE, "https://api.minimaxi.com/v1", "https://proxy.example/v1"],
)
def test_minimax_regional_settings_preserve_explicit_model_and_url(source, base_url):
    values = {
        "MINIMAX_BASE_URL": base_url,
        "MINIMAX_API_KEY": "test-key",
        "MODEL": "minimax/MiniMax-M2.7",
    }
    snapshot = compose_settings_snapshot(
        values if source is ConfigSource.MANAGED else {},
        values if source is ConfigSource.PROCESS else {},
    )
    config = build_provider_config(PROVIDER_CATALOG["minimax"], snapshot.settings)
    assert config.base_url == base_url
    assert snapshot.settings.model == "minimax/MiniMax-M2.7"
    assert snapshot.sources["minimax_base_url"] is source


def test_minimax_default_endpoint():
    config = build_provider_config(
        PROVIDER_CATALOG["minimax"], Settings(minimax_api_keys="test-key")
    )
    assert config.base_url == MINIMAX_DEFAULT_BASE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [MINIMAX_DEFAULT_BASE, "https://api.minimaxi.com/v1", "https://proxy.example/v1"],
)
@pytest.mark.parametrize(
    "model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"]
)
@pytest.mark.parametrize("control", [ReasoningControl.DEFAULT, ReasoningControl.OFF])
async def test_minimax_sdk_wire_payload(base_url, model, control):
    captured: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
            ],
        }
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    config = build_provider_config(
        PROVIDER_CATALOG["minimax"],
        Settings(minimax_base_url=base_url, minimax_api_keys="test-key"),
    )
    content: list[dict[str, object]] = [{"type": "text", "text": "Hi"}]
    if model in {"MiniMax-M3.1-Flash", "MiniMax-M3"}:
        content.append(
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
            }
        )
    request = MessagesRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 32,
            "stream": True,
        }
    )
    async with AsyncOpenAI(
        api_key="test-key",
        base_url=config.base_url,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    ) as client:
        provider = OpenAIChatProvider(
            config,
            profile=OPENAI_CHAT_PROFILES["minimax"],
            admission=immediate_admission(provider_name="MINIMAX"),
            client=client,
        )
        events = [
            event
            async for event in provider.stream_messages(
                request, reasoning=ReasoningPolicy(control=control)
            )
        ]

    assert events
    assert len(captured) == 1
    assert str(captured[0].url) == f"{base_url}/chat/completions"
    payload = json.loads(captured[0].content)
    assert payload["model"] == model
    assert payload["reasoning_split"] is True
    assert payload["max_completion_tokens"] == 32
    assert payload["stream"] is True
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert "extra_body" not in payload
    if model in {"MiniMax-M3.1-Flash", "MiniMax-M3"}:
        assert payload["messages"][0]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
