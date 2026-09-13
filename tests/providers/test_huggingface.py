"""Tests for Hugging Face Inference Providers."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import HUGGINGFACE_DEFAULT_BASE
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


def make_request(**overrides):
    return make_messages_request("openai/gpt-oss-120b:fastest", **overrides)


@pytest.fixture
def huggingface_config():
    return make_provider_config(
        api_key="test_hf_key",
        base_url=HUGGINGFACE_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def huggingface_provider(huggingface_config):
    return profiled_provider(
        "huggingface", huggingface_config, admission=immediate_admission()
    )


def test_default_base_url_constant():
    assert HUGGINGFACE_DEFAULT_BASE == "https://router.huggingface.co/v1"


def test_init_uses_default_base_url_and_api_key(huggingface_config):
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "huggingface", huggingface_config, admission=immediate_admission()
        )

    assert provider._api_key == "test_hf_key"
    assert provider._base_url == HUGGINGFACE_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(huggingface_config):
    config = replace(huggingface_config, base_url=f"{HUGGINGFACE_DEFAULT_BASE}/")

    with patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"):
        provider = profiled_provider(
            "huggingface", config, admission=immediate_admission()
        )

    assert provider._base_url == HUGGINGFACE_DEFAULT_BASE


@pytest.mark.asyncio
async def test_model_catalog_extracts_exact_input_modalities(
    huggingface_provider,
) -> None:
    huggingface_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                {
                    "id": "vision-model",
                    "architecture": {"input_modalities": ["text", "image", "audio"]},
                    "providers": [
                        {"status": "live", "context_length": 131072},
                        {"status": "live", "context_length": 131072},
                        {"status": "staging", "context_length": 999999},
                    ],
                },
                {
                    "id": "unknown-model",
                    "architecture": {"input_modalities": ["image"]},
                },
            ]
        )
    )

    assert await huggingface_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo(
                "vision-model",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131072,
            ),
            ProviderModelInfo("unknown-model"),
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "providers",
    [
        [],
        [{"status": "staging", "context_length": 131072}],
        [{"status": "live"}],
        [{"status": "live", "context_length": "131072"}],
        [
            {"status": "live", "context_length": 131072},
            {"status": "live", "context_length": 65536},
        ],
    ],
)
async def test_model_catalog_requires_consensus_across_live_huggingface_routes(
    huggingface_provider,
    providers: list[dict[str, object]],
) -> None:
    huggingface_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[{"id": "model", "providers": providers}])
    )

    assert await huggingface_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("model")}
    )


def test_build_request_body_keeps_max_tokens(huggingface_provider):
    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "openai/gpt-oss-120b:fastest",
            "messages": [{"role": "user", "name": "alice", "content": "hi"}],
            "max_tokens": 42,
        }

        body = huggingface_provider._chat._build_request_body(make_request())

    mock_convert.assert_called_once()
    assert (
        mock_convert.call_args.kwargs["reasoning_replay"]
        is ReasoningReplayMode.DISABLED
    )
    assert body["messages"][0].get("name") == "alice"
    assert body["max_tokens"] == 42
    assert "max_completion_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(huggingface_provider):
    extra_body = {"provider": "auto", "routing": {"bill_to": "my-org"}}
    req = make_request(extra_body=extra_body)

    body = huggingface_provider._chat._build_request_body(req)

    assert body["extra_body"] == extra_body
    assert body["extra_body"] is not extra_body
    assert body["extra_body"]["routing"] is not extra_body["routing"]


@pytest.mark.parametrize(
    "reasoning",
    (
        ReasoningPolicy.off(),
        ReasoningPolicy.on(effort=ReasoningEffort.MAX),
        ReasoningPolicy.on(budget_tokens=4096),
    ),
)
def test_build_request_body_leaves_reasoning_control_to_selected_upstream(
    huggingface_provider, reasoning
):
    body = huggingface_provider._chat._build_request_body(
        make_request(),
        reasoning=reasoning,
    )

    assert "reasoning_effort" not in body
    assert "reasoning" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_build_request_body_does_not_replay_prior_thinking_blocks(
    huggingface_provider,
):
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hidden prior thought"},
                    {"type": "text", "text": "visible answer"},
                ],
            }
        ],
    )

    body = huggingface_provider._chat._build_request_body(req)

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": "[Earlier reasoning]\nhidden prior thought\n\nvisible answer",
        }
    ]
    assert "reasoning_content" not in body["messages"][0]
    assert "hidden prior thought" in str(body)


def test_build_request_body_does_not_replay_top_level_reasoning_content(
    huggingface_provider,
):
    req = make_request(
        system=None,
        messages=[
            {
                "role": "assistant",
                "content": "visible answer",
                "reasoning_content": "hidden prior reasoning",
            }
        ],
    )

    body = huggingface_provider._chat._build_request_body(req)

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": "[Earlier reasoning]\nhidden prior reasoning\n\nvisible answer",
        }
    ]
    assert "hidden prior reasoning" in str(body)


@pytest.mark.asyncio
async def test_stream_messages_text(huggingface_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Hugging Face",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        huggingface_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in huggingface_provider.stream_messages(make_request())
        ]

    assert any(
        '"text_delta"' in event and "Hello from Hugging Face" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_stream_messages_reasoning_content(huggingface_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via router",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        huggingface_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in huggingface_provider.stream_messages(make_request())
        ]

    assert any(
        '"thinking_delta"' in event and "Thinking via router" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(huggingface_provider):
    huggingface_provider._client = AsyncMock()

    await huggingface_provider.cleanup()

    huggingface_provider._client.close.assert_called_once()
