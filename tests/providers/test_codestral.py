"""Tests for Mistral Codestral provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import CODESTRAL_DEFAULT_BASE
from free_claude_code.core.model_capabilities import ModelInputModality
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


def make_request(**overrides):
    return make_messages_request("devstral-small-latest", **overrides)


@pytest.fixture
def codestral_config():
    return make_provider_config(
        api_key="test_codestral_key",
        base_url=CODESTRAL_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def codestral_provider(codestral_config):
    return profiled_provider(
        "mistral_codestral", codestral_config, admission=immediate_admission()
    )


def test_init(codestral_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider(
            "mistral_codestral",
            codestral_config,
            admission=immediate_admission(),
        )
        assert provider._api_key == "test_codestral_key"
        assert provider._base_url == CODESTRAL_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url():
    assert CODESTRAL_DEFAULT_BASE == "https://codestral.mistral.ai/v1"


def test_build_request_body_basic(codestral_provider):
    """Basic request body conversion works for Codestral."""
    req = make_request()
    body = codestral_provider._chat._build_request_body(req)

    assert body["model"] == "devstral-small-latest"
    assert body["messages"][0]["role"] == "system"


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    """Global disable disables reasoning replay in the converter."""
    provider = profiled_provider(
        "mistral_codestral",
        make_provider_config(
            api_key="test_codestral_key",
            base_url=CODESTRAL_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )
    req = make_request()
    body = provider._chat._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


@pytest.mark.asyncio
async def test_model_catalog_extracts_exact_input_modalities(codestral_provider):
    codestral_provider._client.models.list = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "vision-model",
                    "capabilities": {
                        "completion_chat": True,
                        "vision": True,
                    },
                },
                {
                    "id": "text-model",
                    "capabilities": {
                        "completion_chat": True,
                        "vision": False,
                    },
                },
            ]
        }
    )

    assert await codestral_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo(
                "vision-model",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "text-model",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )


@pytest.mark.parametrize(
    "capabilities",
    [
        None,
        {},
        {"completion_chat": True},
        {"completion_chat": True, "vision": "yes"},
    ],
)
@pytest.mark.asyncio
async def test_model_catalog_degrades_incomplete_capabilities_to_unknown(
    codestral_provider,
    capabilities,
):
    model = {"id": "model"}
    if capabilities is not None:
        model["capabilities"] = capabilities
    codestral_provider._client.models.list = AsyncMock(return_value={"data": [model]})

    assert await codestral_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("model")}
    )


@pytest.mark.asyncio
async def test_stream_messages_text(codestral_provider):
    """Text content deltas are emitted as text blocks."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
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
        codestral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in codestral_provider.stream_messages(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_messages_reasoning_content(codestral_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        codestral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in codestral_provider.stream_messages(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(codestral_provider):
    """cleanup closes the OpenAI client."""
    codestral_provider._client = AsyncMock()

    await codestral_provider.cleanup()

    codestral_provider._client.close.assert_called_once()
