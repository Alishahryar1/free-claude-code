"""Tests for the Amazon Bedrock Mantle provider profile."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import BEDROCK_DEFAULT_BASE
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)

BEDROCK_MODEL = "openai.gpt-oss-120b"


def _provider(base_url: str = BEDROCK_DEFAULT_BASE) -> OpenAIChatProvider:
    return profiled_provider(
        "bedrock",
        make_provider_config(api_key="bedrock-key", base_url=base_url),
        admission=immediate_admission(),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            "https://bedrock-mantle.us-west-2.api.aws",
            "https://bedrock-mantle.us-west-2.api.aws/v1",
        ),
        (
            "https://bedrock-mantle.us-west-2.api.aws/",
            "https://bedrock-mantle.us-west-2.api.aws/v1",
        ),
        (
            "https://bedrock-mantle.us-west-2.api.aws/v1",
            "https://bedrock-mantle.us-west-2.api.aws/v1",
        ),
    ],
)
def test_init_uses_bearer_key_and_normalizes_regional_openai_base(
    configured: str, expected: str
) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = _provider(configured)

    assert provider._provider_name == "BEDROCK"
    assert provider._api_key == "bedrock-key"
    assert provider._base_url == expected
    assert openai_client.call_args.kwargs["api_key"] == "bedrock-key"
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_request_uses_portable_chat_fields_without_invented_reasoning() -> None:
    request = make_messages_request(
        BEDROCK_MODEL,
        output_config={"effort": "high"},
        tools=[
            {
                "name": "read_file",
                "description": "Read one file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )

    body = _provider()._chat._build_request_body(request)

    assert body["model"] == BEDROCK_MODEL
    assert body["messages"][0] == {"role": "system", "content": "System prompt"}
    assert body["max_tokens"] == 100
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning_effort" not in body
    assert "reasoning" not in body
    assert "thinking" not in body


def test_reasoning_history_replays_as_portable_think_tags() -> None:
    request = make_messages_request(
        BEDROCK_MODEL,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the repository."},
                    {"type": "text", "text": "I found the file."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "contents",
                    }
                ],
            },
        ],
    )

    body = _provider()._chat._build_request_body(request)
    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )

    assert assistant["content"] == (
        "<think>\nInspect the repository.\n</think>\n\nI found the file."
    )
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert "reasoning_content" not in assistant
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_model_discovery_uses_mantle_openai_models_endpoint() -> None:
    provider = _provider()
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="openai.gpt-oss-120b"),
                SimpleNamespace(id="amazon.nova-pro-v1:0"),
            ]
        )
    )

    assert await provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("openai.gpt-oss-120b"),
            ProviderModelInfo("amazon.nova-pro-v1:0"),
        }
    )
    provider._client.models.list.assert_awaited_once_with()
