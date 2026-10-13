"""Tests for the Azure OpenAI v1 provider profile."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)

AZURE_OPENAI_BASE_URL = "https://example-resource.openai.azure.com/openai/v1/"
AZURE_OPENAI_DEPLOYMENT = "gpt-5.1"


def _provider() -> OpenAIChatProvider:
    return profiled_provider(
        "azure_openai",
        make_provider_config(
            api_key="azure-key",
            base_url=AZURE_OPENAI_BASE_URL,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_resource_v1_url_and_api_key() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = _provider()

    assert provider._provider_name == "AZURE_OPENAI"
    assert provider._api_key == "azure-key"
    assert provider._base_url == AZURE_OPENAI_BASE_URL.rstrip("/")
    assert openai_client.call_args.kwargs["api_key"] == "azure-key"
    assert openai_client.call_args.kwargs["base_url"] == AZURE_OPENAI_BASE_URL.rstrip(
        "/"
    )


def test_request_uses_deployment_name_modern_token_field_tools_and_reasoning() -> None:
    request = make_messages_request(
        AZURE_OPENAI_DEPLOYMENT,
        temperature=None,
        top_p=None,
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

    body = _provider()._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )

    assert body["model"] == AZURE_OPENAI_DEPLOYMENT
    assert body["messages"][0] == {"role": "system", "content": "System prompt"}
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "high"
    assert body["tools"][0]["function"]["name"] == "read_file"


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy.off(), "none"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ],
)
def test_reasoning_uses_azure_supported_provider_vocabulary(
    policy: ReasoningPolicy,
    expected: str,
) -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            AZURE_OPENAI_DEPLOYMENT,
            temperature=None,
            top_p=None,
        ),
        reasoning=policy,
    )

    assert body["reasoning_effort"] == expected


def test_reasoning_history_replays_as_portable_think_tags() -> None:
    request = make_messages_request(
        AZURE_OPENAI_DEPLOYMENT,
        temperature=None,
        top_p=None,
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the repository."},
                    {"type": "text", "text": "I found the file."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ],
    )

    body = _provider()._chat._build_request_body(request)
    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )

    assert assistant["content"] == (
        "<think>\nInspect the repository.\n</think>\n\nI found the file."
    )
    assert "reasoning_content" not in assistant
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_model_discovery_does_not_advertise_model_ids_as_deployments() -> None:
    provider = _provider()
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-5.1"),
                SimpleNamespace(id="gpt-4.1"),
            ]
        )
    )

    assert await provider.list_model_infos() == frozenset()
    provider._client.models.list.assert_awaited_once_with()
