"""Tests for the official OpenAI API provider profile."""

from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import OPENAI_API_DEFAULT_BASE
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import (
    OpenAIChatProvider,
    is_openai_chat_model,
    is_openai_reasoning_model,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)

_MODEL = "gpt-4o"


def _provider(
    *,
    api_key: str = "test-openai-key",
    base_url: str = OPENAI_API_DEFAULT_BASE,
) -> OpenAIChatProvider:
    return profiled_provider(
        "openai_api",
        make_provider_config(
            api_key=api_key,
            base_url=base_url,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="openai_api"),
    )


def test_constructs_openai_api_provider() -> None:
    provider = _provider()
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._provider_name == "OPENAI_API"
    assert provider._api_key == "test-openai-key"
    assert provider._base_url == OPENAI_API_DEFAULT_BASE


def test_init_uses_api_key_and_base_url() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as openai_client:
        provider = _provider(api_key="custom-key", base_url="https://api.openai.com/v1")

    assert provider._provider_name == "OPENAI_API"
    assert provider._api_key == "custom-key"
    assert provider._base_url == "https://api.openai.com/v1"
    assert openai_client.call_args.kwargs["api_key"] == "custom-key"
    assert openai_client.call_args.kwargs["base_url"] == "https://api.openai.com/v1"


def test_normalizes_base_url_without_v1() -> None:
    provider = _provider(base_url="https://api.openai.com")
    assert provider._base_url == "https://api.openai.com/v1"


def test_gpt4o_request_uses_modern_token_field_tools_and_omits_reasoning() -> None:
    request = make_messages_request(
        "gpt-4o",
        temperature=0.7,
        top_p=0.9,
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

    assert body["model"] == "gpt-4o"
    assert body["messages"][0] == {"role": "system", "content": "System prompt"}
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert "reasoning_effort" not in body
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert body["tools"][0]["function"]["name"] == "read_file"


@pytest.mark.parametrize(
    "policy",
    [
        ReasoningPolicy.off(),
        ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL),
        ReasoningPolicy.on(effort=ReasoningEffort.LOW),
        ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM),
        ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
        ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
        ReasoningPolicy.on(effort=ReasoningEffort.MAX),
    ],
)
def test_gpt4o_omits_reasoning_controls_across_all_policies(
    policy: ReasoningPolicy,
) -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            "gpt-4o",
            temperature=None,
            top_p=None,
        ),
        reasoning=policy,
    )

    assert "reasoning_effort" not in body


def test_gpt4o_preserves_sampling_parameters_and_strips_extra_reasoning() -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            "gpt-4o",
            temperature=0.8,
            top_p=0.95,
        ),
        reasoning=ReasoningPolicy.off(),
    )

    assert body["temperature"] == 0.8
    assert body["top_p"] == 0.95
    assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ],
)
def test_reasoning_model_uses_openai_supported_vocabulary(
    policy: ReasoningPolicy,
    expected: str,
) -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            "o3-mini",
            temperature=None,
            top_p=None,
        ),
        reasoning=policy,
    )

    assert body["reasoning_effort"] == expected


def test_reasoning_model_omits_reasoning_effort_when_disabled() -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            "o3",
            temperature=None,
            top_p=None,
        ),
        reasoning=ReasoningPolicy.off(),
    )

    assert "reasoning_effort" not in body


def test_reasoning_model_omits_sampling_parameters_and_none_effort() -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            "o3",
            temperature=0.7,
            top_p=0.9,
        ),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM),
    )

    assert "temperature" not in body
    assert "top_p" not in body
    assert body["reasoning_effort"] == "medium"


def test_gpt5_uses_reasoning_and_omits_sampling_parameters() -> None:
    body = _provider()._chat._build_request_body(
        make_messages_request(
            "gpt-5",
            temperature=0.7,
            top_p=0.9,
        ),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )

    assert body["model"] == "gpt-5"
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body
    assert "top_p" not in body


def test_finalize_chat_body_sanitizes_both_top_level_and_extra_body() -> None:
    behavior = _provider()._behavior

    o3_body = {
        "model": "o3",
        "temperature": 0.7,
        "top_p": 0.9,
        "reasoning_effort": "none",
        "extra_body": {
            "temperature": 0.5,
            "top_p": 0.8,
            "reasoning_effort": "none",
            "custom_field": "kept",
        },
    }
    sanitized_o3 = behavior.finalize_chat_body(o3_body, reasoning=ReasoningPolicy.off())
    assert "temperature" not in sanitized_o3
    assert "top_p" not in sanitized_o3
    assert "reasoning_effort" not in sanitized_o3
    assert "temperature" not in sanitized_o3["extra_body"]
    assert "top_p" not in sanitized_o3["extra_body"]
    assert "reasoning_effort" not in sanitized_o3["extra_body"]
    assert sanitized_o3["extra_body"]["custom_field"] == "kept"

    gpt4o_body = {
        "model": "gpt-4o",
        "temperature": 0.7,
        "top_p": 0.9,
        "reasoning_effort": "none",
        "extra_body": {
            "reasoning_effort": "none",
            "custom_field": "kept",
        },
    }
    sanitized_gpt4o = behavior.finalize_chat_body(
        gpt4o_body, reasoning=ReasoningPolicy.off()
    )
    assert sanitized_gpt4o["temperature"] == 0.7
    assert sanitized_gpt4o["top_p"] == 0.9
    assert "reasoning_effort" not in sanitized_gpt4o
    assert "reasoning_effort" not in sanitized_gpt4o["extra_body"]
    assert sanitized_gpt4o["extra_body"]["custom_field"] == "kept"


def test_model_predicates_distinguish_chat_and_reasoning_models() -> None:
    assert is_openai_chat_model("gpt-4o")
    assert is_openai_chat_model("gpt-4o-mini")
    assert is_openai_chat_model("gpt-4.5-preview")
    assert is_openai_chat_model("o1")
    assert is_openai_chat_model("o1-mini")
    assert is_openai_chat_model("o3")
    assert is_openai_chat_model("o3-mini")
    assert is_openai_chat_model("o4-preview")
    assert is_openai_chat_model("chatgpt-4o-latest")
    assert is_openai_chat_model("ft:gpt-4o:my-org:custom")
    assert is_openai_chat_model("ft:o3-mini:my-org:custom")

    assert not is_openai_chat_model("text-embedding-3-small")
    assert not is_openai_chat_model("text-embedding-ada-002")
    assert not is_openai_chat_model("dall-e-3")
    assert not is_openai_chat_model("dall-e-2")
    assert not is_openai_chat_model("tts-1")
    assert not is_openai_chat_model("tts-1-hd")
    assert not is_openai_chat_model("whisper-1")
    assert not is_openai_chat_model("text-moderation-latest")
    assert not is_openai_chat_model("omni-moderation-latest")
    assert not is_openai_chat_model("gpt-4o-realtime-preview")
    assert not is_openai_chat_model("gpt-3.5-turbo-instruct")
    assert not is_openai_chat_model("babbage-002")
    assert not is_openai_chat_model("davinci-002")

    assert is_openai_reasoning_model("o1")
    assert is_openai_reasoning_model("o1-mini")
    assert is_openai_reasoning_model("o1-preview")
    assert is_openai_reasoning_model("o3")
    assert is_openai_reasoning_model("o3-mini")
    assert is_openai_reasoning_model("o4")
    assert is_openai_reasoning_model("gpt-5")
    assert is_openai_reasoning_model("gpt-5-mini")
    assert is_openai_reasoning_model("gpt-5.1")
    assert is_openai_reasoning_model("gpt-5.3-codex")
    assert is_openai_reasoning_model("chatgpt-5")
    assert is_openai_reasoning_model("ft:gpt-5:org:custom")
    assert is_openai_reasoning_model("ft:o1:org:id")
    assert not is_openai_reasoning_model("gpt-4o")
    assert not is_openai_reasoning_model("gpt-4o-mini")
    assert not is_openai_reasoning_model("chatgpt-4o-latest")


def test_reasoning_history_replays_as_portable_think_tags() -> None:
    request = make_messages_request(
        _MODEL,
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
async def test_lists_models_from_models_endpoint_filters_non_chat_models() -> None:
    provider = _provider()
    provider._client.get = AsyncMock(
        return_value={
            "data": [
                {"id": "gpt-4o"},
                {"id": "o3-mini"},
                {"id": "text-embedding-3-small"},
                {"id": "dall-e-3"},
                {"id": "whisper-1"},
                {"id": "tts-1"},
                {"id": "omni-moderation-latest"},
                {"id": "gpt-4o-realtime-preview"},
                {"id": "gpt-3.5-turbo-instruct"},
                {"id": "babbage-002"},
            ]
        }
    )

    model_infos = await provider.list_model_infos()

    assert model_infos == frozenset(
        {ProviderModelInfo("gpt-4o"), ProviderModelInfo("o3-mini")}
    )
    provider._client.get.assert_awaited_once()
    call = provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
async def test_rejects_empty_model_ids() -> None:
    provider = _provider()
    provider._client.get = AsyncMock(return_value={"data": [{"id": ""}]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth() -> None:
    provider = _provider()
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL}]})

    await provider._client.close()
    provider._client = AsyncOpenAI(
        api_key="wire-openai-key",
        base_url=OPENAI_API_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-openai-key"
