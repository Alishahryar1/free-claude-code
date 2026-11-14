"""Tests for the TensorX OpenAI-chat gateway profile and catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import TENSORX_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "z-ai/glm-5.2"


@pytest.fixture
def tensorx_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "tensorx",
        make_provider_config(
            api_key="test-tensorx-key",
            base_url=TENSORX_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="tensorx", max_attempts=1),
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
    tensorx_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(tensorx_provider, OpenAIChatProvider)
    assert tensorx_provider._provider_name == "TENSORX"
    assert tensorx_provider._api_key == "test-tensorx-key"
    assert tensorx_provider._base_url == "https://api.tensorx.ai/v1"


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_DEFAULT, None),
        # Qwen 3.8 rejects ``none`` and ``enable_thinking: false``; send nothing.
        (REASONING_OFF, None),
        (REASONING_ON, "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        # Qwen 3.8 rejects ``high``; ``xhigh`` is its documented deepest level.
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "xhigh"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "xhigh"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "xhigh"),
    ],
)
def test_encodes_only_efforts_every_documented_family_accepts(
    tensorx_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str | None,
) -> None:
    body = tensorx_provider._chat._build_request_body(
        _request(),
        reasoning=reasoning,
    )

    assert body.get("reasoning_effort") == expected
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "thinking" not in body
    assert "extra_body" not in body


def test_does_not_inject_max_tokens_that_would_reserve_tpm(
    tensorx_provider: OpenAIChatProvider,
) -> None:
    omitted = tensorx_provider._chat._build_request_body(
        _request(), reasoning=REASONING_DEFAULT
    )
    explicit = tensorx_provider._chat._build_request_body(
        _request(max_tokens=2048), reasoning=REASONING_DEFAULT
    )

    assert "max_tokens" not in omitted
    assert "max_completion_tokens" not in omitted
    assert explicit["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_forwards_caller_chat_template_kwargs_on_the_wire(
    tensorx_provider: OpenAIChatProvider,
) -> None:
    body = tensorx_provider._chat._build_request_body(
        _request(extra_body={"chat_template_kwargs": {"clear_thinking": True}}),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.LOW),
    )

    wire = await capture_openai_chat_wire_body(body)

    assert wire["chat_template_kwargs"] == {"clear_thinking": True}
    assert wire["reasoning_effort"] == "low"


def test_rejects_caller_extra_body_that_overrides_reasoning(
    tensorx_provider: OpenAIChatProvider,
) -> None:
    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        tensorx_provider._chat._build_request_body(
            _request(extra_body={"reasoning_effort": "high"}),
            reasoning=REASONING_DEFAULT,
        )


def test_replays_reasoning_content_with_tool_history(
    tensorx_provider: OpenAIChatProvider,
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

    body = tensorx_provider._chat._build_request_body(
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
        tensorx_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_maps_documented_model_info_metadata(
    tensorx_provider: OpenAIChatProvider,
) -> None:
    tensorx_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "model_name": _MODEL,
                    "model_info": {
                        "mode": "chat",
                        "max_input_tokens": 202752,
                        "max_output_tokens": 131072,
                        "supports_reasoning": True,
                        "supports_function_calling": True,
                        "supports_vision": False,
                        "supports_prompt_caching": True,
                    },
                },
                {
                    "model_name": "z-ai/glm-5.3-flash",
                    "model_info": {
                        "max_input_tokens": 131072,
                        "supports_reasoning": True,
                        "supports_vision": True,
                    },
                },
                {
                    "model_name": "moonshotai/kimi-k2.7-code",
                    "model_info": {
                        "max_input_tokens": 262144,
                        "max_output_tokens": None,
                        "supports_reasoning": False,
                        "supports_vision": None,
                    },
                },
                # Fields that do not apply may be null or absent altogether.
                {"model_name": "qwen/qwen3.8-2.4t-a95b", "model_info": None},
                # LiteLLM lists one row per deployment of the same model group.
                {"model_name": _MODEL, "model_info": {"max_input_tokens": 1}},
            ]
        }
    )

    model_infos = await tensorx_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                supports_thinking=True,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=202752,
                max_output_tokens=131072,
            ),
            ProviderModelInfo(
                "z-ai/glm-5.3-flash",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131072,
            ),
            ProviderModelInfo(
                "moonshotai/kimi-k2.7-code",
                reasoning_capability=ReasoningCapability.NONE,
                supports_thinking=False,
                context_window_tokens=262144,
            ),
            ProviderModelInfo("qwen/qwen3.8-2.4t-a95b"),
        }
    )
    tensorx_provider._client.get.assert_awaited_once_with(
        "/model/info",
        cast_to=object,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{"id": _MODEL}]}, "include model_name"),
        ({"data": []}, "did not include any model ids"),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    tensorx_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    tensorx_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await tensorx_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    tensorx_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"model_name": _MODEL}]})

    await tensorx_provider._client.close()
    tensorx_provider._client = AsyncOpenAI(
        api_key="wire-tensorx-key",
        base_url=TENSORX_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await tensorx_provider.list_model_infos()
    finally:
        await tensorx_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.tensorx.ai/v1/model/info"
    assert requests[0].headers["authorization"] == "Bearer wire-tensorx-key"
