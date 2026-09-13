"""Tests for the ZenMux OpenAI-chat provider profile."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import ZENMUX_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.history_replay import decode_replay
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def zenmux_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "zenmux",
        make_provider_config(
            api_key="test-zenmux-key",
            base_url=ZENMUX_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="zenmux"),
    )


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": "deepseek/deepseek-v4-flash-free",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


class AsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
        tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_init_uses_documented_endpoint(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    assert zenmux_provider._api_key == "test-zenmux-key"
    assert zenmux_provider._base_url == ZENMUX_DEFAULT_BASE
    assert zenmux_provider._provider_name == "ZENMUX"


def test_build_request_body_uses_supported_output_field_tools_and_images(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        max_tokens=100,
        system="Be concise.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U=",
                        },
                    },
                ],
            }
        ],
        tools=[
            {
                "name": "inspect_image",
                "description": "Inspect an image",
                "input_schema": {
                    "type": "object",
                    "properties": {"detail": {"type": "string"}},
                },
            }
        ],
    )

    body = zenmux_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "deepseek/deepseek-v4-flash-free"
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert body["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
    }
    assert body["tools"][0]["function"]["name"] == "inspect_image"


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (ReasoningEffort.MINIMAL, "minimal"),
        (ReasoningEffort.LOW, "low"),
        (ReasoningEffort.MEDIUM, "medium"),
        (ReasoningEffort.HIGH, "high"),
        (ReasoningEffort.XHIGH, "xhigh"),
        (ReasoningEffort.MAX, "xhigh"),
    ],
)
def test_build_request_body_maps_reasoning_effort_to_documented_vocabulary(
    zenmux_provider: OpenAIChatProvider,
    effort: ReasoningEffort,
    expected: str,
) -> None:
    body = zenmux_provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=effort),
    )

    assert body["extra_body"]["reasoning"] == {"effort": expected}


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (ReasoningPolicy.off(), {"enabled": False}),
        (ReasoningPolicy.on(), {"enabled": True}),
        (ReasoningPolicy.on(budget_tokens=2048), {"max_tokens": 2048}),
    ],
)
def test_build_request_body_maps_reasoning_control_and_budget(
    zenmux_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: dict[str, Any],
) -> None:
    body = zenmux_provider._chat._build_request_body(_request(), reasoning=reasoning)

    assert body["extra_body"]["reasoning"] == expected


def test_build_request_body_leaves_reasoning_to_provider_by_default(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    body = zenmux_provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "reasoning" not in body.get("extra_body", {})


def test_build_request_body_preserves_unrelated_gateway_options(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    request = _request(extra_body={"provider": {"fallback": "true"}})

    body = zenmux_provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body["extra_body"] == {"provider": {"fallback": "true"}}


@pytest.mark.parametrize(
    "field",
    ["model", "messages", "max_completion_tokens", "reasoning", "reasoning_effort"],
)
def test_build_request_body_rejects_caller_canonical_override(
    zenmux_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override canonical"):
        zenmux_provider._chat._build_request_body(
            request,
            reasoning=ReasoningPolicy.on(),
        )


def test_build_request_body_replays_reasoning_and_signed_details_unchanged(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    detail = {
        "type": "reasoning.text",
        "text": "Need a tool.",
        "signature": "signed-opaque-value",
        "format": "anthropic-claude-v1",
        "index": 0,
    }
    request = _request(
        messages=[
            {"role": "user", "content": "Inspect the repository."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need a tool."},
                    {"type": "redacted_thinking", "data": json.dumps(detail)},
                    {
                        "type": "tool_use",
                        "id": "call_read",
                        "name": "Read",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_read",
                        "content": "contents",
                    }
                ],
            },
        ]
    )

    body = zenmux_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )
    assert assistant["reasoning"] == "Need a tool."
    assert assistant["reasoning_details"] == [detail]


def test_reasoning_delta_prefers_reasoning_and_falls_back_to_reasoning_content(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    profile = zenmux_provider._profile

    assert (
        profile.reasoning_delta(
            SimpleNamespace(reasoning="primary", reasoning_content="fallback")
        )
        == "primary"
    )
    assert (
        profile.reasoning_delta(
            SimpleNamespace(reasoning=None, reasoning_content="fallback")
        )
        == "fallback"
    )
    assert (
        profile.reasoning_delta(
            SimpleNamespace(reasoning="", reasoning_content="fallback")
        )
        == "fallback"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native_reasoning", ["plan ", None])
async def test_stream_preserves_signed_details_without_duplicating_reasoning(
    zenmux_provider: OpenAIChatProvider,
    native_reasoning: str | None,
) -> None:
    detail = {
        "type": "reasoning.text",
        "text": "plan ",
        "signature": "signed-opaque-value",
        "format": "anthropic-claude-v1",
        "index": 0,
    }
    stream = AsyncStream(
        [
            _chunk(
                reasoning=native_reasoning,
                reasoning_details=[detail],
            ),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    with patch.object(
        zenmux_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        event_text = "".join(
            [event async for event in zenmux_provider.stream_messages(_request())]
        )

    events = parse_sse_text(event_text)
    assert thinking_content(events) == "plan "
    assert text_content(events) == "done"
    records = [
        decode_replay(event.data["delta"]["signature"]).native
        for event in events
        if event.data.get("delta", {}).get("type") == "signature_delta"
    ]
    assert len(records) == 1
    assert records[0]["reasoning_details"] == [detail]
    assert stream.closed


@pytest.mark.asyncio
async def test_model_catalog_filters_modalities_and_maps_reasoning_capability(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    payload = SimpleNamespace(
        data=[
            {
                "id": "reasoning-chat",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "capabilities": {"reasoning": True},
            },
            {
                "id": "plain-chat",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "capabilities": {"reasoning": False},
            },
            {
                "id": "unknown-chat",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            {
                "id": "image-generator",
                "input_modalities": ["text"],
                "output_modalities": ["image"],
                "capabilities": {"reasoning": False},
            },
        ]
    )

    with patch.object(
        zenmux_provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=payload,
    ):
        model_infos = await zenmux_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "reasoning-chat",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "plain-chat",
                supports_thinking=False,
                reasoning_capability=ReasoningCapability.NONE,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo(
                "unknown-chat",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_endpoint_and_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "deepseek/deepseek-v4-flash-free",
                        "object": "model",
                        "created": 0,
                        "owned_by": "deepseek",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "capabilities": {"reasoning": True},
                    }
                ],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def build_client(*args: Any, **kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = http_client
        return AsyncOpenAI(*args, **kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        side_effect=build_client,
    ):
        provider = profiled_provider(
            "zenmux",
            make_provider_config(
                api_key="wire-zenmux-key",
                base_url=ZENMUX_DEFAULT_BASE,
                rate_limit=10,
                rate_window=60,
            ),
            admission=immediate_admission(provider_name="zenmux"),
        )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "deepseek/deepseek-v4-flash-free",
                supports_thinking=True,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == "https://zenmux.ai/api/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-zenmux-key"


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            {
                "id": "missing-input-modalities",
                "output_modalities": ["text"],
                "capabilities": {"reasoning": True},
            },
            "input_modalities string array",
        ),
        (
            {
                "id": "bad-output-modalities",
                "input_modalities": ["text"],
                "output_modalities": [7],
                "capabilities": {"reasoning": True},
            },
            "output_modalities string array",
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_catalog_rejects_malformed_documented_fields(
    zenmux_provider: OpenAIChatProvider,
    model: dict[str, Any],
    message: str,
) -> None:
    payload = SimpleNamespace(data=[model])

    with (
        patch.object(
            zenmux_provider._client.models,
            "list",
            new_callable=AsyncMock,
            return_value=payload,
        ),
        pytest.raises(ModelListResponseError, match=message),
    ):
        await zenmux_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_degrades_malformed_optional_reasoning_to_unknown(
    zenmux_provider: OpenAIChatProvider,
) -> None:
    payload = SimpleNamespace(
        data=[
            {
                "id": "bad-reasoning-capability",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "capabilities": {"reasoning": "yes"},
            }
        ]
    )
    with patch.object(
        zenmux_provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=payload,
    ):
        infos = await zenmux_provider.list_model_infos()

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "bad-reasoning-capability",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
    )
