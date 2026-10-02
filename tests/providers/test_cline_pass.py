"""Tests for the ClinePass OpenAI-chat profile and dynamic model catalog."""

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import CLINE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


@pytest.fixture
def cline_pass_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "cline_pass",
        make_provider_config(
            api_key="test-cline-key",
            base_url=CLINE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="cline_pass"),
    )


def _provider_with_transport(
    transport: httpx.AsyncBaseTransport,
    *,
    api_key: str = "wire-cline-key",
) -> OpenAIChatProvider:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=CLINE_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    )
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI",
        return_value=client,
    ):
        return profiled_provider(
            "cline_pass",
            make_provider_config(
                api_key=api_key,
                base_url=CLINE_DEFAULT_BASE,
                rate_limit=10,
                rate_window=60,
            ),
            admission=immediate_admission(provider_name="cline_pass"),
        )


def _request(**overrides: JsonValue) -> MessagesRequest:
    payload: JsonObject = {
        "model": "cline-pass/kimi-k3",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


class AsyncStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[SimpleNamespace]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    reasoning_details: list[JsonObject] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_details=reasoning_details,
        tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_init_uses_documented_cline_api(
    cline_pass_provider: OpenAIChatProvider,
) -> None:
    assert cline_pass_provider._api_key == "test-cline-key"
    assert cline_pass_provider._base_url == CLINE_DEFAULT_BASE
    assert cline_pass_provider._provider_name == "CLINE_PASS"
    assert cline_pass_provider._profile.user_agent is None


def test_build_request_body_preserves_nested_model_images_tools_and_results(
    cline_pass_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        max_tokens=321,
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
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_inspect",
                        "name": "inspect_image",
                        "input": {"detail": "high"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_inspect",
                        "content": "image contents",
                    }
                ],
            },
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
        tool_choice={"type": "tool", "name": "inspect_image"},
        extra_body={"provider_option": "must-not-pass-through"},
    )

    body = cline_pass_provider._build_request_body(
        request,
        reasoning=ReasoningPolicy.on(),
    )

    assert body["model"] == "cline-pass/kimi-k3"
    assert body["max_tokens"] == 321
    assert "max_completion_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Inspect this image."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
    ]
    assert body["messages"][2]["tool_calls"] == [
        {
            "id": "call_inspect",
            "type": "function",
            "function": {
                "name": "inspect_image",
                "arguments": '{"detail": "high"}',
            },
        }
    ]
    assert body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_inspect",
        "content": "image contents",
    }
    assert body["tools"][0]["function"]["name"] == "inspect_image"
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "inspect_image"},
    }
    assert "extra_body" not in body


@pytest.mark.parametrize(
    "reasoning",
    [REASONING_DEFAULT, REASONING_OFF, REASONING_ON],
)
def test_build_request_body_adds_no_undocumented_reasoning_control_or_default_cap(
    cline_pass_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = cline_pass_provider._build_request_body(
        _request(),
        reasoning=reasoning,
    )

    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert "extra_body" not in body
    for field in (
        "reasoning",
        "reasoning_effort",
        "thinking",
        "enable_thinking",
        "chat_template_kwargs",
    ):
        assert field not in body


def test_build_request_body_replays_only_opaque_reasoning_details(
    cline_pass_provider: OpenAIChatProvider,
) -> None:
    detail: JsonObject = {
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
                    {"type": "text", "text": "I will inspect it."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ]
    )

    body = cline_pass_provider._build_request_body(
        request,
        reasoning=ReasoningPolicy.on(),
    )
    assistant = next(
        message for message in body["messages"] if message["role"] == "assistant"
    )

    assert assistant["content"] == "I will inspect it."
    assert assistant["reasoning_details"] == [detail]
    assert "reasoning" not in assistant
    assert "reasoning_content" not in assistant
    assert (
        cline_pass_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning="next thought", reasoning_content="ignored")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_stream_uses_upstream_sse_and_preserves_reasoning_details(
    cline_pass_provider: OpenAIChatProvider,
) -> None:
    detail: JsonObject = {
        "type": "reasoning.text",
        "text": "plan ",
        "signature": "signed-opaque-value",
        "format": "anthropic-claude-v1",
        "index": 0,
    }
    stream = AsyncStream(
        [
            _chunk(reasoning="plan ", reasoning_details=[detail]),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    create = AsyncMock(return_value=stream)
    with patch.object(
        cline_pass_provider._client.chat.completions,
        "create",
        create,
    ):
        event_text = "".join(
            [
                event
                async for event in cline_pass_provider.stream_response(
                    _request(),
                    reasoning=ReasoningPolicy.on(),
                )
            ]
        )

    events = parse_sse_text(event_text)
    await_args = create.await_args
    assert await_args is not None
    assert await_args.kwargs["stream"] is True
    assert await_args.kwargs["model"] == "cline-pass/kimi-k3"
    assert thinking_content(events) == "plan "
    assert text_content(events) == "done"
    redacted_blocks = [
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "redacted_thinking"
    ]
    assert len(redacted_blocks) == 1
    assert json.loads(redacted_blocks[0]["data"]) == detail
    assert stream.closed


@pytest.mark.asyncio
async def test_model_catalog_uses_cline_pass_collection_endpoint_and_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "recommended": [{"id": "anthropic/claude-sonnet-4-6"}],
                "free": [{"id": "cline/free-model"}],
                "clinePass": [
                    {"id": "cline-pass/kimi-k3"},
                    {"id": "cline-pass/deepseek-v4-flash"},
                ],
                "clineCloud": [{"id": "anthropic/claude-opus-4-6"}],
            },
        )

    provider = _provider_with_transport(
        httpx.MockTransport(handler),
    )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("cline-pass/kimi-k3"),
            ProviderModelInfo("cline-pass/deepseek-v4-flash"),
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://api.cline.bot/api/v1/ai/cline/recommended-models"
    )
    assert requests[0].headers["authorization"] == "Bearer wire-cline-key"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"recommended": []}, "top-level clinePass array"),
        ({"clinePass": []}, "did not include any model ids"),
        ({"clinePass": [{}]}, "clinePass item to include id"),
    ],
)
@pytest.mark.asyncio
async def test_model_catalog_rejects_malformed_or_empty_cline_pass_collection(
    payload: JsonObject,
    message: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    provider = _provider_with_transport(httpx.MockTransport(handler))

    try:
        with pytest.raises(ModelListResponseError, match=message):
            await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert [str(request.url) for request in requests] == [
        "https://api.cline.bot/api/v1/ai/cline/recommended-models"
    ]
    assert requests[0].headers["authorization"] == ("Bearer wire-cline-key")
