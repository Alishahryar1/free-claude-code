"""Tests for the Agnes AI OpenAI-chat provider profile."""

from typing import Any
from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import AGNES_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def agnes_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "agnes",
        make_provider_config(
            api_key="test-agnes-key",
            base_url=AGNES_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="agnes"),
    )


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": "agnes-2.0-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def test_init_uses_documented_endpoint(
    agnes_provider: OpenAIChatProvider,
) -> None:
    assert agnes_provider._api_key == "test-agnes-key"
    assert agnes_provider._base_url == AGNES_DEFAULT_BASE
    assert agnes_provider._provider_name == "AGNES"


def test_build_request_body_preserves_common_chat_tools_and_images(
    agnes_provider: OpenAIChatProvider,
) -> None:
    request = _request(
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

    body = agnes_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "agnes-2.0-flash"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Inspect this image."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
    ]
    assert body["tools"][0]["function"]["name"] == "inspect_image"


@pytest.mark.parametrize(
    ("reasoning", "enabled"),
    [(REASONING_OFF, False), (REASONING_ON, True)],
)
def test_build_request_body_encodes_documented_thinking_control(
    agnes_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    enabled: bool,
) -> None:
    request = _request()

    body = agnes_provider._chat._build_request_body(request, reasoning=reasoning)

    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": enabled}}


def test_build_request_body_omits_thinking_control_for_provider_default(
    agnes_provider: OpenAIChatProvider,
) -> None:
    request = _request()

    body = agnes_provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "extra_body" not in body


def test_build_request_body_replays_reasoning_in_content_not_undocumented_field(
    agnes_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "Solve it."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Work through it."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ]
    )

    body = agnes_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assistant = body["messages"][1]
    assert assistant == {
        "role": "assistant",
        "content": "<think>\nWork through it.\n</think>\n\nThe answer is 42.",
    }


@pytest.mark.parametrize("field", ["reasoning", "reasoning_effort"])
def test_build_request_body_rejects_caller_reasoning_override(
    agnes_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        agnes_provider._chat._build_request_body(request, reasoning=REASONING_ON)


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
                        "id": "agnes-2.0-flash",
                        "object": "model",
                        "created": 0,
                        "owned_by": "agnes-ai",
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
            "agnes",
            make_provider_config(
                api_key="wire-agnes-key",
                base_url=AGNES_DEFAULT_BASE,
                rate_limit=10,
                rate_window=60,
            ),
            admission=immediate_admission(provider_name="agnes"),
        )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo("agnes-2.0-flash")})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://apihub.agnes-ai.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-agnes-key"
