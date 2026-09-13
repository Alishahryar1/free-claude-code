"""Tests for the W&B Inference OpenAI-chat provider profile."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import WANDB_INFERENCE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def wandb_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "wandb",
        make_provider_config(
            api_key="test-wandb-key",
            base_url=WANDB_INFERENCE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="wandb"),
    )


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def test_init_uses_documented_endpoint(wandb_provider: OpenAIChatProvider) -> None:
    assert wandb_provider._api_key == "test-wandb-key"
    assert wandb_provider._base_url == WANDB_INFERENCE_DEFAULT_BASE
    assert wandb_provider._provider_name == "WANDB"


@pytest.mark.parametrize(
    ("reasoning", "enabled"),
    [(REASONING_OFF, False), (REASONING_ON, True)],
)
def test_build_request_body_encodes_documented_thinking_control(
    wandb_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    enabled: bool,
) -> None:
    body = wandb_provider._chat._build_request_body(_request(), reasoning=reasoning)

    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": enabled}}


def test_build_request_body_uses_provider_defaults_when_reasoning_is_inherited(
    wandb_provider: OpenAIChatProvider,
) -> None:
    body = wandb_provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "extra_body" not in body


def test_build_request_body_drops_undocumented_reasoning_replay_but_keeps_tools(
    wandb_provider: OpenAIChatProvider,
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

    body = wandb_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "[Earlier reasoning]\nRead it first.\n\nI will inspect it.",
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
        wandb_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning="new thought")
        )
        == "new thought"
    )


@pytest.mark.parametrize(
    "field",
    ["reasoning", "reasoning_effort", "thinking", "enable_thinking"],
)
def test_build_request_body_rejects_caller_reasoning_override(
    wandb_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        wandb_provider._chat._build_request_body(request, reasoning=REASONING_ON)


@pytest.mark.asyncio
async def test_wire_body_uses_current_output_limit_and_thinking_fields(
    wandb_provider: OpenAIChatProvider,
) -> None:
    body = wandb_provider._chat._build_request_body(_request(), reasoning=REASONING_ON)

    wire_body = await capture_openai_chat_wire_body(body)

    assert wire_body["max_completion_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert "max_tokens" not in wire_body
    assert wire_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "extra_body" not in wire_body


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_endpoint_and_bearer_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "openai/gpt-oss-20b",
                        "object": "model",
                        "created": 0,
                        "owned_by": "system",
                        "root": "openai/gpt-oss-20b",
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
            "wandb",
            make_provider_config(
                api_key="wire-wandb-key",
                base_url=WANDB_INFERENCE_DEFAULT_BASE,
                rate_limit=10,
                rate_window=60,
            ),
            admission=immediate_admission(provider_name="wandb"),
        )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo("openai/gpt-oss-20b")})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.inference.wandb.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-wandb-key"
    assert "openai-project" not in requests[0].headers
