"""Tests for the xKiro OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import XKIRO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "z-ai/glm-5.3"


@pytest.fixture
def xkiro_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "xkiro",
        make_provider_config(
            api_key="test-xkiro-key",
            base_url=XKIRO_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="xkiro", max_attempts=1),
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
    xkiro_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(xkiro_provider, OpenAIChatProvider)
    assert xkiro_provider._provider_name == "XKIRO"
    assert xkiro_provider._api_key == "test-xkiro-key"
    assert xkiro_provider._base_url == XKIRO_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected_effort"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_ON, "high"),
        (REASONING_OFF, "none"),
    ],
)
def test_encodes_client_reasoning_intent_for_the_gateway(
    xkiro_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected_effort: str | None,
) -> None:
    """The gateway translates effort per model, so FCC always states its intent.

    xKiro's documented rule is that omitting the field is not disabling: most
    models default to reasoning on and bill for it. Sending a level the target
    model lacks is adjusted downward, never rejected, so FCC forwards the
    client's intent verbatim.
    """

    body = xkiro_provider._build_request_body(_request(), reasoning=reasoning)

    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert body.get("reasoning_effort") == expected_effort
    assert "thinking" not in body
    assert "extra_body" not in body


def test_replays_reasoning_content_with_tool_history(
    xkiro_provider: OpenAIChatProvider,
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

    body = xkiro_provider._build_request_body(
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


def test_slash_separated_model_ids_pass_through_untouched(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    """xKiro model ids are vendor-prefixed (vendor/model), not aliases."""

    body = xkiro_provider._build_request_body(
        _request(model="openai/gpt-5.3-codex-spark"),
        reasoning=REASONING_OFF,
    )

    assert body["model"] == "openai/gpt-5.3-codex-spark"
    assert body["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_model_catalog_uses_default_listing_shape(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    """The default listing reads GET /models with collection 'data', id 'id'."""

    xkiro_provider._client.models.list = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "z-ai/glm-5.3",
                    "object": "model",
                    "display_name": "GLM 5.3",
                    "access_tier": "paid",
                },
                {
                    "id": "openai/gpt-5.3-codex-spark",
                    "object": "model",
                    "display_name": "GPT 5.3 Codex Spark",
                    "access_tier": "free",
                },
            ]
        }
    )

    model_infos = await xkiro_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("z-ai/glm-5.3"),
            ProviderModelInfo("openai/gpt-5.3-codex-spark"),
        }
    )
