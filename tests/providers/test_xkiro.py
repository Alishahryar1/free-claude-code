"""Tests for the xKiro OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import XKIRO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.harnesses.codex_model_catalog import build_codex_model_catalog
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_DEFAULT, {}),
        (REASONING_ON, {"thinking": {"type": "adaptive"}}),
        (REASONING_OFF, {"reasoning_effort": "none"}),
    ],
)
async def test_encodes_client_reasoning_intent_for_the_gateway(
    xkiro_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: JsonObject,
) -> None:
    """No opinion means actual omission, not null or an invented effort."""
    body = xkiro_provider._chat._build_request_body(_request(), reasoning=reasoning)
    wire = await capture_openai_chat_wire_body(body)

    assert wire["model"] == _MODEL
    assert wire["tools"][0]["function"]["name"] == "read_file"
    assert {
        key: wire[key] for key in ("reasoning_effort", "thinking") if key in wire
    } == expected
    assert "extra_body" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", list(ReasoningEffort))
async def test_named_effort_survives_serialization(
    xkiro_provider: OpenAIChatProvider, effort: ReasoningEffort
) -> None:
    request = _request(output_config={"effort": effort.value})
    body = xkiro_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    wire = await capture_openai_chat_wire_body(body)
    assert wire["reasoning_effort"] == effort.value
    assert "thinking" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [4096, 6000, 6001, 13000, 13001, 24000, 24001])
async def test_native_budget_survives_serialization(
    xkiro_provider: OpenAIChatProvider, budget: int
) -> None:
    request = _request(thinking={"type": "enabled", "budget_tokens": budget})
    body = xkiro_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert body["extra_body"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": budget,
    }
    wire = await capture_openai_chat_wire_body(body)
    assert wire["thinking"] == {"type": "enabled", "budget_tokens": budget}
    assert "reasoning_effort" not in wire
    assert "extra_body" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"thinking": {"type": "adaptive"}}, {"thinking": {"type": "adaptive"}}),
        ({"thinking": {"type": "enabled"}}, {"thinking": {"type": "adaptive"}}),
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            {"reasoning_effort": "low"},
        ),
        (
            {
                "thinking": {"type": "enabled", "budget_tokens": 4096},
                "output_config": {"effort": "max"},
            },
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
        ),
        (
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "max"}},
            {"reasoning_effort": "none"},
        ),
        (
            {
                "thinking": {"type": "enabled", "budget_tokens": 4096},
                "output_config": {"effort": "none"},
            },
            {"reasoning_effort": "none"},
        ),
    ],
)
async def test_client_reasoning_precedence_on_wire(
    xkiro_provider: OpenAIChatProvider, overrides: JsonObject, expected: JsonObject
) -> None:
    request = _request(**overrides)
    body = xkiro_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    wire = await capture_openai_chat_wire_body(body)
    assert {
        key: wire[key] for key in ("reasoning_effort", "thinking") if key in wire
    } == expected


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_effort": "high"},
        {"thinking": {"type": "adaptive"}},
        {"model": "other/model"},
    ],
)
def test_extra_body_cannot_override_owned_fields(
    xkiro_provider: OpenAIChatProvider, extra: JsonObject
) -> None:
    with pytest.raises(InvalidRequestError, match="must not override"):
        xkiro_provider._chat._build_request_body(
            _request(extra_body=extra), reasoning=REASONING_OFF
        )


@pytest.mark.asyncio
async def test_budget_merges_with_allowed_extras_without_mutating_request(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        thinking={"type": "enabled", "budget_tokens": 4096}, extra_body={"seed": 7}
    )
    body = xkiro_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    wire = await capture_openai_chat_wire_body(body)
    assert wire["seed"] == 7
    assert wire["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert request.extra_body == {"seed": 7}


def test_tolerant_off_keeps_the_existing_recovery_field(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    assert xkiro_provider._chat._behavior.reasoning_off_fields == (
        ("reasoning_effort",),
    )
    body = xkiro_provider._chat._build_request_body(
        _request(), reasoning=ReasoningPolicy.prefer_off()
    )
    assert body["reasoning_effort"] == "none"
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

    body = xkiro_provider._chat._build_request_body(
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

    body = xkiro_provider._chat._build_request_body(
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


@pytest.mark.asyncio
async def test_catalog_preserves_capabilities_and_limits_for_codex(
    xkiro_provider: OpenAIChatProvider,
) -> None:
    xkiro_provider._client.models.list = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "vision/model",
                    "capabilities": {"vision": True, "reasoning": True},
                    "context_length": 65536,
                    "max_output_tokens": 8192,
                },
                {
                    "id": "text/model",
                    "capabilities": {"vision": False, "reasoning": False},
                    "context_length": 256000,
                    "max_output_tokens": 16384,
                },
                {"id": "unknown/model"},
            ]
        }
    )
    infos = {info.model_id: info for info in await xkiro_provider.list_model_infos()}
    assert infos["vision/model"] == ProviderModelInfo(
        "vision/model",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=65536,
        max_output_tokens=8192,
    )
    assert infos["text/model"] == ProviderModelInfo(
        "text/model",
        supports_thinking=False,
        input_modalities=frozenset({ModelInputModality.TEXT}),
        context_window_tokens=256000,
        max_output_tokens=16384,
        reasoning_capability=ReasoningCapability.NONE,
    )
    assert infos["unknown/model"] == ProviderModelInfo("unknown/model")
    # No reasoning_efforts metadata is needed to establish reasoning support.
    catalog = build_codex_model_catalog(
        [
            CatalogModel(
                wire_slug=f"xkiro/{info.model_id}",
                provider_model_ref=f"xkiro/{info.model_id}",
                display_name=info.model_id,
                supports_reasoning=info.supports_thinking,
                input_modalities=info.input_modalities,
                context_window_tokens=info.context_window_tokens,
                max_output_tokens=info.max_output_tokens,
            )
            for info in infos.values()
        ]
    )
    models = catalog["models"]
    assert isinstance(models, list)
    entries = {entry["slug"]: entry for entry in models if isinstance(entry, dict)}
    vision = entries["xkiro/vision/model"]
    text = entries["xkiro/text/model"]
    assert vision["context_window"] == 65536
    assert vision["input_modalities"] == ["text", "image"]
    assert text["supported_reasoning_levels"] == []
    assert "default_reasoning_level" not in text
