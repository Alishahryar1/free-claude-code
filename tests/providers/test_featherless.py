"""Tests for the Featherless AI OpenAI-chat provider profile."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import FEATHERLESS_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "Qwen/Qwen3-32B"


@pytest.fixture
def featherless_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "featherless",
        make_provider_config(
            api_key="test-featherless-key",
            base_url=FEATHERLESS_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(
            provider_name="featherless",
            max_attempts=1,
        ),
    )


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _catalog_model(
    model_id: str = _MODEL,
    *,
    tool_use: object = True,
    gated: object = False,
    available: object = True,
    image_input: object = False,
) -> dict[str, object]:
    return {
        "id": model_id,
        "features": {
            "tool_use": tool_use,
            "image_input": image_input,
        },
        "is_gated": gated,
        "available_on_current_plan": available,
    }


def _page(
    page: object,
    total_pages: object,
    models: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "data": models,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
        },
    }


def test_build_request_body_preserves_shared_chat_contract(
    featherless_provider: OpenAIChatProvider,
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

    body = featherless_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == _MODEL
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
    featherless_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    enabled: bool,
) -> None:
    body = featherless_provider._chat._build_request_body(
        _request(),
        reasoning=reasoning,
    )

    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": enabled}}


def test_build_request_body_omits_thinking_control_for_provider_default(
    featherless_provider: OpenAIChatProvider,
) -> None:
    body = featherless_provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "extra_body" not in body


@pytest.mark.parametrize(
    "field",
    ["reasoning", "reasoning_effort", "thinking", "enable_thinking"],
)
def test_build_request_body_rejects_caller_reasoning_override(
    featherless_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        featherless_provider._chat._build_request_body(request, reasoning=REASONING_ON)


def test_build_request_body_replays_reasoning_and_tool_history(
    featherless_provider: OpenAIChatProvider,
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

    body = featherless_provider._chat._build_request_body(
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


@pytest.mark.asyncio
async def test_catalog_fetches_all_pages_filters_strictly_and_deduplicates_overlap(
    featherless_provider: OpenAIChatProvider,
) -> None:
    featherless_provider._client.get = AsyncMock(
        side_effect=[
            _page(
                1,
                2,
                [
                    _catalog_model(image_input=True),
                    _catalog_model("gated", gated=True),
                    _catalog_model("unavailable", available=False),
                ],
            ),
            _page(
                2,
                2,
                [
                    _catalog_model(),
                    _catalog_model("plain-agent"),
                    _catalog_model("no-tools", tool_use=False),
                ],
            ),
        ]
    )

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        model_infos = await featherless_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "plain-agent",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )
    assert featherless_provider._client.get.await_count == 2
    for index, call in enumerate(
        featherless_provider._client.get.await_args_list,
        start=1,
    ):
        assert call.args == ("/models",)
        assert call.kwargs["cast_to"] is object
        assert call.kwargs["options"]["params"] == {
            "capabilities": "chat,tool-use",
            "available_on_current_plan": "true",
            "status": "active",
            "per_page": "1000",
            "page": str(index),
        }
    attempt_rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event", "").startswith("provider.attempt.")
    ]
    assert [row["event"] for row in attempt_rows] == [
        "provider.attempt.started",
        "provider.attempt.resolved",
        "provider.attempt.started",
        "provider.attempt.resolved",
    ]
    assert {row["attempt"] for row in attempt_rows} == {1}
    assert len({row["execution_id"] for row in attempt_rows}) == 2


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"id": _MODEL}, "features.tool_use as bool"),
        (_catalog_model(tool_use=1), "features.tool_use as bool"),
        (_catalog_model(gated="false"), "is_gated as bool"),
        (_catalog_model(available=None), "available_on_current_plan as bool"),
        (_catalog_model(""), "include id"),
    ],
)
@pytest.mark.asyncio
async def test_catalog_rejects_missing_or_wrong_typed_metadata_atomically(
    featherless_provider: OpenAIChatProvider,
    model: dict[str, object],
    message: str,
) -> None:
    featherless_provider._client.get = AsyncMock(return_value=_page(1, 1, [model]))

    with pytest.raises(ModelListResponseError, match=message):
        await featherless_provider.list_model_infos()


@pytest.mark.asyncio
async def test_catalog_validates_overlapping_records_before_deduplicating(
    featherless_provider: OpenAIChatProvider,
) -> None:
    featherless_provider._client.get = AsyncMock(
        side_effect=[
            _page(1, 2, [_catalog_model()]),
            _page(2, 2, [{"id": _MODEL}]),
        ]
    )

    with pytest.raises(ModelListResponseError, match=r"features\.tool_use as bool"):
        await featherless_provider.list_model_infos()


@pytest.mark.parametrize("image_input", [None, "yes", 1, []])
@pytest.mark.asyncio
async def test_catalog_degrades_invalid_optional_image_capability_to_unknown(
    featherless_provider: OpenAIChatProvider,
    image_input: object,
) -> None:
    model = _catalog_model(image_input=image_input)
    featherless_provider._client.get = AsyncMock(return_value=_page(1, 1, [model]))

    assert await featherless_provider.list_model_infos() == frozenset(
        {ProviderModelInfo(_MODEL)}
    )


@pytest.mark.parametrize(
    ("current_page", "total_pages", "message"),
    [
        (False, 1, "current_page to be an integer"),
        (1, True, "total_pages to be an integer"),
        (1, 0, "total_pages between 1 and 100"),
        (1, 101, "total_pages between 1 and 100"),
    ],
)
@pytest.mark.asyncio
async def test_catalog_rejects_invalid_pagination_metadata(
    featherless_provider: OpenAIChatProvider,
    current_page: object,
    total_pages: object,
    message: str,
) -> None:
    featherless_provider._client.get = AsyncMock(
        return_value=_page(current_page, total_pages, [_catalog_model()])
    )

    with pytest.raises(ModelListResponseError, match=message):
        await featherless_provider.list_model_infos()


@pytest.mark.parametrize(
    ("second_page", "message"),
    [
        (_page(1, 2, [_catalog_model("second")]), "current_page to be 2"),
        (_page(2, 3, [_catalog_model("second")]), "total_pages to remain 2"),
    ],
)
@pytest.mark.asyncio
async def test_catalog_rejects_inconsistent_later_page_metadata(
    featherless_provider: OpenAIChatProvider,
    second_page: dict[str, object],
    message: str,
) -> None:
    featherless_provider._client.get = AsyncMock(
        side_effect=[
            _page(1, 2, [_catalog_model()]),
            second_page,
        ]
    )

    with pytest.raises(ModelListResponseError, match=message):
        await featherless_provider.list_model_infos()


@pytest.mark.asyncio
async def test_later_page_failure_cannot_return_a_partial_catalog(
    featherless_provider: OpenAIChatProvider,
) -> None:
    featherless_provider._client.get = AsyncMock(
        side_effect=[
            _page(1, 2, [_catalog_model()]),
            RuntimeError("page two failed"),
        ]
    )

    with pytest.raises(RuntimeError, match="page two failed"):
        await featherless_provider._fetch_models_payload()


@pytest.mark.asyncio
async def test_catalog_uses_documented_url_query_and_bearer_auth(
    featherless_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        page = int(request.url.params["page"])
        return httpx2.Response(
            200,
            json=_page(
                page,
                2,
                [_catalog_model(_MODEL if page == 1 else "second-agent")],
            ),
        )

    await featherless_provider._client.close()
    featherless_provider._client = AsyncOpenAI(
        api_key="wire-featherless-key",
        base_url=FEATHERLESS_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await featherless_provider.list_model_infos()
    finally:
        await featherless_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
            ProviderModelInfo(
                "second-agent",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )
    assert len(requests) == 2
    assert [request.url.params["page"] for request in requests] == ["1", "2"]
    for request in requests:
        assert str(request.url.copy_remove_param("page")).startswith(
            "https://api.featherless.ai/v1/models?"
        )
        assert request.url.params["capabilities"] == "chat,tool-use"
        assert request.url.params["available_on_current_plan"] == "true"
        assert request.url.params["status"] == "active"
        assert request.url.params["per_page"] == "1000"
        assert "gated" not in request.url.params
        assert request.headers["authorization"] == "Bearer wire-featherless-key"
