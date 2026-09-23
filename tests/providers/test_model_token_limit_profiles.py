"""Provider-profile contracts for advertised model token limits."""

from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "payload", "context_window_tokens", "max_output_tokens"),
    [
        (
            "vercel",
            {"data": [{"id": "model", "context_window": 131072, "max_tokens": 8192}]},
            131072,
            8192,
        ),
        (
            "chutes",
            {
                "data": [
                    {
                        "id": "model",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "supported_features": ["tools"],
                        "context_length": 32768,
                        "max_output_length": 4096,
                    }
                ]
            },
            32768,
            4096,
        ),
        (
            "featherless",
            {
                "data": [
                    {
                        "id": "model",
                        "features": {"tool_use": True, "image_input": False},
                        "is_gated": False,
                        "available_on_current_plan": True,
                        "context_length": 65536,
                        "max_completion_tokens": 8192,
                    }
                ]
            },
            65536,
            8192,
        ),
        (
            "zenmux",
            {
                "data": [
                    {
                        "id": "model",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "context_length": 200000,
                    }
                ]
            },
            200000,
            None,
        ),
        (
            "deepinfra",
            [
                {
                    "model_name": "model",
                    "reported_type": "text-generation",
                    "deprecated": None,
                    "max_tokens": 131072,
                }
            ],
            131072,
            None,
        ),
        (
            "nebius",
            {
                "data": [
                    {
                        "id": "model",
                        "architecture": {"modality": "text->text"},
                        "context_length": 128000,
                    }
                ]
            },
            128000,
            None,
        ),
        (
            "mistral_codestral",
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {
                            "completion_chat": True,
                            "vision": False,
                        },
                        "max_context_length": 256000,
                    }
                ]
            },
            256000,
            None,
        ),
        (
            "together",
            [{"id": "model", "type": "chat", "context_length": 131072}],
            131072,
            None,
        ),
        (
            "llm7",
            {
                "data": [
                    {
                        "id": "model",
                        "model_type": "chat",
                        "stream": True,
                        "tools_calling": True,
                        "context_window": {"tokens": 114688, "characters": 458752},
                    }
                ]
            },
            114688,
            None,
        ),
    ],
)
async def test_profile_extracts_only_its_documented_token_limit_fields(
    provider_id: str,
    payload: object,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
) -> None:
    provider = profiled_provider(
        provider_id,
        make_provider_config(api_key="test", base_url="https://provider.test/v1"),
        admission=immediate_admission(provider_name=provider_id),
    )
    try:
        with patch.object(
            provider,
            "_list_models_payload",
            new=AsyncMock(return_value=payload),
        ):
            infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    info = next(info for info in infos if info.model_id == "model")
    assert info == ProviderModelInfo(
        "model",
        supports_thinking=info.supports_thinking,
        input_modalities=info.input_modalities,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
    )
    if provider_id == "llm7":
        selectors = {info.model_id: info for info in infos if info.model_id != "model"}
        assert set(selectors) == {"default", "fast", "pro"}
        assert all(
            info.context_window_tokens is None and info.max_output_tokens is None
            for info in selectors.values()
        )
