"""Tests for NaraRoute's OpenAI-compatible model catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import NARAROUTE_DEFAULT_BASE
from free_claude_code.core.reasoning import ReasoningCapability
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


@pytest.mark.asyncio
async def test_model_catalog_extracts_strict_optional_reasoning_boolean() -> None:
    provider = profiled_provider(
        "nararoute",
        make_provider_config(
            api_key="test-nararoute-key",
            base_url=NARAROUTE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="nararoute"),
    )
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                {"id": "reasoning", "reasoning": True},
                {"id": "plain", "reasoning": False},
                {"id": "unknown"},
                {"id": "malformed", "reasoning": "true"},
            ]
        )
    )
    try:
        infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert infos == frozenset(
        {
            ProviderModelInfo("reasoning", supports_thinking=True),
            ProviderModelInfo(
                "plain",
                supports_thinking=False,
                reasoning_capability=ReasoningCapability.NONE,
            ),
            ProviderModelInfo("unknown"),
            ProviderModelInfo("malformed"),
        }
    )
