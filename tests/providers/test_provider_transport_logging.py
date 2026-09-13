"""Tests for credential-safe provider transport logging."""

import logging
from unittest.mock import AsyncMock, patch

import httpx2
import openai
import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)


def _provider(*, verbose: bool = False) -> NvidiaNimProvider:
    return NvidiaNimProvider(
        make_provider_config(
            api_key="k",
            base_url="http://localhost:1/v1",
            log_api_error_tracebacks=verbose,
        ),
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )


@pytest.mark.asyncio
async def test_stream_failure_default_logs_exclude_exception_text(caplog) -> None:
    provider = _provider()
    with (
        patch.object(
            provider._chat,
            "_create_stream",
            new_callable=AsyncMock,
            side_effect=RuntimeError("SECRET_OPENAI_COMPAT"),
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(ExecutionFailure),
    ):
        [event async for event in provider.stream_messages(make_messages_request())]

    messages = " | ".join(record.getMessage() for record in caplog.records)
    assert "SECRET_OPENAI_COMPAT" not in messages
    assert "exc_type=RuntimeError" in messages


@pytest.mark.asyncio
async def test_stream_failure_default_logs_cause_types_only(caplog) -> None:
    provider = _provider()
    error = openai.APIConnectionError(
        request=httpx2.Request("POST", "http://localhost:1/v1/chat/completions")
    )
    error.__cause__ = httpx2.ConnectError("SECRET_CAUSE_DETAIL")
    with (
        patch.object(
            provider._chat,
            "_create_stream",
            new_callable=AsyncMock,
            side_effect=error,
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(ExecutionFailure),
    ):
        [event async for event in provider.stream_messages(make_messages_request())]

    messages = " | ".join(record.getMessage() for record in caplog.records)
    assert "SECRET_CAUSE_DETAIL" not in messages
    assert "exc_type=APIConnectionError" in messages
    assert "cause_types=ConnectError" in messages


@pytest.mark.asyncio
async def test_stream_failure_verbose_traceback_redacts_credentials(caplog) -> None:
    provider = _provider(verbose=True)
    with (
        patch.object(
            provider._chat,
            "_create_stream",
            new_callable=AsyncMock,
            side_effect=RuntimeError(
                "api_key=SECRET_OPENAI_COMPAT useful traceback detail"
            ),
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(ExecutionFailure),
    ):
        [event async for event in provider.stream_messages(make_messages_request())]

    messages = " | ".join(record.getMessage() for record in caplog.records)
    assert "api_key=<redacted>" in messages
    assert "useful traceback detail" in messages
    assert "SECRET_OPENAI_COMPAT" not in messages
