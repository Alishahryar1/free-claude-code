"""Safe default logging tests for the application runtime owner."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.runtime.application import ApplicationRuntime, best_effort
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.mark.asyncio
async def test_messaging_start_failure_default_logs_exclude_traceback(caplog):
    settings = Settings().model_copy(
        update={
            "messaging_platform": "telegram",
            "telegram_bot_token": "t",
            "allowed_telegram_user_id": "1",
            "log_api_error_tracebacks": False,
        }
    )
    runtime = ApplicationRuntime(
        ProviderRuntimeManager(settings),
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
    )

    with (
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            side_effect=RuntimeError("SECRET_RUNTIME_DETAIL"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        await runtime._start_messaging_if_configured()

    blob = " | ".join(record.getMessage() for record in caplog.records)
    assert "SECRET_RUNTIME_DETAIL" not in blob
    assert "exc_type=RuntimeError" in blob


@pytest.mark.asyncio
async def test_best_effort_default_logs_exclude_exception_text(caplog):
    async def boom():
        raise ValueError("SECRET_SHUTDOWN")

    with caplog.at_level(logging.WARNING):
        await best_effort("test_step", boom(), log_verbose_errors=False)

    blob = " | ".join(record.getMessage() for record in caplog.records)
    assert "SECRET_SHUTDOWN" not in blob
    assert "exc_type=ValueError" in blob


@pytest.mark.asyncio
async def test_best_effort_verbose_includes_exception_text(caplog):
    async def boom():
        raise ValueError("VISIBLE_SHUTDOWN")

    with caplog.at_level(logging.WARNING):
        await best_effort("test_step", boom(), log_verbose_errors=True)

    blob = " | ".join(record.getMessage() for record in caplog.records)
    assert "VISIBLE_SHUTDOWN" in blob
