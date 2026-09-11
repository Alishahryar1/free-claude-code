import asyncio
import contextlib
import logging
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.config import env_migrations, paths
from free_claude_code.config.loader import clear_settings_cache
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)

# Set deterministic process overrides before any test resolves Settings.
os.environ.setdefault("NVIDIA_NIM_API_KEY", "test_key")
os.environ.setdefault("MODEL", "nvidia_nim/test-model")
os.environ["PTB_TIMEDELTA"] = "1"
# Tests keep proxy authentication disabled unless a case enables it explicitly.
os.environ["ANTHROPIC_AUTH_TOKEN"] = ""


@pytest.fixture(autouse=True)
def _isolate_managed_config(monkeypatch, tmp_path):
    """Keep every test away from real home, checkout, and running-server config."""

    config_dir = tmp_path / ".fcc"
    monkeypatch.setattr(paths, "config_dir_path", lambda: config_dir)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    monkeypatch.setattr(
        LocalInstallations,
        "current",
        classmethod(
            lambda cls: LocalInstallations(
                tmp_path, "linux", {}, tmp_path / "python", ()
            )
        ),
    )
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def provider_config():

    return make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
        max_concurrency=5,
        http_read_timeout=300.0,
        http_write_timeout=10.0,
        http_connect_timeout=10.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )


@pytest.fixture
def nim_provider(provider_config):
    from free_claude_code.config.nim import NimSettings
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )


@pytest.fixture
def open_router_provider(provider_config):
    from free_claude_code.providers.open_router import OpenRouterProvider

    return OpenRouterProvider(provider_config, admission=immediate_admission())


@pytest.fixture
def lmstudio_provider(provider_config):
    from free_claude_code.providers.lmstudio import LMStudioProvider

    lmstudio_config = make_provider_config(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=provider_config.rate_limit,
        rate_window=provider_config.rate_window,
        max_concurrency=provider_config.max_concurrency,
        http_read_timeout=provider_config.http_read_timeout,
        http_write_timeout=provider_config.http_write_timeout,
        http_connect_timeout=provider_config.http_connect_timeout,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    return LMStudioProvider(lmstudio_config, admission=immediate_admission())


@pytest.fixture
def llamacpp_provider(provider_config):
    from free_claude_code.providers.openai_chat import create_openai_chat_provider

    llamacpp_config = make_provider_config(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1",
        rate_limit=10,
        rate_window=60,
        max_concurrency=5,
        http_read_timeout=300.0,
        http_write_timeout=10.0,
        http_connect_timeout=10.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    return create_openai_chat_provider(
        "llamacpp",
        llamacpp_config,
        immediate_admission(),
    )


@pytest.fixture
def mock_cli_session():
    from free_claude_code.messaging.managed_protocols import (
        ManagedClaudeSessionProtocol,
    )

    session = MagicMock(spec=ManagedClaudeSessionProtocol)
    session.start_task = MagicMock()  # This will return an async generator
    session.is_busy = False
    return session


@pytest.fixture
def mock_cli_manager():
    from free_claude_code.messaging.managed_protocols import (
        ManagedClaudeSessionManagerProtocol,
    )

    manager = MagicMock(spec=ManagedClaudeSessionManagerProtocol)
    manager.get_or_create_session = AsyncMock()
    manager.register_real_session_id = AsyncMock(return_value=True)
    manager.stop_all = AsyncMock()
    manager.remove_session = AsyncMock(return_value=True)
    manager.get_stats = MagicMock(return_value={"active_sessions": 0})
    return manager


@pytest.fixture
def mock_platform():
    from free_claude_code.messaging.platforms.ports import OutboundMessenger

    platform = MagicMock(spec=OutboundMessenger)
    platform.send_message = AsyncMock(return_value="msg_123")
    platform.edit_message = AsyncMock()
    platform.delete_message = AsyncMock()
    platform.queue_send_message = AsyncMock(return_value="msg_123")
    platform.queue_edit_message = AsyncMock()
    platform.queue_delete_messages = AsyncMock()
    platform.cancel_pending_voice = AsyncMock(return_value=None)
    platform.cancel_all_pending_voices = AsyncMock(return_value=())
    platform.cancel_pending_voices_in_scope = AsyncMock(return_value=())

    def _fire_and_forget(task):
        if asyncio.iscoroutine(task):
            # Create a task to avoid "coroutine was never awaited" warning
            return asyncio.create_task(task)
        return None

    platform.fire_and_forget = MagicMock(side_effect=_fire_and_forget)
    return platform


@pytest.fixture
def mock_session_store():
    from free_claude_code.messaging.session import SessionStore

    store = MagicMock(spec=SessionStore)
    store.save_tree = MagicMock()
    store.get_tree = MagicMock(return_value=None)
    store.register_node = MagicMock()
    store.record_message_id = MagicMock()
    store.get_tracked_message_ids_for_chat = MagicMock(return_value=[])
    store.forget_tracked_message_ids = MagicMock()
    store.clear_scope = MagicMock()
    return store


@pytest.fixture
def incoming_message_factory():
    _valid_keys = frozenset(
        {
            "text",
            "chat_id",
            "user_id",
            "message_id",
            "platform",
            "reply_to_message_id",
            "message_thread_id",
            "username",
            "timestamp",
            "raw_event",
            "status_message_id",
        }
    )

    def _create(**kwargs):
        from free_claude_code.messaging.models import IncomingMessage

        defaults: dict[str, Any] = {
            "text": "hello",
            "chat_id": "chat_1",
            "user_id": "user_1",
            "message_id": "msg_1",
            "platform": "telegram",
        }
        defaults.update(kwargs)
        if "timestamp" in defaults and isinstance(defaults["timestamp"], str):
            from datetime import datetime

            defaults["timestamp"] = datetime.fromisoformat(defaults["timestamp"])
        filtered = {k: v for k, v in defaults.items() if k in _valid_keys}
        return IncomingMessage(**filtered)

    return _create


@pytest.fixture(autouse=True)
def _propagate_loguru_to_caplog():
    """Route loguru logs to stdlib logging so pytest caplog captures them."""
    from loguru import logger as loguru_logger

    class _PropagateHandler:
        def write(self, message):
            record = message.record
            level = record["level"].no
            stdlib_level = min(level, logging.CRITICAL)
            py_logger = logging.getLogger(record["name"])
            py_logger.log(stdlib_level, record["message"])

    handler_id = loguru_logger.add(_PropagateHandler(), format="{message}")
    yield
    with contextlib.suppress(ValueError):
        loguru_logger.remove(
            handler_id
        )  # Handler already removed (e.g. by test_logging_config)
