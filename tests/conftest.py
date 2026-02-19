import asyncio
import contextlib
import logging
import os
import sys

import pytest

# Set mock environment BEFORE any imports that use Settings
os.environ.setdefault("NVIDIA_NIM_API_KEY", "test_key")
os.environ.setdefault("MODEL", "test-model")
os.environ["PTB_TIMEDELTA"] = "1"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from config.nim import NimSettings
from messaging.base import CLISession, MessagingPlatform, SessionManagerInterface
from messaging.models import IncomingMessage
from messaging.session import SessionStore
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def nim_provider(provider_config):
    return NvidiaNimProvider(provider_config, nim_settings=NimSettings())


@pytest.fixture
def open_router_provider(provider_config):
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(provider_config)


@pytest.fixture
def lmstudio_provider(provider_config):
    from providers.lmstudio import LMStudioProvider

    lmstudio_config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=provider_config.rate_limit,
        rate_window=provider_config.rate_window,
    )
    return LMStudioProvider(lmstudio_config)


@pytest.fixture
def mock_cli_session():
    session = MagicMock(spec=CLISession)
    session.start_task = MagicMock()  # This will return an async generator
    session.is_busy = False
    return session


@pytest.fixture
def mock_cli_manager():
    manager = MagicMock(spec=SessionManagerInterface)
    manager.get_or_create_session = AsyncMock()
    manager.register_real_session_id = AsyncMock(return_value=True)
    manager.stop_all = AsyncMock()
    manager.remove_session = AsyncMock(return_value=True)
    manager.get_stats = MagicMock(return_value={"active_sessions": 0})
    return manager


@pytest.fixture
def mock_platform():
    platform = MagicMock(spec=MessagingPlatform)
    platform.send_message = AsyncMock(return_value="msg_123")
    platform.edit_message = AsyncMock()
    platform.delete_message = AsyncMock()
    platform.queue_send_message = AsyncMock(return_value="msg_123")
    platform.queue_edit_message = AsyncMock()
    platform.queue_delete_message = AsyncMock()

    def _fire_and_forget(task):
        if asyncio.iscoroutine(task):
            # Create a task to avoid "coroutine was never awaited" warning
            return asyncio.create_task(task)
        return None

    platform.fire_and_forget = MagicMock(side_effect=_fire_and_forget)
    return platform


@pytest.fixture
def mock_session_store():
    store = MagicMock(spec=SessionStore)
    store.save_tree = MagicMock()
    store.get_tree = MagicMock(return_value=None)
    store.register_node = MagicMock()
    store.clear_all = MagicMock()
    store.record_message_id = MagicMock()
    store.get_message_ids_for_chat = MagicMock(return_value=[])
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
