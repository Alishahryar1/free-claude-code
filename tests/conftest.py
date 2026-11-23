import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Unpack
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from free_claude_code.config import env_migrations, paths
from free_claude_code.config.loader import clear_settings_cache
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.harnesses import (
    antigravity,
    claude_desktop_integration,
    claude_integration,
    claude_onboarding,
    codex_integration,
    jetbrains_acp_integration,
    vscode_chat_integration,
)
from free_claude_code.messaging.models import IncomingMessage
from free_claude_code.providers.base import ProviderConfig
from tests.fixture_types import (
    DatabaseFactory,
    IncomingMessageFactory,
    IncomingMessageOverrides,
    MessagingStoreFactory,
)
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
def _response_delivery_context(request: pytest.FixtureRequest):
    """Direct provider tests run inside an unreleased private response context."""
    if "api" in request.node.path.relative_to(Path(__file__).parent).parts:
        yield
        return
    with bind_stream_delivery(StreamDeliveryState()):
        yield


@pytest.fixture(autouse=True)
def _isolate_managed_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Keep every test away from real home, checkout, and running-server config."""

    config_dir = tmp_path / ".fcc"
    monkeypatch.setattr(
        antigravity,
        "config_path",
        lambda: tmp_path / ".gemini/antigravity-cli/settings.json",
    )
    monkeypatch.setenv("DSH_HOME", str(tmp_path / ".dsh"))
    monkeypatch.delenv("FCC_DSH_DESKTOP_API_KEY", raising=False)
    monkeypatch.setattr(
        vscode_chat_integration,
        "config_path",
        lambda: tmp_path / "vscode/chatLanguageModels.json",
    )
    monkeypatch.setattr(
        jetbrains_acp_integration,
        "config_path",
        lambda: tmp_path / ".jetbrains/acp.json",
    )
    monkeypatch.setattr(
        jetbrains_acp_integration,
        "registry_path",
        lambda: tmp_path / "jetbrains/installed.json",
    )
    monkeypatch.setattr(
        jetbrains_acp_integration, "system_root", lambda: tmp_path / "jetbrains/systems"
    )
    monkeypatch.setattr(
        claude_desktop_integration, "config_root", lambda: tmp_path / "Claude-3p"
    )
    monkeypatch.setattr(claude_desktop_integration, "check_unmanaged", lambda: None)
    monkeypatch.setattr(
        claude_desktop_integration,
        "legacy_windows_root",
        lambda: tmp_path / "LegacyClaude-3p",
    )
    monkeypatch.setattr(paths, "config_dir_path", lambda: config_dir)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    monkeypatch.setattr(
        claude_integration, "settings_path", lambda: tmp_path / "vscode/settings.json"
    )
    monkeypatch.setattr(
        claude_onboarding, "state_path", lambda: tmp_path / ".claude.json"
    )
    monkeypatch.setattr(
        codex_integration, "config_path", lambda: tmp_path / ".codex/config.toml"
    )
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def provider_config():

    return make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        http_read_timeout=300.0,
        http_write_timeout=10.0,
        http_connect_timeout=10.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )


@pytest.fixture
def nim_provider(provider_config: ProviderConfig):
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(
        provider_config,
        admission=immediate_admission(),
    )


@pytest.fixture
def open_router_provider(provider_config: ProviderConfig):
    from free_claude_code.providers.open_router import OpenRouterProvider

    return OpenRouterProvider(provider_config, admission=immediate_admission())


@pytest.fixture
def lmstudio_provider(provider_config: ProviderConfig):
    from free_claude_code.providers.lmstudio import LMStudioProvider

    lmstudio_config = make_provider_config(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        http_read_timeout=provider_config.http_read_timeout,
        http_write_timeout=provider_config.http_write_timeout,
        http_connect_timeout=provider_config.http_connect_timeout,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    return LMStudioProvider(lmstudio_config, admission=immediate_admission())


@pytest.fixture
def llamacpp_provider(provider_config: object):
    from free_claude_code.providers.openai_chat import create_openai_chat_provider

    llamacpp_config = make_provider_config(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1",
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

    def _fire_and_forget(task: object):
        if asyncio.iscoroutine(task):
            # Create a task to avoid "coroutine was never awaited" warning
            return asyncio.create_task(task)
        return None

    platform.fire_and_forget = MagicMock(side_effect=_fire_and_forget)
    return platform


@pytest.fixture
def mock_session_store():
    from free_claude_code.messaging.trees import ConversationSnapshot, MessagingStore

    store = AsyncMock(spec=MessagingStore)
    tracked_message_ids: list[str] = []
    store.get_tracked_message_ids_for_chat.return_value = tracked_message_ids
    store.load_conversation_snapshot.return_value = ConversationSnapshot()
    return store


@pytest.fixture
def incoming_message_factory() -> IncomingMessageFactory:
    def _create(**kwargs: Unpack[IncomingMessageOverrides]) -> IncomingMessage:
        message = IncomingMessage(
            text=kwargs.get("text", "hello"),
            chat_id=kwargs.get("chat_id", "chat_1"),
            user_id=kwargs.get("user_id", "user_1"),
            message_id=kwargs.get("message_id", "msg_1"),
            platform=kwargs.get("platform", "telegram"),
            reply_to_message_id=kwargs.get("reply_to_message_id"),
            message_thread_id=kwargs.get("message_thread_id"),
            username=kwargs.get("username"),
            raw_event=kwargs.get("raw_event"),
            status_message_id=kwargs.get("status_message_id"),
        )
        timestamp = kwargs.get("timestamp")
        if isinstance(timestamp, str):
            message.timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is not None:
            message.timestamp = timestamp
        return message

    return _create


@pytest.fixture(autouse=True)
def _propagate_loguru_to_caplog(caplog: pytest.LogCaptureFixture):
    """Capture Loguru directly without re-entering the stdlib interceptor."""
    from loguru import logger as loguru_logger

    handler_id = loguru_logger.add(
        caplog.handler,
        format="{message}",
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield
    with contextlib.suppress(ValueError):
        loguru_logger.remove(
            handler_id
        )  # Handler already removed (e.g. by test_logging_config)


@pytest_asyncio.fixture
async def messaging_store_factory(
    tmp_path: Path,
) -> AsyncIterator[MessagingStoreFactory]:
    """Real SQLite stores with fixture-owned lifetime and optional old JSON input."""
    from free_claude_code.runtime.messaging_import import import_legacy
    from free_claude_code.runtime.messaging_sqlite import SQLiteMessagingStore
    from free_claude_code.runtime.sqlite_database import SQLiteDatabase

    databases: dict[Path, SQLiteDatabase] = {}

    async def create(
        *,
        storage_path: str | Path | None = None,
        managed_message_cap: int | None = None,
    ):
        path = Path(storage_path) if storage_path is not None else tmp_path / "fcc.db"
        database_path = path.with_suffix(".db")
        database = databases.get(database_path)
        first = database is None
        if first:
            database = SQLiteDatabase(database_path, database_path.with_suffix(".lock"))
            await database.start()
            databases[database_path] = database
        store = SQLiteMessagingStore(database, managed_message_cap=managed_message_cap)
        if first and path.suffix == ".json":
            await import_legacy(database, path)
            await store.trim()
        return store

    try:
        yield create
    finally:
        for database in databases.values():
            await database.close()


@pytest_asyncio.fixture
async def database_factory() -> AsyncIterator[DatabaseFactory]:
    """Own database resources for tests that compose individual services."""
    from free_claude_code.runtime.sqlite_database import SQLiteDatabase

    databases = []

    def create(path: Path, lock_path: Path, *, legacy_path: Path | None = None):
        database = SQLiteDatabase(path, lock_path, legacy_path=legacy_path)
        databases.append(database)
        return database

    try:
        yield create
    finally:
        for database in reversed(databases):
            await database.close()
