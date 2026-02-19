import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def test_create_app_provider_error_handler_returns_anthropic_format():
    from api.app import create_app
    from providers.exceptions import AuthenticationError

    app = create_app()

    @app.get("/raise_provider")
    async def _raise_provider():
        raise AuthenticationError("bad key")

    api_app_mod = importlib.import_module("api.app")
    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
        log_file="server.log",
    )
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=AsyncMock()),
    ):
        with TestClient(app) as client:
            resp = client.get("/raise_provider")
        assert resp.status_code == 401
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error"


def test_create_app_general_exception_handler_returns_500():
    from api.app import create_app

    app = create_app()

    @app.get("/raise_general")
    async def _raise_general():
        raise RuntimeError("boom")

    api_app_mod = importlib.import_module("api.app")
    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
        log_file="server.log",
    )
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=AsyncMock()),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/raise_general")
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"


@pytest.mark.parametrize(
    "messaging_enabled", [True, False], ids=["with_platform", "no_platform"]
)
def test_app_lifespan_sets_state_and_cleans_up(tmp_path, messaging_enabled):
    from api.app import create_app

    app = create_app()

    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token="token" if messaging_enabled else None,
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
        log_file=str(tmp_path / "server.log"),
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock()
    fake_platform.stop = AsyncMock()

    session_store = MagicMock()
    session_store.get_all_trees.return_value = [{"t": 1}] if messaging_enabled else []
    session_store.get_node_mapping.return_value = {"n": "t"}
    session_store.sync_from_tree_data = MagicMock()

    fake_queue = MagicMock()
    fake_queue.cleanup_stale_nodes.return_value = 1
    fake_queue.to_dict.return_value = {
        "trees": [{"t": 1}],
        "node_to_tree": {"n": "t"},
    }

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")

    cleanup_provider = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=cleanup_provider),
        patch(
            "messaging.factory.create_messaging_platform",
            return_value=fake_platform if messaging_enabled else None,
        ) as create_platform,
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.manager.CLISessionManager", return_value=cli_manager),
        patch(
            "messaging.tree_queue.TreeQueueManager.from_dict", return_value=fake_queue
        ),
        TestClient(app),
    ):
        pass

    if messaging_enabled:
        create_platform.assert_called_once()
        fake_platform.on_message.assert_called_once()
        fake_platform.start.assert_awaited_once()
        fake_platform.stop.assert_awaited_once()
        cli_manager.stop_all.assert_awaited_once()
        assert getattr(app.state, "message_handler", None) is not None
        session_store.sync_from_tree_data.assert_called_once_with(
            [{"t": 1}],
            {"n": "t"},
        )
    else:
        fake_platform.start.assert_not_awaited()
        fake_platform.stop.assert_not_awaited()
        cli_manager.stop_all.assert_not_awaited()
        assert getattr(app.state, "messaging_platform", "missing") is None

    cleanup_provider.assert_awaited_once()


def test_app_lifespan_cleanup_continues_if_platform_stop_raises(tmp_path):
    from api.app import create_app

    app = create_app()

    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
        log_file=str(tmp_path / "server.log"),
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock()
    fake_platform.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

    session_store = MagicMock()
    session_store.get_all_trees.return_value = []
    session_store.get_node_mapping.return_value = {}
    session_store.sync_from_tree_data = MagicMock()

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")
    cleanup_provider = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=cleanup_provider),
        patch(
            "messaging.factory.create_messaging_platform",
            return_value=fake_platform,
        ),
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.manager.CLISessionManager", return_value=cli_manager),
        TestClient(app),
    ):
        pass

    fake_platform.stop.assert_awaited_once()
    cli_manager.stop_all.assert_awaited_once()
    cleanup_provider.assert_awaited_once()


def test_app_lifespan_messaging_import_error_no_crash(tmp_path, caplog):
    """Messaging import failure logs warning and continues without crash."""
    from api.app import create_app

    app = create_app()

    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
        log_file=str(tmp_path / "server.log"),
    )

    api_app_mod = importlib.import_module("api.app")
    cleanup_provider = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=cleanup_provider),
        patch(
            "messaging.factory.create_messaging_platform",
            side_effect=ImportError("discord not installed"),
        ),
        TestClient(app),
    ):
        pass

    assert getattr(app.state, "messaging_platform", None) is None
    cleanup_provider.assert_awaited_once()


def test_app_lifespan_platform_start_exception_cleanup_still_runs(tmp_path):
    """Exception during platform.start() logs error, cleanup still runs."""
    from api.app import create_app

    app = create_app()

    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
        log_file=str(tmp_path / "server.log"),
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock(side_effect=RuntimeError("start failed"))
    fake_platform.stop = AsyncMock()

    session_store = MagicMock()
    session_store.get_all_trees.return_value = []
    session_store.get_node_mapping.return_value = {}
    session_store.sync_from_tree_data = MagicMock()

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")
    cleanup_provider = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=cleanup_provider),
        patch(
            "messaging.factory.create_messaging_platform",
            return_value=fake_platform,
        ),
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.manager.CLISessionManager", return_value=cli_manager),
        TestClient(app),
    ):
        pass

    cleanup_provider.assert_awaited_once()


def test_app_lifespan_flush_pending_save_exception_warning_only(tmp_path):
    """Session store flush exception on shutdown is logged as warning, no crash."""
    from api.app import create_app

    app = create_app()

    settings = SimpleNamespace(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
        log_file=str(tmp_path / "server.log"),
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock()
    fake_platform.stop = AsyncMock()

    session_store = MagicMock()
    session_store.get_all_trees.return_value = []
    session_store.get_node_mapping.return_value = {}
    session_store.sync_from_tree_data = MagicMock()
    session_store.flush_pending_save = MagicMock(side_effect=OSError("disk full"))

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")
    cleanup_provider = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(api_app_mod, "cleanup_provider", new=cleanup_provider),
        patch(
            "messaging.factory.create_messaging_platform",
            return_value=fake_platform,
        ),
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.manager.CLISessionManager", return_value=cli_manager),
        TestClient(app),
    ):
        pass

    session_store.flush_pending_save.assert_called_once()
    cleanup_provider.assert_awaited_once()
