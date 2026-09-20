"""Tests for installed CLI entrypoints, commands, and launchers."""

import json
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.settings import Settings


def _launcher_settings(
    *,
    port: int = 8082,
    token: str = "freecc",
    open_admin_browser: bool = True,
) -> Settings:
    return Settings(
        host="0.0.0.0",
        port=port,
        proxy_auth_enabled=False,
        proxy_auth_token=token,
        model="nvidia_nim/test-model",
        open_admin_browser=open_admin_browser,
    )


def test_cli_scripts_are_registered() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert pyproject["project"]["scripts"] == {
        "fcc-server": "free_claude_code.cli.entrypoints:serve",
        "fcc-claude": "free_claude_code.cli.launchers.claude:launch",
        "fcc-codex": "free_claude_code.cli.launchers.codex:launch",
        "fcc-pi": "free_claude_code.cli.launchers.pi:launch",
        "fcc-opencode": "free_claude_code.cli.launchers.opencode:launch",
        "fcc-cline": "free_claude_code.cli.launchers.cline:launch",
        "fcc-hermes": "free_claude_code.cli.launchers.hermes:launch",
        "fcc-dsh": "free_claude_code.cli.launchers.dsh:launch",
        "fcc-grok": "free_claude_code.cli.launchers.grok:launch",
        "fcc-muse": "free_claude_code.cli.launchers.muse:launch",
        "fcc-aider": "free_claude_code.cli.launchers.aider:launch",
    }
    assert pyproject["project"]["gui-scripts"] == {
        "fcc-desktop": "free_claude_code.cli.desktop_entrypoint:launch",
    }


@pytest.mark.parametrize(
    "argv",
    [("--version",), ("--version", "--help"), ("--help", "--version")],
)
def test_fcc_server_reports_version_without_side_effects(
    argv: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli import entrypoints

    with patch.object(entrypoints, "package_version", return_value="9.8.7"):
        entrypoints.serve(argv)

    assert capsys.readouterr() == ("free-claude-code 9.8.7\n", "")


def test_version_entrypoint_does_not_import_command_runtime() -> None:
    script = "\n".join(
        (
            "import json",
            "import sys",
            "from free_claude_code.cli.entrypoints import serve",
            "serve(['--version'])",
            "forbidden = ('uvicorn', 'fastapi', 'openai', "
            "'free_claude_code.cli.commands', "
            "'free_claude_code.runtime.bootstrap')",
            "print(json.dumps([name for name in forbidden if name in sys.modules]))",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == []


def test_non_version_entrypoint_delegates_to_server_command() -> None:
    from free_claude_code.cli import commands, entrypoints

    with patch.object(commands, "serve") as command:
        entrypoints.serve(())

    command.assert_called_once_with()


def test_explicit_open_admin_waits_for_owned_http_ready():
    from free_claude_code.cli.commands import ServerSupervisor

    supervisor = ServerSupervisor()
    with patch.object(supervisor, "_open_admin") as open_admin:
        supervisor.request_open_admin()
        open_admin.assert_not_called()
        settings = _launcher_settings()
        supervisor._ready_settings = settings
        supervisor.request_open_admin()
        open_admin.assert_called_once_with(settings, 0)


@pytest.mark.parametrize("open_admin_browser", (False, True))
def test_serve_respects_admin_browser_setting(open_admin_browser: bool) -> None:
    from free_claude_code.cli import commands

    settings = _launcher_settings(open_admin_browser=open_admin_browser)
    get_settings = MagicMock(return_value=settings)

    with (
        patch.object(commands, "get_settings", get_settings),
        patch.object(
            commands.ServerSupervisor, "_run_once", return_value=False
        ) as run_server,
        patch.object(commands, "kill_all_best_effort"),
    ):
        commands.serve()

    run_server.assert_called_once_with(
        settings,
        open_admin_browser=open_admin_browser,
        restart_generation=0,
    )


def test_server_startup_repairs_invalid_managed_provider_proxy(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.cli import commands
    from free_claude_code.config.env_files import dotenv_values_from_file
    from free_claude_code.config.paths import managed_env_path

    monkeypatch.delenv("OPENAI_PROXY", raising=False)
    invalid_proxy = "invalid://user:leaked-secret@proxy.example:8080"
    managed = managed_env_path()
    managed.parent.mkdir(parents=True)
    managed.write_text(
        "\n".join(
            (
                "FCC_CONFIG_SCHEMA=1",
                "MODEL=nvidia_nim/test-model",
                f"OPENAI_PROXY={invalid_proxy}",
                "",
            )
        ),
        encoding="utf-8",
    )
    handed_off_settings: list[Settings] = []

    def run_once(
        _self: commands.ServerSupervisor,
        settings: Settings,
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        assert open_admin_browser is False
        assert restart_generation == 0
        handed_off_settings.append(settings)
        return False

    with (
        caplog.at_level("WARNING"),
        patch.object(commands.ServerSupervisor, "_run_once", run_once),
        patch.object(commands, "kill_all_best_effort"),
    ):
        commands.ServerSupervisor().run(open_admin_browser=False)

    assert len(handed_off_settings) == 1
    assert handed_off_settings[0].openai_proxy is None
    assert "OPENAI_PROXY" not in dotenv_values_from_file(managed)
    assert "OPENAI_PROXY" in caplog.text
    assert invalid_proxy not in caplog.text
    assert "leaked-secret" not in caplog.text


def test_load_server_settings_skips_reload_when_no_repair_occurs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from free_claude_code.cli import commands

    settings = _launcher_settings()
    with (
        caplog.at_level("WARNING"),
        patch.object(commands, "get_settings", return_value=settings) as get_settings,
        patch.object(
            commands.ManagedConfigStore,
            "repair_invalid_provider_proxies",
            return_value=(),
        ) as repair,
        patch.object(commands, "clear_settings_cache") as clear_cache,
    ):
        loaded = commands.load_server_settings()

    assert loaded is settings
    get_settings.assert_called_once_with()
    repair.assert_called_once_with()
    clear_cache.assert_not_called()
    assert "Removed invalid managed provider proxy settings" not in caplog.text


def test_load_server_settings_reloads_once_and_warns_after_repair(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from free_claude_code.cli import commands
    from free_claude_code.config.paths import managed_env_path

    invalid_proxy = "invalid://user:leaked-secret@proxy.example:8080"
    stale = Settings(openai_proxy=invalid_proxy)
    repaired = Settings()
    get_settings = MagicMock(side_effect=(stale, repaired))

    with (
        caplog.at_level("WARNING"),
        patch.object(commands, "get_settings", get_settings),
        patch.object(
            commands.ManagedConfigStore,
            "repair_invalid_provider_proxies",
            return_value=("OPENAI_PROXY", "GROQ_PROXY"),
        ),
        patch.object(commands, "clear_settings_cache") as clear_cache,
    ):
        loaded = commands.load_server_settings()

    assert loaded is repaired
    assert get_settings.call_count == 2
    clear_cache.assert_called_once_with()
    warning = next(
        record.message
        for record in caplog.records
        if "Removed invalid managed provider proxy settings" in record.message
    )
    assert str(managed_env_path()) in warning
    assert "OPENAI_PROXY, GROQ_PROXY" in warning
    assert invalid_proxy not in warning
    assert "leaked-secret" not in warning


def test_load_server_settings_leaves_process_owned_invalid_proxy_explicit(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.cli import commands
    from free_claude_code.config.paths import managed_env_path

    process_proxy = "invalid://process-proxy"
    monkeypatch.setenv("OPENAI_PROXY", process_proxy)
    managed = managed_env_path()
    managed.parent.mkdir(parents=True)
    managed.write_text(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n",
        encoding="utf-8",
    )
    baseline = managed.read_bytes()

    with caplog.at_level("WARNING"):
        settings = commands.load_server_settings()

    assert settings.openai_proxy == process_proxy
    assert managed.read_bytes() == baseline
    assert "Removed invalid managed provider proxy settings" not in caplog.text


def test_serve_supervisor_restarts_when_app_requests_restart() -> None:
    from free_claude_code.cli import commands

    settings = _launcher_settings(port=0)
    get_settings = MagicMock(side_effect=[settings, settings])
    servers: list[object] = []
    restart_callbacks: list[Callable[[], None]] = []

    apps: list[SimpleNamespace] = []

    def build_asgi_app(_settings: Settings, restart_callback: Callable[[], None]):
        restart_callbacks.append(restart_callback)
        app = SimpleNamespace(
            runtime=SimpleNamespace(
                is_closed=False,
                begin_shutdown=lambda: None,
                http_started=lambda: None,
                close=AsyncMock(return_value=True),
            )
        )
        apps.append(app)
        return app

    class FakeServer:
        def __init__(self, config, *, begin_shutdown, on_started, close_runtime):
            self.config = config
            self.on_started = on_started
            self.started = True
            self.should_exit = False
            servers.append(self)

        def run(self, sockets=None):
            self.on_started()
            if len(servers) == 1:
                restart_callbacks[-1]()
                assert self.should_exit is True
                self.config.app.runtime.is_closed = True

    def fake_config(app, **kwargs):
        return SimpleNamespace(app=app, kwargs=kwargs)

    with (
        patch.object(commands, "get_settings", get_settings),
        patch("uvicorn.Config", side_effect=fake_config),
        patch(
            "free_claude_code.cli.uvicorn_server.RuntimeServer", side_effect=FakeServer
        ),
        patch(
            "free_claude_code.runtime.bootstrap.build_asgi_app",
            side_effect=build_asgi_app,
        ),
        patch.object(commands.ServerSupervisor, "_open_admin") as open_admin,
        patch.object(commands, "clear_settings_cache") as clear_settings_cache,
        patch.object(commands, "kill_all_best_effort") as kill_all,
    ):
        commands.serve()

    assert len(servers) == 2
    open_admin.assert_called_once_with(settings, 0)
    clear_settings_cache.assert_called_once()
    kill_all.assert_called_once()


def test_serve_supervisor_refuses_restart_after_incomplete_shutdown() -> None:
    from free_claude_code.cli import commands

    settings = _launcher_settings(port=0)
    get_settings = MagicMock(return_value=settings)
    servers: list[object] = []
    restart_callbacks: list[Callable[[], None]] = []

    def build_asgi_app(_settings: Settings, restart_callback: Callable[[], None]):
        restart_callbacks.append(restart_callback)
        return SimpleNamespace(
            runtime=SimpleNamespace(
                is_closed=False,
                begin_shutdown=lambda: None,
                http_started=lambda: None,
                close=AsyncMock(return_value=True),
            )
        )

    class FakeServer:
        def __init__(self, config, *, begin_shutdown, on_started, close_runtime):
            self.config = config
            self.on_started = on_started
            self.started = True
            self.should_exit = False
            servers.append(self)

        def run(self, sockets=None):
            self.on_started()
            restart_callbacks[-1]()
            assert self.should_exit is True

    def fake_config(app, **kwargs):
        return SimpleNamespace(app=app, kwargs=kwargs)

    with (
        patch.object(commands, "get_settings", get_settings),
        patch("uvicorn.Config", side_effect=fake_config),
        patch(
            "free_claude_code.cli.uvicorn_server.RuntimeServer", side_effect=FakeServer
        ),
        patch(
            "free_claude_code.runtime.bootstrap.build_asgi_app",
            side_effect=build_asgi_app,
        ),
        patch.object(commands.ServerSupervisor, "_open_admin"),
        patch.object(commands, "clear_settings_cache") as clear_settings_cache,
        patch.object(commands, "kill_all_best_effort") as kill_all,
    ):
        commands.serve()

    assert len(servers) == 1
    clear_settings_cache.assert_not_called()
    kill_all.assert_called_once()


def test_serve_handles_keyboard_interrupt_without_traceback() -> None:
    from free_claude_code.cli import commands

    settings = _launcher_settings()
    get_settings = MagicMock(return_value=settings)

    with (
        patch.object(commands, "get_settings", get_settings),
        patch.object(
            commands.ServerSupervisor,
            "_run_once",
            side_effect=KeyboardInterrupt,
        ),
        patch.object(commands, "clear_settings_cache") as clear_settings_cache,
        patch.object(commands, "kill_all_best_effort") as kill_all,
    ):
        commands.serve()

    clear_settings_cache.assert_not_called()
    kill_all.assert_not_called()


def test_claude_child_env_targets_current_proxy_config() -> None:
    from free_claude_code.harnesses.claude import build_claude_proxy_env

    env = build_claude_proxy_env(
        proxy_root_url="http://127.0.0.1:9090",
        auth_token="proxy-token",
        base_env={
            "PATH": "keep",
            "ANTHROPIC_API_URL": "https://api.anthropic.com/v1",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            "ANTHROPIC_AUTH_TOKEN": "old-token",
            "ANTHROPIC_API_KEY": "official-key",
            "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "0",
            "DISABLE_FEEDBACK_COMMAND": "0",
            "DISABLE_ERROR_REPORTING": "0",
            "DISABLE_TELEMETRY": "0",
        },
    )

    assert env["PATH"] == "keep"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9090"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert env["DISABLE_AUTOUPDATER"] == "1"
    assert env["DISABLE_FEEDBACK_COMMAND"] == "1"
    assert env["DISABLE_ERROR_REPORTING"] == "1"
    assert env["DISABLE_TELEMETRY"] == "0"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]
    assert "ANTHROPIC_API_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in env
