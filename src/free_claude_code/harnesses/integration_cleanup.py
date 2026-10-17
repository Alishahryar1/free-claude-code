"""Disconnect native integrations before removing FCC's installed commands."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.paths import (
    claude_desktop_disconnect_path,
    codex_model_catalog_path,
    managed_env_path,
)
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import (
    claude_desktop_integration,
    claude_integration,
    codex_integration,
    dsh_desktop_integration,
    jetbrains_acp_integration,
    vscode_chat_integration,
)


class IntegrationCleanupError(RuntimeError):
    """One local connection cannot safely be removed before uninstalling."""


@dataclass(frozen=True)
class IntegrationPaths:
    env: Path
    claude_vscode: Path
    vscode_chat: Path
    codex: Path
    codex_catalog: Path
    jetbrains: Path
    desktop: Path
    desktop_disconnect: Path
    dsh: Path

    @classmethod
    def current(cls) -> IntegrationPaths:
        return cls(
            managed_env_path(),
            claude_integration.settings_path(),
            vscode_chat_integration.config_path(),
            codex_integration.config_path(),
            codex_model_catalog_path(),
            jetbrains_acp_integration.config_path(),
            claude_desktop_integration.config_root(),
            claude_desktop_disconnect_path(),
            dsh_desktop_integration.config_home(),
        )


def _path_exists(path: Path) -> bool:
    """Treat missing paths as absent, but let access failures stop uninstall."""
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _connection_settings(path: Path) -> Settings | None:
    if not _path_exists(path):
        return None
    # Cleanup must also work when an unrelated provider or model setting is invalid.
    keys = {"HOST", "PORT", "ANTHROPIC_AUTH_TOKEN"}
    managed = {
        key: value
        for key, value in dotenv_values_from_file(path).items()
        if key in keys
    }
    process = {key: value for key, value in os.environ.items() if key in keys}
    return compose_settings_snapshot(managed, process).settings


def disconnect_integrations(paths: IntegrationPaths | None = None) -> None:
    """Reuse each connection owner without starting a server or contacting providers.

    A failed operation leaves the installed tool and its recovery data available.
    Earlier operations are idempotent, so uninstall can safely retry.
    """
    paths = paths or IntegrationPaths.current()
    try:
        settings = _connection_settings(paths.env)
    except OSError, ValueError:
        raise IntegrationCleanupError(
            "Could not read FCC's connection settings. Correct the managed config "
            "or disconnect the integrations manually before uninstalling."
        ) from None
    url = local_proxy_root_url(settings) if settings is not None else None
    token = settings.proxy_auth_token if settings is not None else None
    operations: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "Claude Code in VS Code",
            lambda: claude_integration.disconnect_existing(
                paths.claude_vscode, url, token
            ),
        ),
        ("VS Code chat", lambda: vscode_chat_integration.disconnect(paths.vscode_chat)),
        (
            "Codex",
            lambda: codex_integration.configure(
                paths.codex, paths.codex_catalog, url or "", False
            ),
        ),
        (
            "JetBrains ACP",
            lambda: jetbrains_acp_integration.configure(
                paths.jetbrains, url or "", token or "", False
            ),
        ),
        (
            "Claude Desktop",
            lambda: (
                claude_desktop_integration.configure(
                    paths.desktop,
                    url or "",
                    token or "",
                    False,
                    disconnect_path=paths.desktop_disconnect,
                )
                if _path_exists(paths.desktop) or _path_exists(paths.desktop_disconnect)
                else None
            ),
        ),
        (
            "DSH Desktop",
            lambda: (
                dsh_desktop_integration.disconnect(paths.dsh)
                if _path_exists(paths.dsh)
                else None
            ),
        ),
    )
    for name, operation in operations:
        try:
            operation()
        except OSError, ValueError:
            raise IntegrationCleanupError(
                f"Could not disconnect {name}. Check its local settings and permissions, "
                "close the app, then retry. If its connection details changed, disconnect "
                "it manually first. FCC and its recovery data were not removed."
            ) from None
