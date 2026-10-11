"""Headroom scopes for desktop integrations already connected to FCC."""

import shutil
from pathlib import Path

from free_claude_code.config.paths import (
    claude_desktop_disconnect_path,
    codex_model_catalog_path,
)
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import (
    claude_desktop_integration as claude_desktop,
)
from free_claude_code.harnesses import (
    claude_integration as claude,
)
from free_claude_code.harnesses import (
    codex_integration as codex,
)
from free_claude_code.harnesses import (
    dsh_desktop_integration as dsh,
)
from free_claude_code.harnesses import (
    jetbrains_acp_integration as jetbrains,
)
from free_claude_code.harnesses import (
    vscode_chat_integration as vscode,
)
from free_claude_code.harnesses.headroom_clients import NativeClients
from free_claude_code.harnesses.headroom_dsh import dsh_registration
from free_claude_code.harnesses.headroom_mcp import (
    McpSetupError,
    Registration,
    json_registration,
    read_object,
)


def _path(state: JsonObject, name: str) -> Path:
    paths = state.get("paths")
    value = paths.get(name) if isinstance(paths, dict) else None
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise McpSetupError(f"Desktop integration did not expose its {name} path.")
    return Path(value)


def prepare_desktops(clients: NativeClients, settings: Settings) -> list[Registration]:
    """Read saved ownership through passive status branches, never connect/refresh."""
    targets = []
    url = local_proxy_root_url(settings)
    token = settings.proxy_auth_token
    native = []
    if claude.configure(
        claude.settings_path(), claude.claude_state_path(), url, token
    ).get("connected"):
        native.append("claude")
    if codex.configure(codex.config_path(), codex_model_catalog_path(), url).get(
        "connected"
    ):
        native.append("codex")
    targets.extend(
        target for agent in native if (target := clients.prepare(agent)) is not None
    )

    try:
        state = claude_desktop.configure(
            claude_desktop.config_root(),
            url,
            token,
            disconnect_path=claude_desktop_disconnect_path(),
        )
    except claude_desktop.ManagedDesktopError:
        clients.outcomes.append("Claude Desktop: skipped (managed application policy)")
    else:
        if state.get("connected") and not state.get("disconnect_pending"):
            path = _path(state, "desktop_settings")
            mode = read_object(path)
            profile = read_object(_path(state, "desktop_profile"))
            features = mode.get("features", {})
            if not isinstance(features, dict):
                raise McpSetupError(
                    "Claude Desktop features configuration must be an object."
                )
            blocked = None
            if (
                profile.get("isLocalDevMcpEnabled") is False
                or features.get("isLocalDevMcpEnabled") is False
            ):
                blocked = "local stdio MCP is disabled"
            elif "managedMcpServers" in profile:
                blocked = "managed MCP servers control this desktop"
            targets.append(
                json_registration(
                    "Claude Desktop",
                    path,
                    ("mcpServers",),
                    clients.headroom,
                    blocked=blocked,
                )
            )

    model_path = vscode.config_path()
    if vscode.status(model_path).get("connected"):
        targets.append(
            json_registration(
                "VS Code Chat",
                model_path.with_name("mcp.json"),
                ("servers",),
                clients.headroom,
                expected={
                    "type": "stdio",
                    "command": str(clients.headroom),
                    "args": ["mcp", "serve"],
                },
            )
        )

    if jetbrains.configure(jetbrains.config_path(), url, token).get("connected"):
        # ACP's per-session IDE options can override the adapter's user settings.
        # Saved connection state cannot establish those future session options.
        clients.outcomes.append(
            "JetBrains Claude ACP: skipped (MCP inheritance depends on IDE session settings; use the Claude user registration)"
        )

    if dsh.has_provider(dsh.config_home()):
        binary = shutil.which("dsh", path=clients.env.get("PATH"))
        if binary:
            if target := dsh_registration(
                clients.headroom,
                dsh.config_home(),
                lambda: clients.run([binary, "--profile", "desktop", "--dump-config"]),
                profile="desktop",
            ):
                targets.append(target)
            else:
                clients.outcomes.append(
                    "DSH Desktop: skipped (native desktop profile does not exist)"
                )
        else:
            clients.outcomes.append(
                "DSH Desktop: skipped (native dsh command is unavailable for verification)"
            )
    return targets
