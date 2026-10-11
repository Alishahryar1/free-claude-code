import json
from pathlib import Path

import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import headroom_desktops as desktops
from free_claude_code.harnesses.headroom_clients import NativeClients
from free_claude_code.harnesses.headroom_mcp import configure_registrations


@pytest.fixture
def desktop_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for module in (
        desktops.claude,
        desktops.codex,
        desktops.claude_desktop,
        desktops.jetbrains,
    ):
        monkeypatch.setattr(
            module, "configure", lambda *args, **kwargs: {"connected": False}
        )
    monkeypatch.setattr(desktops.vscode, "status", lambda *args: {"connected": False})
    monkeypatch.setattr(desktops.dsh, "has_provider", lambda *args: False)
    return NativeClients(tmp_path / "headroom", {}, tmp_path), Settings()


def test_disconnected_desktops_are_not_registered(desktop_setup, tmp_path):
    clients, settings = desktop_setup
    assert desktops.prepare_desktops(clients, settings) == []
    assert not list(tmp_path.iterdir())


def test_claude_3p_uses_user_mcp_without_changing_profile_policy(
    desktop_setup, tmp_path, monkeypatch
):
    clients, settings = desktop_setup
    root = tmp_path / "Claude-3p"
    root.mkdir()
    profile = root / "fcc.json"
    profile.write_text('{"isLocalDevMcpEnabled":true,"gateway":"keep"}')
    mode = root / "claude_desktop_config.json"
    mode.write_text(
        '{"deploymentMode":"3p","mcpServers":{"other":{"command":"other"}}}'
    )
    monkeypatch.setattr(
        desktops.claude_desktop,
        "configure",
        lambda *args, **kwargs: {
            "connected": True,
            "paths": {"desktop_profile": str(profile), "desktop_settings": str(mode)},
        },
    )

    targets = desktops.prepare_desktops(clients, settings)
    configure_registrations(targets)

    document = json.loads(mode.read_text())
    assert document["mcpServers"]["headroom"]["command"] == str(clients.headroom)
    assert document["mcpServers"]["other"] == {"command": "other"}
    assert "managedMcpServers" not in profile.read_text() + mode.read_text()
    assert json.loads(profile.read_text())["gateway"] == "keep"


def test_claude_3p_disabled_stdio_is_preserved(desktop_setup, tmp_path, monkeypatch):
    clients, settings = desktop_setup
    profile = tmp_path / "profile.json"
    profile.write_text('{"isLocalDevMcpEnabled":false}')
    mode = tmp_path / "desktop.json"
    mode.write_text("{}")
    monkeypatch.setattr(
        desktops.claude_desktop,
        "configure",
        lambda *args, **kwargs: {
            "connected": True,
            "paths": {"desktop_profile": str(profile), "desktop_settings": str(mode)},
        },
    )
    results = configure_registrations(desktops.prepare_desktops(clients, settings))
    assert "skipped" in results[0]
    assert mode.read_text() == "{}"


def test_pending_desktop_disconnect_is_not_registered(desktop_setup, monkeypatch):
    clients, settings = desktop_setup
    monkeypatch.setattr(
        desktops.claude_desktop,
        "configure",
        lambda *args, **kwargs: {"connected": True, "disconnect_pending": True},
    )
    assert desktops.prepare_desktops(clients, settings) == []
