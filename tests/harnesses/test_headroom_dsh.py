import json
from pathlib import Path

import pytest

from free_claude_code.harnesses.headroom_clients import NativeClients
from free_claude_code.harnesses.headroom_dsh import dsh_registration
from free_claude_code.harnesses.headroom_mcp import (
    McpSetupError,
    configure_registrations,
)


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "dsh"
    (root / "profiles/web").mkdir(parents=True)
    (root / "profiles/web/package.json").write_text("{}")
    owner = NativeClients(tmp_path / "headroom", {"DSH_HOME": str(root)}, tmp_path)
    monkeypatch.setattr(owner, "run", lambda args, **kwargs: json.dumps([]))
    return root, owner


def test_dsh_home_patch_preserves_comments_and_other_plugins(tmp_path, monkeypatch):
    root, owner = setup(tmp_path, monkeypatch)
    path = root / "cordis.patch.yml"
    path.write_text("# native settings\n- id: existing\n  config:\n    custom: true\n")
    target = dsh_registration(
        owner.headroom,
        Path(owner.env["DSH_HOME"]),
        lambda: owner.run(["dsh", "--profile", "web", "--dump-config"]),
    )
    assert target is not None
    configure_registrations([target])
    first = path.read_text()
    assert "# native settings" in first and "custom: true" in first
    assert "@deepseek-ai/dsh-mcp-client" in first
    assert "serverName: headroom" in first
    configure_registrations([target])
    assert path.read_text() == first


def test_dsh_profile_collision_prevents_home_patch(tmp_path, monkeypatch):
    root, owner = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        owner,
        "run",
        lambda args, **kwargs: json.dumps(
            [
                {
                    "id": "custom",
                    "name": "@deepseek-ai/dsh-mcp-client",
                    "config": {
                        "serverName": "headroom",
                        "transport": "stdio",
                        "command": "custom",
                        "args": [],
                    },
                }
            ]
        ),
    )
    target = dsh_registration(
        owner.headroom,
        Path(owner.env["DSH_HOME"]),
        lambda: owner.run(["dsh", "--profile", "web", "--dump-config"]),
    )
    assert target is not None
    with pytest.raises(McpSetupError, match="conflicting headroom"):
        configure_registrations([target])
    assert not (root / "cordis.patch.yml").exists()


def test_dsh_does_not_create_a_missing_profile(tmp_path, monkeypatch):
    owner = NativeClients(
        tmp_path / "headroom", {"DSH_HOME": str(tmp_path / "absent")}, tmp_path
    )
    assert (
        dsh_registration(
            owner.headroom,
            Path(owner.env["DSH_HOME"]),
            lambda: owner.run(["dsh", "--profile", "web", "--dump-config"]),
        )
        is None
    )
    assert not (tmp_path / "absent").exists()


def test_dsh_reserved_id_cannot_belong_to_another_server(tmp_path, monkeypatch):
    root, owner = setup(tmp_path, monkeypatch)
    path = root / "cordis.patch.yml"
    path.write_text(
        "- insert:\n  - id: fcc-headroom-mcp\n    name: '@deepseek-ai/dsh-mcp-client'\n    config:\n      serverName: other\n"
    )
    target = dsh_registration(
        owner.headroom,
        Path(owner.env["DSH_HOME"]),
        lambda: owner.run(["dsh", "--profile", "web", "--dump-config"]),
    )
    assert target is not None
    with pytest.raises(McpSetupError, match="already uses"):
        configure_registrations([target])


def test_dsh_invalid_composed_yaml_is_sanitized(tmp_path, monkeypatch):
    _, owner = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "run", lambda args: "secret: [invalid")
    target = dsh_registration(
        owner.headroom,
        Path(owner.env["DSH_HOME"]),
        lambda: owner.run(["dsh", "--profile", "web", "--dump-config"]),
    )
    assert target is not None
    with pytest.raises(McpSetupError, match="invalid composed configuration"):
        target.state()
