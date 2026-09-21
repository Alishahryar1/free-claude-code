import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from free_claude_code.harnesses import jetbrains_acp_integration as jb
from free_claude_code.harnesses.jetbrains_acp_integration import (
    registry_path,
    system_root,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def install(registry, systems, version="0.79.0", ide="IDE 2026.2"):
    write(
        registry,
        {
            "agents": {
                "acp.registry.claude-acp": {
                    "agent": {
                        "version": version,
                        "distribution": {
                            "npx": {
                                "package": f"@agentclientprotocol/claude-agent-acp@{version}"
                            }
                        },
                    }
                }
            }
        },
    )
    root = systems / ide / "acp-agents"
    package = (
        root
        / "claude-acp"
        / version
        / "node_modules/@agentclientprotocol/claude-agent-acp"
    )
    write(
        package / "package.json",
        {
            "name": "@agentclientprotocol/claude-agent-acp",
            "version": version,
            "bin": {"claude-agent-acp": "dist/index.js"},
            "engines": {"node": ">=22"},
        },
    )
    script = package / "dist/index.js"
    script.parent.mkdir()
    script.write_text("adapter", encoding="utf-8")
    (root / "claude-acp" / version / ".installed").touch()
    node = (
        root
        / ".runtimes/node/24.19.0"
        / ("node.exe" if jb.sys.platform == "win32" else "bin/node")
    )
    node.parent.mkdir(parents=True, exist_ok=True)
    node.touch()
    return node.resolve(), script.resolve()


@pytest.fixture
def files(tmp_path, monkeypatch):
    config, registry, systems = (
        tmp_path / "acp.json",
        tmp_path / "installed.json",
        tmp_path / "systems",
    )
    monkeypatch.setattr(jb, "registry_path", lambda: registry)
    monkeypatch.setattr(jb, "system_root", lambda: systems)
    monkeypatch.setattr(jb.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        jb.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 0, "v24.19.0\n", ""),
    )
    return config, registry, systems


def test_lifecycle_preserves_other_agents_and_refreshes_owned_settings(files):
    config, registry, systems = files
    node, script = install(registry, systems)
    original = registry.read_bytes()
    other = {"command": "other", "env": {"SECRET": "private"}}
    write(
        config,
        {
            "agent_servers": {"Other": other},
            "default_mcp_settings": {"use_idea_mcp": True},
        },
    )
    assert not jb.configure(config, "http://localhost:8082", "token")["connected"]
    assert jb.configure(config, "http://localhost:8082", "token", True)["connected"]
    assert registry.read_bytes() == original
    saved = json.loads(config.read_text())
    entry = saved["agent_servers"]["Claude Code (FCC)"]
    assert entry["command"] == str(node)
    assert entry["args"] == [str(script), "--hide-claude-auth"]
    assert entry["env"]["ANTHROPIC_AUTH_TOKEN"] == "token"
    assert entry["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    entry["env"]["KEEP"] = "yes"
    entry["use_idea_mcp"] = False
    write(config, saved)
    _, updated = install(registry, systems, version="0.80.0")
    assert jb.refresh_connected(config, "http://localhost:9090", "rotated")
    refreshed = json.loads(config.read_text())["agent_servers"]["Claude Code (FCC)"]
    assert refreshed["args"][0] == str(updated)
    assert refreshed["env"]["ANTHROPIC_AUTH_TOKEN"] == "rotated"
    assert refreshed["env"]["KEEP"] == "yes"
    assert refreshed["use_idea_mcp"] is False
    before = config.stat().st_mtime_ns
    assert not jb.refresh_connected(config, "http://localhost:9090", "rotated")
    assert config.stat().st_mtime_ns == before
    registry.unlink()
    node.unlink()
    assert not jb.configure(config, "http://localhost:9090", "rotated", False)[
        "connected"
    ]
    assert json.loads(config.read_text()) == {
        "agent_servers": {"Other": other},
        "default_mcp_settings": {"use_idea_mcp": True},
    }
    assert not jb.configure(config, "http://localhost:9090", "rotated", False)[
        "connected"
    ]


def test_missing_installation_never_creates_config(files):
    config, _, _ = files
    assert not jb.refresh_connected(config, "http://localhost:8082", "token")
    with pytest.raises(jb.SetupError):
        jb.configure(config, "http://localhost:8082", "token", True)
    assert not config.exists()


@pytest.mark.parametrize(
    "source",
    [
        "{",
        "[]",
        '{"agent_servers":[]}',
        '{"agent_servers":{"Claude Code (FCC)":{"command":"other"}}}',
        '{"agent_servers":{},"agent_servers":{}}',
    ],
)
def test_invalid_or_unowned_configuration_is_never_overwritten(files, source):
    config, _, _ = files
    config.write_text(source)
    for action in [None, True, False]:
        with pytest.raises(ValueError):
            jb.configure(config, "http://localhost:8082", "token", action)
        assert config.read_text() == source


def test_status_and_disconnect_do_not_discover_or_run_programs(files, monkeypatch):
    config, registry, systems = files
    install(registry, systems)
    jb.configure(config, "http://localhost:8082", "token", True)
    monkeypatch.setattr(jb, "registry_path", lambda: pytest.fail("registry access"))
    monkeypatch.setattr(jb.subprocess, "run", lambda *a, **k: pytest.fail("subprocess"))
    assert jb.configure(config, "http://localhost:8082", "token")["connected"]
    assert not jb.configure(config, "http://localhost:8082", "token", False)[
        "connected"
    ]


@pytest.mark.parametrize("fault", ["marker", "version", "escape", "engine", "node"])
def test_unusable_installation_preserves_connected_entry(files, monkeypatch, fault):
    config, registry, systems = files
    _, script = install(registry, systems)
    jb.configure(config, "http://localhost:8082", "token", True)
    before = config.read_bytes()
    package = script.parent.parent / "package.json"
    metadata = json.loads(package.read_text())
    if fault == "marker":
        next(systems.rglob(".installed")).unlink()
    elif fault == "version":
        metadata["version"] = "0.1.0"
    elif fault == "escape":
        metadata["bin"]["claude-agent-acp"] = str(config)
    elif fault == "engine":
        metadata["engines"]["node"] = "^22"
    else:
        monkeypatch.setattr(
            jb.subprocess,
            "run",
            lambda args, **kw: subprocess.CompletedProcess(args, 0, "v20.0.0", ""),
        )
    write(package, metadata)
    with pytest.raises(jb.SetupError):
        jb.refresh_connected(config, "http://localhost:8082", "new")
    assert config.read_bytes() == before
    assert jb.configure(config, "http://localhost:8082", "new")["connected"]


@pytest.mark.parametrize(
    "platform,data,cache",
    [
        ("win32", "AppData/Roaming", "AppData/Local"),
        ("darwin", "Library/Application Support", "Library/Caches"),
        ("linux", ".local/share", ".cache"),
    ],
)
def test_platform_layout_and_launch(
    files, monkeypatch, tmp_path, platform, data, cache
):
    monkeypatch.setattr(jb, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("APPDATA", "LOCALAPPDATA", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    assert registry_path() == tmp_path / data / "JetBrains/acp-agents/installed.json"
    assert system_root() == tmp_path / cache / "JetBrains"
    config, registry, systems = files
    node, script = install(registry, systems)
    jb.configure(config, "http://localhost:8082", "token", True)
    entry = json.loads(config.read_text())["agent_servers"]["Claude Code (FCC)"]
    assert entry["command"] == str(node)
    assert entry["args"][0] == str(script)


def test_previous_ide_is_preferred_and_system_node_is_fallback(files, monkeypatch):
    config, registry, systems = files
    node, script = install(registry, systems, ide="Z IDE")
    jb.configure(config, "http://localhost:8082", "token", True)
    install(registry, systems, ide="A IDE")
    assert not jb.refresh_connected(config, "http://localhost:8082", "token")
    system = config.parent / "system node.exe"
    system.touch()
    node.unlink()
    monkeypatch.setattr(jb.shutil, "which", lambda _: str(system))
    jb.refresh_connected(config, "http://localhost:8082", "token")
    entry = json.loads(config.read_text())["agent_servers"]["Claude Code (FCC)"]
    assert entry["command"] == str(system.resolve())
    assert entry["args"][0] == str(script)


def test_timed_out_node_leaves_config_unchanged(files, monkeypatch):
    config, registry, systems = files
    install(registry, systems)
    write(config, {"agent_servers": {"Other": {"command": "other"}}})
    before = config.read_bytes()

    def timeout(args, **kwargs):
        assert kwargs["timeout"] == 3
        raise subprocess.TimeoutExpired(args, 3)

    monkeypatch.setattr(jb.subprocess, "run", timeout)
    with pytest.raises(jb.SetupError):
        jb.configure(config, "http://localhost:8082", "token", True)
    assert config.read_bytes() == before


def test_failed_atomic_replace_preserves_existing_config(files, monkeypatch):
    from free_claude_code.harnesses import config_file

    config, registry, systems = files
    install(registry, systems)
    write(config, {"agent_servers": {"Other": {"command": "other"}}})
    before = config.read_bytes()

    def fail(*args):
        raise PermissionError("busy")

    monkeypatch.setattr(config_file.os, "replace", fail)
    with pytest.raises(PermissionError):
        jb.configure(config, "http://localhost:8082", "token", True)
    assert config.read_bytes() == before
