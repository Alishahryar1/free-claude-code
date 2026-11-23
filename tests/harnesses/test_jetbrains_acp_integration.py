import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.harnesses import jetbrains_acp_integration as jb
from free_claude_code.harnesses.jetbrains_acp_integration import (
    registry_path,
    system_root,
)
from tests.wire_payload import array_payload, decode_object, object_payload


def write(path: Path, value: JsonObject):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def install(
    registry: Path, systems: Path, version: str = "0.79.0", ide: str = "IDE 2026.2"
):
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
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, registry, systems = (
        tmp_path / "acp.json",
        tmp_path / "installed.json",
        tmp_path / "systems",
    )
    monkeypatch.setattr(jb, "registry_path", lambda: registry)
    monkeypatch.setattr(jb, "system_root", lambda: systems)

    def replace_which(_: object):
        return None

    monkeypatch.setattr(jb.shutil, "which", replace_which)

    def run_probe(
        args: Sequence[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "v24.19.0\n", "")

    monkeypatch.setattr(jb.subprocess, "run", run_probe)
    return config, registry, systems


def test_lifecycle_preserves_other_agents_and_refreshes_owned_settings(
    files: tuple[Path, Path, Path],
) -> None:
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
    saved = decode_object(config.read_text())
    entry = object_payload(saved["agent_servers"])["Claude Code (FCC)"]
    assert object_payload(entry)["command"] == str(node)
    assert object_payload(entry)["args"] == [str(script), "--hide-claude-auth"]
    assert (
        object_payload(object_payload(entry)["env"])["ANTHROPIC_AUTH_TOKEN"] == "token"
    )
    assert (
        object_payload(object_payload(entry)["env"])["CLAUDE_CODE_AUTO_MODE_SERVER"]
        == "0"
    )
    object_payload(object_payload(entry)["env"])["KEEP"] = "yes"
    object_payload(object_payload(entry)["env"])["CLAUDE_CODE_MAX_RETRIES"] = "10"
    object_payload(object_payload(entry)["env"])["API_TIMEOUT_MS"] = "1000"
    object_payload(entry)["use_idea_mcp"] = False
    write(config, saved)
    _, updated = install(registry, systems, version="0.80.0")
    assert jb.refresh_connected(config, "http://localhost:9090", "rotated")
    refreshed = object_payload(decode_object(config.read_text())["agent_servers"])[
        "Claude Code (FCC)"
    ]
    assert array_payload(object_payload(refreshed)["args"])[0] == str(updated)
    assert (
        object_payload(object_payload(refreshed)["env"])["ANTHROPIC_AUTH_TOKEN"]
        == "rotated"
    )
    assert object_payload(object_payload(refreshed)["env"])["KEEP"] == "yes"
    assert (
        object_payload(object_payload(refreshed)["env"])["CLAUDE_CODE_MAX_RETRIES"]
        == "10"
    )
    assert (
        object_payload(object_payload(refreshed)["env"])["API_TIMEOUT_MS"]
        == "2147483647"
    )
    assert (
        object_payload(object_payload(refreshed)["env"])[
            "CLAUDE_ENABLE_STREAM_WATCHDOG"
        ]
        == "0"
    )
    assert (
        object_payload(object_payload(refreshed)["env"])["CLAUDE_ENABLE_BYTE_WATCHDOG"]
        == "0"
    )
    assert (
        object_payload(object_payload(refreshed)["env"])["API_FORCE_IDLE_TIMEOUT"]
        == "0"
    )
    assert object_payload(refreshed)["use_idea_mcp"] is False
    before = config.stat().st_mtime_ns
    assert not jb.refresh_connected(config, "http://localhost:9090", "rotated")
    assert config.stat().st_mtime_ns == before
    registry.unlink()
    node.unlink()
    assert not jb.configure(config, "http://localhost:9090", "rotated", False)[
        "connected"
    ]
    assert load_json_value(config.read_text()) == {
        "agent_servers": {"Other": other},
        "default_mcp_settings": {"use_idea_mcp": True},
    }
    assert not jb.configure(config, "http://localhost:9090", "rotated", False)[
        "connected"
    ]


def test_missing_installation_never_creates_config(
    files: tuple[Path, Path, Path],
) -> None:
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
def test_invalid_or_unowned_configuration_is_never_overwritten(
    files: tuple[Path, Path, Path], source: str
) -> None:
    config, _, _ = files
    config.write_text(source)
    for action in [None, True, False]:
        with pytest.raises(ValueError):
            jb.configure(config, "http://localhost:8082", "token", action)
        assert config.read_text() == source


def test_status_and_disconnect_do_not_discover_or_run_programs(
    files: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, registry, systems = files
    install(registry, systems)
    jb.configure(config, "http://localhost:8082", "token", True)
    monkeypatch.setattr(jb, "registry_path", lambda: pytest.fail("registry access"))

    def replace_run(*a: object, **k: object):
        return pytest.fail("subprocess")

    monkeypatch.setattr(jb.subprocess, "run", replace_run)
    assert jb.configure(config, "http://localhost:8082", "token")["connected"]
    assert not jb.configure(config, "http://localhost:8082", "token", False)[
        "connected"
    ]


@pytest.mark.parametrize("fault", ["marker", "version", "escape", "engine", "node"])
def test_unusable_installation_preserves_connected_entry(
    files: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    config, registry, systems = files
    _, script = install(registry, systems)
    jb.configure(config, "http://localhost:8082", "token", True)
    before = config.read_bytes()
    package = script.parent.parent / "package.json"
    metadata = decode_object(package.read_text())
    if fault == "marker":
        next(systems.rglob(".installed")).unlink()
    elif fault == "version":
        metadata["version"] = "0.1.0"
    elif fault == "escape":
        object_payload(metadata["bin"])["claude-agent-acp"] = str(config)
    elif fault == "engine":
        object_payload(metadata["engines"])["node"] = "^22"
    else:

        def run_probe(
            args: Sequence[str], **options: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args, 0, "v20.0.0", "")

        monkeypatch.setattr(jb.subprocess, "run", run_probe)
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
    files: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    data: str,
    cache: str,
) -> None:
    monkeypatch.setattr(jb, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(jb.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("APPDATA", "LOCALAPPDATA", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    assert registry_path() == tmp_path / data / "JetBrains/acp-agents/installed.json"
    assert system_root() == tmp_path / cache / "JetBrains"
    config, registry, systems = files
    node, script = install(registry, systems)
    jb.configure(config, "http://localhost:8082", "token", True)
    entry = object_payload(decode_object(config.read_text())["agent_servers"])[
        "Claude Code (FCC)"
    ]
    assert object_payload(entry)["command"] == str(node)
    assert array_payload(object_payload(entry)["args"])[0] == str(script)


def test_previous_ide_is_preferred_and_system_node_is_fallback(
    files: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, registry, systems = files
    node, script = install(registry, systems, ide="Z IDE")
    jb.configure(config, "http://localhost:8082", "token", True)
    install(registry, systems, ide="A IDE")
    assert not jb.refresh_connected(config, "http://localhost:8082", "token")
    system = config.parent / "system node.exe"
    system.touch()
    node.unlink()

    def replace_which(_: object):
        return str(system)

    monkeypatch.setattr(jb.shutil, "which", replace_which)
    jb.refresh_connected(config, "http://localhost:8082", "token")
    entry = object_payload(decode_object(config.read_text())["agent_servers"])[
        "Claude Code (FCC)"
    ]
    assert object_payload(entry)["command"] == str(system.resolve())
    assert array_payload(object_payload(entry)["args"])[0] == str(script)


def test_timed_out_node_leaves_config_unchanged(
    files: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, registry, systems = files
    install(registry, systems)
    write(config, {"agent_servers": {"Other": {"command": "other"}}})
    before = config.read_bytes()

    def timeout(args: list[str], **kwargs: object):
        assert kwargs["timeout"] == 3
        raise subprocess.TimeoutExpired(args, 3)

    monkeypatch.setattr(jb.subprocess, "run", timeout)
    with pytest.raises(jb.SetupError):
        jb.configure(config, "http://localhost:8082", "token", True)
    assert config.read_bytes() == before


def test_failed_atomic_replace_preserves_existing_config(
    files: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from free_claude_code.harnesses import config_file

    config, registry, systems = files
    install(registry, systems)
    write(config, {"agent_servers": {"Other": {"command": "other"}}})
    before = config.read_bytes()

    def fail(*args: object):
        raise PermissionError("busy")

    monkeypatch.setattr(config_file.os, "replace", fail)
    with pytest.raises(PermissionError):
        jb.configure(config, "http://localhost:8082", "token", True)
    assert config.read_bytes() == before
