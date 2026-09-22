"""Configure a custom FCC agent using JetBrains' installed Claude ACP adapter."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import json5

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.claude import claude_proxy_values
from free_claude_code.harnesses.config_file import atomic_write_text

_NAME = "Claude Code (FCC)"
_MARKER = "FCC_JETBRAINS_ACP"
_PACKAGE = "@agentclientprotocol/claude-agent-acp"
_MISSING = (
    "Could not find a complete Claude Agent installation and compatible Node runtime. "
    "Install Claude Agent in JetBrains and start it once, then retry. "
    "FCC supports standard local JetBrains installation locations."
)


class SetupError(ValueError):
    """Actionable configuration or prerequisite problem, without secret details."""


def config_path() -> Path:
    return Path.home() / ".jetbrains/acp.json"


def _env_root(name: str, fallback: Path) -> Path:
    value = Path(os.environ.get(name) or fallback)
    return value if value.is_absolute() else fallback


def registry_path() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        parent = _env_root("APPDATA", home / "AppData/Roaming")
    elif sys.platform == "darwin":
        parent = home / "Library/Application Support"
    else:
        parent = _env_root("XDG_DATA_HOME", home / ".local/share")
    return parent / "JetBrains/acp-agents/installed.json"


def system_root() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        parent = _env_root("LOCALAPPDATA", home / "AppData/Local")
    elif sys.platform == "darwin":
        parent = home / "Library/Caches"
    else:
        parent = _env_root("XDG_CACHE_HOME", home / ".cache")
    return parent / "JetBrains"


def _object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return cast(JsonObject, value)


def _read(path: Path) -> JsonObject:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    value = _object(json5.loads(source, allow_duplicate_keys=False))
    json.dumps(value, allow_nan=False)
    return value


def _entry(document: JsonObject) -> tuple[JsonObject, JsonObject | None]:
    agents = _object(document.get("agent_servers", {}))
    if _NAME not in agents:
        return agents, None
    entry = _object(agents[_NAME])
    env = _object(entry.get("env", {}))
    if any(not isinstance(value, str) for value in env.values()):
        raise ValueError("Environment values must be strings")
    if env.get(_MARKER) != "1":
        raise SetupError(
            "An unrelated agent already uses the name Claude Code (FCC). "
            "Rename that entry in acp.json before connecting."
        )
    return agents, entry


def _version(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def _node(
    root: Path, minimum: tuple[int, int, int], previous: JsonObject
) -> Path | None:
    executable = "node.exe" if sys.platform == "win32" else "bin/node"
    runtimes = root / ".runtimes/node"
    versions = sorted(
        (
            (version, directory)
            for directory in runtimes.glob("*")
            if (version := _version(directory.name)) is not None
        ),
        reverse=True,
    )
    candidates = [(directory / executable).resolve() for _, directory in versions]
    if system_node := shutil.which("node"):
        candidates.append(Path(system_node).resolve())
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda path: str(path) != previous.get("command"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            result = subprocess.run(
                [str(path), "--version"],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32"
                else 0,
            )
        except OSError, subprocess.TimeoutExpired:
            continue
        actual = _version(result.stdout.strip())
        if result.returncode == 0 and actual is not None and actual >= minimum:
            return path
    return None


def _launch(previous: JsonObject) -> tuple[str, list[str]]:
    registry = _read(registry_path())
    try:
        installed = _object(
            _object(_object(registry["agents"])["acp.registry.claude-acp"])["agent"]
        )
        version = installed["version"]
        package_spec = _object(_object(installed["distribution"])["npx"])["package"]
        if not isinstance(version, str) or not re.fullmatch(
            r"[0-9][A-Za-z0-9._-]*", version
        ):
            raise SetupError(_MISSING)
        if package_spec != f"{_PACKAGE}@{version}":
            raise SetupError(_MISSING)
    except KeyError, ValueError:
        raise SetupError(_MISSING) from None
    previous_args = previous.get("args")
    old_bin = (
        previous_args[0] if isinstance(previous_args, list) and previous_args else None
    )
    candidates = sorted(
        system_root().glob("*/acp-agents"), key=lambda path: str(path.resolve())
    )
    candidates.sort(
        key=lambda root: (
            not (
                isinstance(old_bin, str)
                and Path(old_bin).is_relative_to(root.resolve())
            )
        )
    )
    for root in candidates:
        directory = root / "claude-acp" / version
        if not (directory / ".installed").is_file():
            continue
        package = (directory / "node_modules" / _PACKAGE).resolve()
        try:
            metadata = _read(package / "package.json")
            if metadata.get("name") != _PACKAGE or metadata.get("version") != version:
                continue
            bin_path = _object(metadata.get("bin"))["claude-agent-acp"]
            engine = _object(metadata.get("engines"))["node"]
            if not isinstance(bin_path, str) or not isinstance(engine, str):
                continue
            minimum = re.fullmatch(r">=(\d+)(?:\.(\d+))?(?:\.(\d+))?", engine.strip())
            if minimum is None:
                continue
            script = (package / bin_path).resolve()
            if not script.is_relative_to(package) or not script.is_file():
                continue
            required = (int(minimum[1]), int(minimum[2] or 0), int(minimum[3] or 0))
            node = _node(root, required, previous)
            if node is not None:
                return str(node), [str(script), "--hide-claude-auth"]
        except KeyError, ValueError, OSError:
            continue
    raise SetupError(_MISSING)


def _connect(
    path: Path,
    document: JsonObject,
    agents: JsonObject,
    previous: JsonObject,
    url: str,
    token: str,
) -> bool:
    command, args = _launch(previous)
    before = json.dumps(document, allow_nan=False)
    entry = dict(previous)
    entry.update(command=command, args=args)
    entry["env"] = {
        **_object(previous.get("env", {})),
        **claude_proxy_values(url, token),
        _MARKER: "1",
    }
    agents[_NAME] = entry
    document["agent_servers"] = agents
    if json.dumps(document, allow_nan=False) == before:
        return False
    atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    return True


def refresh_connected(path: Path, url: str, token: str) -> bool:
    path = path.resolve()
    document = _read(path)
    agents, entry = _entry(document)
    return entry is not None and _connect(path, document, agents, entry, url, token)


def configure(
    path: Path, url: str, token: str, connected: bool | None = None
) -> JsonObject:
    path = path.resolve()
    document = _read(path)
    agents, entry = _entry(document)
    if connected is True:
        _connect(path, document, agents, entry or {}, url, token)
    elif connected is False and entry is not None:
        del agents[_NAME]
        atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    _, entry = _entry(_read(path)) if connected is not None else (agents, entry)
    return {"connected": entry is not None, "paths": {"acp_config": str(path)}}
