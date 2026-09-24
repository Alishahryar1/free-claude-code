"""Register the installed FCC OpenCode launcher in Devin Desktop's ACP registry."""

import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import cast

import json5

from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.harnesses.config_file import atomic_write_text
from free_claude_code.harnesses.opencode import STABLE_VERSION_PATTERN

_ID = "fcc-opencode"
_MARKER = "FCC_DEVIN_ACP"
_REPOSITORY = "https://github.com/Alishahryar1/free-claude-code"


class SetupError(ValueError):
    """Safe, actionable prerequisite or registration conflict."""


def config_path() -> Path:
    relative = (
        "AppData/Roaming/Code/User/acp/registry.json"
        if sys.platform == "win32"
        else ".windsurf/acp/registry.json"
    )
    return Path.home() / relative


def platform_key() -> str:
    system = {"win32": "windows", "darwin": "darwin", "linux": "linux"}.get(
        sys.platform
    )
    arch = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(platform.machine().lower())
    if system is None or arch is None:
        raise SetupError("Devin ACP requires Windows, macOS or Linux on x64 or ARM64.")
    return f"{system}-{arch}"


def _object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return cast(JsonObject, value)


def _read(path: Path) -> JsonObject:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    document = _object(json5.loads(source, allow_duplicate_keys=False))
    json.dumps(document, allow_nan=False)
    return document


def _entry(document: JsonObject) -> tuple[list[JsonValue], JsonObject | None]:
    agents = document.get("agents", [])
    if not isinstance(agents, list):
        raise ValueError("Expected an agents array")
    matches = [_object(agent) for agent in agents if _object(agent).get("id") == _ID]
    if len(matches) > 1:
        raise SetupError(
            "Duplicate FCC agent IDs in Devin's registry. Remove the duplicate before retrying."
        )
    if not matches:
        return agents, None
    entry = matches[0]
    binary = _object(_object(entry.get("distribution", {})).get("binary", {}))
    target = _object(binary.get(platform_key(), {}))
    env = _object(target.get("env", {}))
    if any(not isinstance(value, str) for value in env.values()):
        raise ValueError("Environment values must be strings")
    if env.get(_MARKER) != "1":
        raise SetupError(
            "An unrelated agent uses the ID fcc-opencode. Rename it in Devin's registry before connecting."
        )
    return agents, entry


def _launch() -> tuple[str, str]:
    name = "fcc-opencode.exe" if sys.platform == "win32" else "fcc-opencode"
    launcher = Path(sysconfig.get_path("scripts")) / name
    native = shutil.which("opencode")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SetupError(
            "Could not find this FCC installation's fcc-opencode command. Rerun the FCC installer and restart FCC."
        )
    if native is None:
        raise SetupError(
            "Install OpenCode 2 using the FCC installer, then restart FCC and retry."
        )
    try:
        result = subprocess.run(
            [native, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except OSError, subprocess.TimeoutExpired:
        raise SetupError(
            "Could not check OpenCode. Install stable OpenCode 2 and retry."
        ) from None
    if result.returncode != 0 or STABLE_VERSION_PATTERN.search(result.stdout) is None:
        raise SetupError("FCC requires stable OpenCode 2. Upgrade OpenCode and retry.")
    parts = [
        str(Path(native).absolute().parent),
        *os.environ.get("PATH", "").split(os.pathsep),
    ]
    seen: set[str] = set()
    paths = []
    for part in parts:
        key = os.path.normcase(os.path.normpath(part))
        if part and key not in seen:
            paths.append(part)
            seen.add(key)
    return str(launcher.absolute()), os.pathsep.join(paths)


def _connect(
    path: Path,
    document: JsonObject,
    agents: list[JsonValue],
    previous: JsonObject | None,
) -> bool:
    before = json.dumps(document, allow_nan=False)
    key = platform_key()
    command, search_path = _launch()
    entry = dict(previous or {})
    distribution = dict(_object(entry.get("distribution", {})))
    binary = dict(_object(distribution.get("binary", {})))
    target = dict(_object(binary.get(key, {})))
    env = dict(_object(target.get("env", {})))
    for name in list(env):
        if name == "PATH" or (sys.platform == "win32" and name.upper() == "PATH"):
            del env[name]
    env.update({_MARKER: "1", "PATH": search_path})
    target.update(archive="", cmd=command, args=["acp"], env=env)
    binary[key] = target
    distribution["binary"] = binary
    entry.update(
        id=_ID,
        name="OpenCode (FCC)",
        version="1.0.0",
        description="OpenCode through your Free Claude Code server",
        repository=_REPOSITORY,
        license="MIT",
        license_url=f"{_REPOSITORY}/blob/main/LICENSE",
        distribution=distribution,
    )
    if previous is None:
        agents.append(entry)
    else:
        agents[agents.index(previous)] = entry
    document["agents"] = agents
    document.setdefault("version", "1.0.0")
    document.setdefault("extensions", [])
    if json.dumps(document, allow_nan=False) == before:
        return False
    atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    return True


def refresh_connected(path: Path) -> bool:
    path = path.resolve()
    document = _read(path)
    agents, entry = _entry(document)
    return entry is not None and _connect(path, document, agents, entry)


def configure(path: Path, connected: bool | None = None) -> JsonObject:
    path = path.resolve()
    document = _read(path)
    agents, entry = _entry(document)
    if connected is True:
        _connect(path, document, agents, entry)
    elif connected is False and entry is not None:
        agents.remove(entry)
        atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    _, entry = _entry(_read(path)) if connected is not None else (agents, entry)
    return {"connected": entry is not None, "paths": {"acp_config": str(path)}}
