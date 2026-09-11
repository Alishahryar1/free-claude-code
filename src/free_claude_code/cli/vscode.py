"""Read and update the standard global Claude Code settings for VS Code."""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import json5

from free_claude_code.cli.claude_env import claude_proxy_values
from free_claude_code.core.json_types import JsonObject

_ENV = "claudeCode.environmentVariables"
_LOGIN = "claudeCode.disableLoginPrompt"
_ONBOARDING = "hasCompletedOnboarding"


def settings_path() -> Path:
    """Locate the current user's standard native VS Code settings."""
    home = Path.home()
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA") or home / "AppData/Roaming")
    elif sys.platform == "darwin":
        root = home / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        if not root.is_absolute():
            root = home / ".config"
    return root / "Code/User/settings.json"


def claude_state_path() -> Path:
    return Path.home() / ".claude.json"


def _read_object(path: Path) -> JsonObject:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    document = json5.loads(source, allow_duplicate_keys=False)
    if not isinstance(document, dict):
        raise ValueError("Settings must be an object")
    # Also reject non-finite JSON5 numbers before any operation or status result.
    json.dumps(document, allow_nan=False)
    return cast(JsonObject, document)


def _read(path: Path, names: set[str]) -> tuple[JsonObject, list[JsonObject]]:
    document = _read_object(path)
    entries = document.get(_ENV, [])
    if not isinstance(entries, list):
        raise ValueError("Environment settings must be an array")
    seen: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("value"), str)
        ):
            raise ValueError("Environment entries require a name and value")
        name = entry["name"]
        if name in names and name in seen:
            raise ValueError("Duplicate integration environment entry")
        seen.add(name)
    return document, cast(list[JsonObject], entries)


def _same_url(value: object, expected: str) -> bool:
    if not isinstance(value, str):
        return False

    def normalized(url: str) -> tuple[str, str | None, int | None, str]:
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Unexpected URL components")
        host = parsed.hostname
        if host in {"localhost", "127.0.0.1", "::1"}:
            host = "localhost"
        return parsed.scheme, host, parsed.port, parsed.path.rstrip("/")

    try:
        return normalized(value) == normalized(expected)
    except ValueError:
        return False


def _connected(
    document: JsonObject, entries: list[JsonObject], values: dict[str, str]
) -> bool:
    environment = {entry["name"]: entry["value"] for entry in entries}
    return (
        document.get(_LOGIN) is True
        and _same_url(
            environment.get("ANTHROPIC_BASE_URL"), values["ANTHROPIC_BASE_URL"]
        )
        and environment.get("ANTHROPIC_AUTH_TOKEN") == values["ANTHROPIC_AUTH_TOKEN"]
        and environment.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            # Windows cannot delete a temporary file after copying a read-only mode.
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
            temporary.unlink()


def configure(
    path: Path,
    state_path: Path,
    proxy_root_url: str,
    auth_token: str,
    connected: bool | None = None,
) -> JsonObject:
    """Inspect, connect, or disconnect; preserve unrelated values, not formatting."""
    path = path.resolve()
    values = claude_proxy_values(proxy_root_url, auth_token)
    document, entries = _read(path, set(values))
    onboarding: JsonObject = {}
    if connected is not False:
        state_path = state_path.resolve()
        onboarding = _read_object(state_path)
    if connected is not None:
        before = json.dumps(document, allow_nan=False)
        if connected:
            document[_LOGIN] = True
            remaining = dict(values)
            for entry in entries:
                name = cast(str, entry["name"])
                if name in remaining:
                    entry["value"] = remaining.pop(name)
            entries.extend(
                {"name": name, "value": value} for name, value in remaining.items()
            )
            document[_ENV] = entries
        else:
            document.pop(_LOGIN, None)
            retained = [entry for entry in entries if entry["name"] not in values]
            if len(retained) != len(entries):
                if retained:
                    document[_ENV] = retained
                else:
                    document.pop(_ENV, None)
        settings_changed = json.dumps(document, allow_nan=False) != before
        settings_content = json.dumps(document, indent=2, allow_nan=False) + "\n"
        if connected and onboarding.get(_ONBOARDING) is not True:
            onboarding[_ONBOARDING] = True
            _write(state_path, json.dumps(onboarding, indent=2, allow_nan=False) + "\n")
        if settings_changed:
            _write(path, settings_content)
        document, entries = _read(path, set(values))
        if connected:
            onboarding = _read_object(state_path)
    return {
        "connected": _connected(document, entries, values)
        and onboarding.get(_ONBOARDING) is True,
    }
