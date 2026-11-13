"""Read and update the standard global Claude Code settings for VS Code."""

import json
import os
import sys
from pathlib import Path
from typing import cast

from free_claude_code.config.server_urls import same_proxy_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.claude import claude_proxy_values
from free_claude_code.harnesses.config_file import atomic_write_text, decode_json

_ENV = "claudeCode.environmentVariables"
_LOGIN = "claudeCode.disableLoginPrompt"


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


def _read_object(path: Path) -> JsonObject:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    document = decode_json(source)
    if not isinstance(document, dict):
        raise ValueError("Settings must be an object")
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


def _connected(
    document: JsonObject, entries: list[JsonObject], values: dict[str, str]
) -> bool:
    environment = {entry["name"]: entry["value"] for entry in entries}
    return (
        document.get(_LOGIN) is True
        and same_proxy_url(
            environment.get("ANTHROPIC_BASE_URL"), values["ANTHROPIC_BASE_URL"]
        )
        and environment.get("ANTHROPIC_AUTH_TOKEN") == values["ANTHROPIC_AUTH_TOKEN"]
        and environment.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"
    )


def _connect(
    path: Path,
    document: JsonObject,
    entries: list[JsonObject],
    values: dict[str, str],
) -> bool:
    before = json.dumps(document, allow_nan=False)
    document[_LOGIN] = True
    remaining = dict(values)
    for entry in entries:
        name = cast(str, entry["name"])
        if name in remaining:
            entry["value"] = remaining.pop(name)
    entries.extend({"name": name, "value": value} for name, value in remaining.items())
    document[_ENV] = entries
    settings_changed = json.dumps(document, allow_nan=False) != before
    if settings_changed:
        atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    return settings_changed


def refresh_connected(path: Path, proxy_root_url: str, auth_token: str) -> bool:
    """Apply current settings only to an existing connection, even from release one."""
    path = path.resolve()
    values = claude_proxy_values(proxy_root_url, auth_token)
    document, entries = _read(path, set(values))
    if not _connected(document, entries, values):
        return False
    return _connect(path, document, entries, values)


def _disconnect(
    path: Path, document: JsonObject, entries: list[JsonObject], values: dict[str, str]
) -> None:
    before = json.dumps(document, allow_nan=False)
    document.pop(_LOGIN, None)
    retained = [entry for entry in entries if entry["name"] not in values]
    if len(retained) != len(entries):
        if retained:
            document[_ENV] = retained
        else:
            document.pop(_ENV, None)
    if json.dumps(document, allow_nan=False) != before:
        atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")


def disconnect_existing(
    path: Path, proxy_root_url: str | None, auth_token: str | None
) -> bool:
    """Remove a recognized connection; stop cleanup when ownership is uncertain."""
    path = path.resolve()
    values = claude_proxy_values(proxy_root_url or "", auth_token or "")
    document, entries = _read(path, set(values))
    environment = {entry["name"]: entry["value"] for entry in entries}
    matching_url = proxy_root_url is not None and same_proxy_url(
        environment.get("ANTHROPIC_BASE_URL"), proxy_root_url
    )
    matching_token = (
        auth_token is not None and environment.get("ANTHROPIC_AUTH_TOKEN") == auth_token
    )
    # Login and discovery preferences can change without changing the gateway.
    if matching_url and matching_token:
        _disconnect(path, document, entries, values)
        return True
    if (
        matching_url
        or matching_token
        or (
            document.get(_LOGIN) is True
            and environment.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"
        )
    ):
        raise ValueError("Existing gateway connection identity cannot be confirmed")
    return False


def configure(
    path: Path,
    proxy_root_url: str,
    auth_token: str,
    connected: bool | None = None,
) -> JsonObject:
    """Inspect, connect, or disconnect; preserve unrelated values, not formatting."""
    path = path.resolve()
    values = claude_proxy_values(proxy_root_url, auth_token)
    document, entries = _read(path, set(values))
    if connected is True:
        _connect(path, document, entries, values)
    elif connected is False:
        _disconnect(path, document, entries, values)
    if connected is not None:
        document, entries = _read(path, set(values))
    result: JsonObject = {"connected": _connected(document, entries, values)}
    if connected is None:
        result["paths"] = {
            "vscode_settings": str(path),
        }
    return result
