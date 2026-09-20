"""Manage FCC's own saved Claude Desktop gateway configuration."""

import getpass
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

import json5

from free_claude_code.config.server_urls import same_proxy_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.config_file import atomic_write_text

_IDENTITY = (
    "https://github.com/Alishahryar1/free-claude-code/integrations/claude-desktop"
)
FCC_ID = str(uuid5(NAMESPACE_URL, _IDENTITY))
DEFAULT_ID = str(uuid5(NAMESPACE_URL, _IDENTITY + "/default"))
_VIEW_HEADER = "X-FCC-Model-View"


class ManagedDesktopError(ValueError):
    """Local configuration cannot override an organization policy."""


class PendingDisconnectError(ValueError):
    """A new connection must wait for the requested removal to finish."""


class PendingMigrationError(ValueError):
    """Desktop must move its legacy Windows data before FCC creates the new root."""


def config_root() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        parent = Path(os.environ.get("LOCALAPPDATA") or home / "AppData/Local")
    elif sys.platform == "darwin":
        parent = home / "Library/Application Support"
    else:
        parent = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        if not parent.is_absolute():
            parent = home / ".config"
    return parent / "Claude-3p"


def legacy_windows_root() -> Path | None:
    appdata = os.environ.get("APPDATA")
    return Path(appdata) / "Claude-3p" if appdata else None


def _check_pending_migration(root: Path) -> None:
    if sys.platform != "win32":
        return
    try:
        root.stat()
    except FileNotFoundError:
        pass
    else:
        return
    legacy = legacy_windows_root()
    if legacy is not None:
        try:
            legacy.stat()
        except FileNotFoundError:
            return
        raise PendingMigrationError


def check_unmanaged() -> None:
    """Conservatively leave managed installations to their administrator."""
    try:
        if sys.platform == "win32":
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(hive, r"SOFTWARE\Policies\Claude") as key:
                        if winreg.QueryInfoKey(key)[1]:
                            raise ManagedDesktopError
                except FileNotFoundError:
                    pass
        else:
            candidates = (
                [
                    Path(
                        "/Library/Managed Preferences/com.anthropic.claudefordesktop.plist"
                    ),
                    Path("/Library/Managed Preferences")
                    / getpass.getuser()
                    / "com.anthropic.claudefordesktop.plist",
                ]
                if sys.platform == "darwin"
                else [Path("/etc/claude-desktop/managed-settings.json")]
            )
            for path in candidates:
                try:
                    path.stat()
                except FileNotFoundError:
                    continue
                raise ManagedDesktopError
    except OSError:
        raise ManagedDesktopError from None


def _read(path: Path) -> JsonObject | None:
    try:
        value = json5.loads(
            path.read_text(encoding="utf-8-sig"), allow_duplicate_keys=False
        )
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError("Desktop settings must be objects")
    json.dumps(value, allow_nan=False)
    return cast(JsonObject, value)


def _write(path: Path, value: JsonObject) -> bool:
    if _read(path) == value:
        return False
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")
    return True


def _headers(profile: JsonObject) -> dict[str, str]:
    value = profile.get("inferenceCustomHeaders", {})
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("Invalid Desktop custom headers")
    return cast(dict[str, str], value)


def _owned(profile: JsonObject) -> bool:
    return profile.get("inferenceProvider") == "gateway" and any(
        key.lower() == _VIEW_HEADER.lower() and value == "claude-desktop"
        for key, value in _headers(profile).items()
    )


@dataclass
class _Library:
    root: Path
    metadata: JsonObject | None
    entries: list[JsonObject]
    profile: JsonObject | None
    mode: JsonObject

    def file(self, name: str) -> Path:
        return (self.root / name).resolve()

    @property
    def profile_path(self) -> Path:
        return self.file(f"configLibrary/{FCC_ID}.json")

    @property
    def metadata_path(self) -> Path:
        return self.file("configLibrary/_meta.json")

    @property
    def mode_path(self) -> Path:
        return self.file("claude_desktop_config.json")

    @property
    def default_path(self) -> Path:
        return self.file(f"configLibrary/{DEFAULT_ID}.json")

    @property
    def selected(self) -> bool:
        return self.metadata is not None and self.metadata.get("appliedId") == FCC_ID

    @property
    def active(self) -> bool:
        return (
            self.selected
            and self.mode.get("deploymentMode") == "3p"
            and self.metadata is not None
            and not self.metadata.get("hybridPointer")
        )

    def paths(self) -> JsonObject:
        result: JsonObject = {
            "desktop_profile": str(self.profile_path),
            "desktop_library": str(self.metadata_path),
            "desktop_settings": str(self.mode_path),
        }
        if not self.entries or self.selected:
            result["default_profile"] = str(self.default_path)
        return result

    def validate_default(self) -> None:
        if _read(self.default_path) not in (None, {}):
            raise ValueError("Reserved Default configuration is already used")

    def ensure_default(self) -> JsonObject:
        self.validate_default()
        _write(self.default_path, {})
        return {"id": DEFAULT_ID, "name": "Default"}


def _load(root: Path) -> _Library:
    check_unmanaged()
    library = _Library(root.resolve(), None, [], None, {})
    library.metadata = _read(library.metadata_path)
    library.profile = _read(library.profile_path)
    library.mode = _read(library.mode_path) or {}
    if library.metadata is not None:
        entries = library.metadata.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Invalid Desktop configuration library")
        ids: set[str] = set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("id"), str)
                or not isinstance(entry.get("name"), str)
            ):
                raise ValueError("Invalid Desktop entry")
            entry_id = cast(str, entry["id"])
            if not re.fullmatch(r"[a-f0-9-]{36}", entry_id) or entry_id in ids:
                raise ValueError("Invalid or duplicate Desktop entry ID")
            if entry_id == FCC_ID and entry["name"] != "FCC":
                raise ValueError("FCC profile ID is already used")
            ids.add(entry_id)
        applied_id = library.metadata.get("appliedId")
        if not isinstance(applied_id, str) or applied_id not in ids:
            raise ValueError("Invalid Desktop active entry")
        library.entries = cast(list[JsonObject], entries)
    if library.profile is not None:
        _headers(library.profile)
        registered = any(entry["id"] == FCC_ID for entry in library.entries)
        if library.profile.get("inferenceProvider") != "gateway" or (
            not registered and not _owned(library.profile)
        ):
            raise ValueError("FCC profile ID is already used")
    return library


def _values(url: str, token: str) -> JsonObject:
    parsed = urlsplit(url)
    if (
        not token
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Invalid Desktop gateway")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Desktop requires a loopback HTTP or HTTPS gateway")
    return {
        "inferenceProvider": "gateway",
        "inferenceGatewayBaseUrl": url.rstrip("/"),
        "inferenceCredentialKind": "static",
        "inferenceGatewayApiKey": token,
        "inferenceGatewayAuthScheme": "bearer",
        "modelDiscoveryEnabled": True,
    }


def _current(library: _Library, values: JsonObject) -> bool:
    profile = library.profile
    return bool(
        library.active
        and profile is not None
        and _owned(profile)
        and all(
            profile.get(key) == value
            for key, value in values.items()
            if key != "inferenceGatewayBaseUrl"
        )
        and same_proxy_url(
            profile.get("inferenceGatewayBaseUrl"),
            str(values["inferenceGatewayBaseUrl"]),
        )
        and "inferenceModels" not in profile
    )


def _update_profile(library: _Library, values: JsonObject) -> bool:
    profile = dict(library.profile or {})
    headers = {
        k: v for k, v in _headers(profile).items() if k.lower() != _VIEW_HEADER.lower()
    }
    headers[_VIEW_HEADER] = "claude-desktop"
    profile.update(values)
    profile["inferenceCustomHeaders"] = headers
    profile.pop("inferenceModels", None)
    return _write(library.profile_path, profile)


def _disconnect_intent(path: Path, root: Path) -> bool | None:
    record = _read(path)
    if record is None:
        return None
    if (
        set(record) != {"root", "return_to_sign_in"}
        or record["root"] != str(root.resolve())
        or not isinstance(record["return_to_sign_in"], bool)
    ):
        raise ValueError("Invalid Desktop disconnect record")
    return record["return_to_sign_in"]


def _empty_profile(path: Path) -> bool:
    try:
        return _read(path) == {}
    except ValueError, UnicodeError:
        # A user may change the neutral profile between a failure and retry.
        return False


def _disconnect(library: _Library, path: Path, intent: bool | None) -> None:
    has_entry = any(entry["id"] == FCC_ID for entry in library.entries)
    if not has_entry and library.profile is None and intent is None:
        return
    if library.selected:
        library.validate_default()
    return_to_sign_in = intent is True or library.selected
    _write(path, {"root": str(library.root), "return_to_sign_in": return_to_sign_in})
    if library.selected:
        default = library.ensure_default()
        entries = list(library.entries)
        if not any(entry["id"] == DEFAULT_ID for entry in entries):
            entries.append(default)
        metadata = dict(library.metadata or {})
        metadata.update({"entries": entries, "appliedId": DEFAULT_ID})
        metadata.pop("hybridPointer", None)
        _write(library.metadata_path, metadata)
        library = _load(library.root)
    if (
        return_to_sign_in
        and library.metadata is not None
        and library.metadata.get("appliedId") == DEFAULT_ID
        and not library.metadata.get("hybridPointer")
        and _empty_profile(library.default_path)
    ):
        _write(library.mode_path, library.mode | {"deploymentMode": "1p"})
    # Keep the registered ownership until unlink succeeds, even if its header
    # was changed. A retry after unlink safely accepts the missing profile.
    library.profile_path.unlink(missing_ok=True)
    library = _load(library.root)
    entries = [entry for entry in library.entries if entry["id"] != FCC_ID]
    if len(entries) != len(library.entries):
        metadata = dict(library.metadata or {})
        metadata["entries"] = entries
        _write(library.metadata_path, metadata)
    remaining = _load(library.root)
    if remaining.profile is not None or any(
        entry["id"] == FCC_ID for entry in remaining.entries
    ):
        raise ValueError("Desktop disconnect cleanup is incomplete")
    path.unlink(missing_ok=True)


def refresh_connected(
    root: Path, proxy_root_url: str, auth_token: str, *, disconnect_path: Path
) -> bool:
    library = _load(root)
    if _disconnect_intent(disconnect_path, root) is not None:
        return False
    if not library.active or library.profile is None:
        return False
    return _update_profile(library, _values(proxy_root_url, auth_token))


def configure(
    root: Path,
    proxy_root_url: str,
    auth_token: str,
    connected: bool | None = None,
    *,
    disconnect_path: Path,
) -> JsonObject:
    library = _load(root)
    disconnect_path = disconnect_path.resolve()
    intent = _disconnect_intent(disconnect_path, root)
    if connected is True:
        if intent is not None:
            raise PendingDisconnectError
        _check_pending_migration(library.root)
        values = _values(proxy_root_url, auth_token)
        if not library.entries:
            library.validate_default()
            library.entries = [library.ensure_default()]
        _update_profile(library, values)
        entries = [entry for entry in library.entries if entry["id"] != FCC_ID]
        entries.append({"id": FCC_ID, "name": "FCC"})
        metadata = dict(library.metadata or {})
        metadata.update({"entries": entries, "appliedId": FCC_ID})
        metadata.pop("hybridPointer", None)
        _write(library.metadata_path, metadata)
        _write(library.mode_path, library.mode | {"deploymentMode": "3p"})
    elif connected is False:
        _disconnect(library, disconnect_path, intent)
    if connected is not None:
        library = _load(root)
    is_connected = (
        library.active and library.profile is not None and connected is not False
    )
    if is_connected:
        is_connected = _current(library, _values(proxy_root_url, auth_token))
    return {
        "connected": is_connected,
        "disconnect_pending": _disconnect_intent(disconnect_path, root) is not None,
        "paths": library.paths() | {"disconnect_record": str(disconnect_path)},
    }
