"""Installer-only Headroom registration and native configuration checks."""

import json
import os
import shutil
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.harnesses.config_file import atomic_write_text, decode_json


class McpSetupError(ValueError):
    """An installation cannot proceed without replacing user configuration."""


def read_object(path: Path) -> JsonObject:
    """Read a native JSON/JSONC or TOML document without logging its contents."""
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    try:
        value = tomllib.loads(source) if path.suffix == ".toml" else decode_json(source)
    except ValueError:
        raise McpSetupError(
            f"Invalid configuration at {path}. Correct it and retry."
        ) from None
    if not isinstance(value, dict):
        raise McpSetupError(f"Expected a configuration object at {path}.")
    return cast(JsonObject, value)


def server_entry(document: JsonObject, table: tuple[str, ...]) -> JsonObject | None:
    value: JsonValue = document
    for key in (*table, "headroom"):
        if not isinstance(value, dict):
            raise McpSetupError("Invalid MCP configuration structure.")
        if key not in value:
            return None
        value = value[key]
    if not isinstance(value, dict):
        raise McpSetupError("The existing headroom entry must be an object.")
    return value


def _same_command(actual: JsonValue, expected: JsonValue) -> bool:
    if actual == expected:
        return True
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    resolved = shutil.which(actual) if not Path(actual).is_absolute() else actual
    return resolved is not None and os.path.normcase(
        os.path.realpath(resolved)
    ) == os.path.normcase(os.path.realpath(expected))


def _matches(actual: JsonObject, expected: JsonObject) -> bool:
    # A custom environment, URL, or working directory changes the launch contract.
    if any(actual.get(key) for key in ("env", "url", "cwd", "headers")):
        return False
    for key in ("type", "transport"):
        if key not in expected and key in actual and actual[key] != "stdio":
            return False
    for key, value in expected.items():
        other = actual.get(key)
        if key == "command" and isinstance(value, str):
            if not _same_command(other, value):
                return False
        elif isinstance(value, dict):
            if not isinstance(other, dict) or not _matches(other, value):
                return False
        elif (key in {"type", "transport"} and other is None and value == "stdio") or (
            key in {"enabled", "required", "mode"} and other is None
        ):
            continue
        elif other != value:
            return False
    return True


def _disabled(entry: JsonObject) -> bool:
    return (
        entry.get("enabled") is False
        or entry.get("disabled") is True
        or entry.get("enabled_tools") == []
        or bool(entry.get("blocked_reason"))
    )


@dataclass(frozen=True)
class Registration:
    label: str
    path: Path
    expected: JsonObject
    read: Callable[[], JsonObject | None]
    write: Callable[[], None]
    blocked: str | None = None

    def state(self) -> Literal["missing", "existing", "disabled"]:
        if self.blocked:
            return "disabled"
        entry = self.read()
        if entry is None:
            return "missing"
        if _disabled(entry):
            return "disabled"
        if not _matches(entry, self.expected):
            raise McpSetupError(
                f"{self.label}: conflicting headroom entry at {self.path}. "
                "Keep it and skip the bundle, or rename/remove it using the native client before retrying."
            )
        return "existing"


def configure_registrations(
    targets: Sequence[Registration], *, check_only: bool = False
) -> list[str]:
    """Preflight all scopes, then perform and verify only missing registrations."""
    for target in targets:
        target.state()
    outcomes = []
    for target in targets:
        state = target.state()
        if state == "disabled":
            outcomes.append(
                f"{target.label}: skipped ({target.blocked or 'Headroom is disabled or blocked by policy'})"
            )
        elif state == "existing":
            outcomes.append(f"{target.label}: already configured")
        elif check_only:
            outcomes.append(f"{target.label}: ready")
        else:
            target.write()
            if target.state() != "existing":
                raise McpSetupError(
                    f"{target.label}: could not verify Headroom registration at {target.path}."
                )
            outcomes.append(f"{target.label}: configured")
    return outcomes


def json_registration(
    label: str,
    path: Path,
    table: tuple[str, ...],
    headroom: Path,
    *,
    expected: JsonObject | None = None,
    write: Callable[[], None] | None = None,
    defaults: JsonObject | None = None,
    blocked: str | None = None,
) -> Registration:
    entry: JsonObject = (
        expected
        if expected is not None
        else {"command": str(headroom), "args": ["mcp", "serve"]}
    )

    def read() -> JsonObject | None:
        return server_entry(read_object(path), table)

    def write_json() -> None:
        if path.is_symlink():
            raise McpSetupError(f"Cannot replace linked configuration at {path}.")
        document = read_object(path)
        if defaults:
            for key, value in defaults.items():
                document.setdefault(key, value)
        parent = document
        for key in table:
            child = parent.setdefault(key, {})
            if not isinstance(child, dict):
                raise McpSetupError(f"Invalid MCP configuration at {path}.")
            parent = child
        # Reread immediately before modification as well as in orchestration.
        if "headroom" in parent:
            raise McpSetupError(
                f"{label}: headroom configuration changed during setup. Retry the installer."
            )
        parent["headroom"] = entry
        atomic_write_text(
            path,
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            private=True,
        )

    return Registration(label, path, entry, read, write or write_json, blocked)
