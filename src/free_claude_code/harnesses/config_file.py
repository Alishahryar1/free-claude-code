"""Atomic text-file replacement for local client configuration."""

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import cast

import json5

from free_claude_code.core.json_types import JsonObject, JsonValue


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    document: JsonObject = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("Duplicate configuration key")
        document[key] = value
    return document


def _reject_constant(value: str) -> JsonValue:
    raise ValueError("Nonfinite configuration value")


def decode_json(source: str) -> JsonValue:
    """Decode native JSON/JSON5 without duplicate or nonfinite values."""
    try:
        document = json.loads(
            source, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError:
        document = json5.loads(source, allow_duplicate_keys=False)
    json.dumps(document, allow_nan=False)
    return cast(JsonValue, document)


def ensure_private_permissions(path: Path) -> None:
    """Restrict POSIX access; Windows files inherit their profile directory ACL."""
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        path.chmod(0o600)


def atomic_write_text(path: Path, content: str, *, private: bool = False) -> None:
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
        if private and os.name != "nt":
            ensure_private_permissions(temporary)
        elif path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            # Windows cannot delete a temporary file after copying a read-only mode.
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
            temporary.unlink()
