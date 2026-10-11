"""Read or complete user-global Claude Code onboarding independently of editors."""

import json
from pathlib import Path
from typing import cast

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.config_file import atomic_write_text, decode_json

_ONBOARDING = "hasCompletedOnboarding"


def state_path() -> Path:
    return Path.home() / ".claude.json"


def _read(path: Path) -> JsonObject:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    document = decode_json(source)
    if not isinstance(document, dict):
        raise ValueError("Claude state must be an object")
    return cast(JsonObject, document)


def status(path: Path) -> JsonObject:
    path = path.resolve()
    return {
        "completed": _read(path).get(_ONBOARDING) is True,
        "paths": {"claude_state": str(path)},
    }


def complete(path: Path) -> JsonObject:
    path = path.resolve()
    document = _read(path)
    changed = document.get(_ONBOARDING) is not True
    if changed:
        document[_ONBOARDING] = True
        atomic_write_text(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    return {"completed": True, "changed": changed}
