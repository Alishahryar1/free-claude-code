"""Configure FCC's model group in the native VS Code chat model picker."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import json5

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.claude_integration import settings_path
from free_claude_code.harnesses.config_file import atomic_write_text

_NAME = "Free Claude Code"
_MARKER = "fccIntegration"


def config_path() -> Path:
    return settings_path().with_name("chatLanguageModels.json")


def model_entry(model: CatalogModel, proxy_root_url: str) -> JsonObject:
    context = model.context_window_tokens
    context = context if context is not None and context >= 2 else 32768
    output = model.max_output_tokens
    output = output if output is not None and output > 0 else 4096
    output = min(output, context // 2)
    return {
        "id": model.wire_slug,
        "name": model.display_name,
        "url": proxy_root_url.rstrip("/") + "/v1/messages",
        "toolCalling": True,
        "vision": ModelInputModality.IMAGE in (model.input_modalities or ()),
        "maxInputTokens": context - output,
        "maxOutputTokens": output,
    }


def _read(path: Path) -> tuple[list[JsonObject], JsonObject | None]:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return [], None
    document = json5.loads(source, allow_duplicate_keys=False)
    if not isinstance(document, list) or any(not isinstance(g, dict) for g in document):
        raise ValueError("Model configuration must be an array of groups")
    json.dumps(document, allow_nan=False)
    groups = cast(list[JsonObject], document)
    owned = [g for g in groups if g.get(_MARKER) == "vscode"]
    if len(owned) > 1:
        raise ValueError("Duplicate FCC model groups")
    group = owned[0] if owned else None
    if group is not None and (
        group.get("vendor") != "customendpoint"
        or not isinstance(group.get("name"), str)
        or not group["name"]
    ):
        raise ValueError("Invalid FCC model group")
    return groups, group


def status(path: Path) -> JsonObject:
    path = path.resolve()
    _, group = _read(path)
    return {"connected": group is not None, "paths": {"vscode_models": str(path)}}


def configure(
    path: Path,
    proxy_root_url: str,
    auth_token: str,
    models: Sequence[CatalogModel],
    *,
    only_existing: bool = False,
) -> bool:
    """Merge the generated group, preserving other groups and user model settings."""
    path = path.resolve()
    groups, group = _read(path)
    if group is None and only_existing:
        return False
    if not auth_token.strip():
        raise ValueError("FCC's managed token is required")
    name = group["name"] if group is not None else _NAME
    if any(
        g is not group and g.get("vendor") == "customendpoint" and g.get("name") == name
        for g in groups
    ):
        raise ValueError("Rename the conflicting custom endpoint group in VS Code")
    before = json.dumps(groups, allow_nan=False)
    if group is None:
        new_group: JsonObject = {"name": _NAME}
        group = new_group
        groups.append(group)
    group.update(
        {
            "vendor": "customendpoint",
            _MARKER: "vscode",
            "apiType": "messages",
            "models": [
                {
                    **model_entry(model, proxy_root_url),
                    "requestHeaders": {"x-api-key": auth_token},
                }
                for model in models
            ],
        }
    )
    # A provider URL selects VS Code's discovery branch instead of these entries.
    group.pop("url", None)
    # VS Code treats group apiKey as a secret-store reference, not a raw token.
    group.pop("apiKey", None)
    if json.dumps(groups, allow_nan=False) == before:
        return False
    atomic_write_text(path, json.dumps(groups, indent=2, allow_nan=False) + "\n")
    return True


def disconnect(path: Path) -> None:
    path = path.resolve()
    groups, group = _read(path)
    if group is not None:
        groups.remove(group)
        atomic_write_text(path, json.dumps(groups, indent=2, allow_nan=False) + "\n")
