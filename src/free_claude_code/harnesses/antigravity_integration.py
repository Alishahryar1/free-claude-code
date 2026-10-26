"""Explicit ownership of Antigravity's persistent Gemini-provider settings."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.config import paths
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.config_file import atomic_write_text

_LABEL_PREFIX = "FCC: "
_MODEL_PREFIX = "gemini-api://models/"
_NATIVE_TOOLS = "TOOL_FORMATTER_TYPE_NONE"


class DisconnectedError(ValueError):
    def __init__(self) -> None:
        super().__init__("Antigravity CLI is disconnected from FCC.")


def config_path() -> Path:
    return Path.home() / ".gemini/antigravity-cli/settings.json"


def state_path() -> Path:
    return paths.config_dir_path() / "antigravity-integration.json"


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate Antigravity configuration key")
        result[key] = value
    return result


def _reject_constant(value: str) -> JsonValue:
    raise ValueError("Nonfinite Antigravity configuration value")


def _read(path: Path) -> JsonObject:
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    value = json.loads(
        source, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError("Antigravity settings and connection records must be objects")
    return value


def _models(document: JsonObject) -> JsonObject:
    config = document.get("customModelsConfig", {})
    if not isinstance(config, dict):
        raise ValueError("Antigravity customModelsConfig must be an object")
    models = config.get("customModels", {})
    if not isinstance(models, dict) or any(
        not isinstance(value, dict) for value in models.values()
    ):
        raise ValueError("Antigravity customModels must contain model objects")
    return models


def _provider(document: JsonObject) -> dict[str, str]:
    if "modelProvider" not in document:
        return {}
    value = document["modelProvider"]
    if not isinstance(value, str):
        raise ValueError("Antigravity modelProvider must be a string")
    return {"modelProvider": value}


@dataclass(frozen=True)
class _Connection:
    previous_provider: dict[str, str]
    model_keys: frozenset[str]


def _connection(state: Path, path: Path) -> _Connection | None:
    try:
        state.stat()
    except FileNotFoundError:
        return None
    record = _read(state)
    keys, previous = record.get("model_keys"), record.get("previous_provider")
    if (
        record.get("settings_path") != str(path)
        or not isinstance(keys, list)
        or not all(isinstance(key, str) and key for key in keys)
        or not isinstance(previous, dict)
        or set(previous) - {"modelProvider"}
    ):
        raise ValueError(
            "Invalid Antigravity connection record; leave it intact for recovery"
        )
    return _Connection(_provider(previous), frozenset(str(key) for key in keys))


@contextmanager
def _locked(state: Path) -> Generator[None]:
    lock = InterprocessFileLock(state.with_suffix(".lock"))
    if not lock.acquire(wait=True, timeout=5):
        raise ValueError("Another FCC process is updating Antigravity; retry shortly")
    try:
        yield
    finally:
        lock.release()


def _write_state(state: Path, path: Path, connection: _Connection) -> None:
    value: JsonObject = {
        "settings_path": str(path),
        "previous_provider": dict(connection.previous_provider),
        "model_keys": sorted(connection.model_keys),
    }
    if _read(state) != value:
        atomic_write_text(state, json.dumps(value, indent=2) + "\n", private=True)


def _write_native(path: Path, before: JsonObject, after: JsonObject) -> None:
    if _read(path) != before:
        raise ValueError("Antigravity settings changed during configuration; retry")
    if before != after:
        atomic_write_text(path, json.dumps(after, indent=2) + "\n")


def _owned(key: str, value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    label = value.get("label")
    return (
        value.get("modelName") == _MODEL_PREFIX + key
        and value.get("toolFormatterType") == _NATIVE_TOOLS
        and isinstance(label, str)
        and label.startswith(_LABEL_PREFIX)
    )


def _catalog_models(catalog: ModelCatalog) -> JsonObject:
    if not catalog.models:
        raise ValueError("Antigravity requires at least one routable FCC model")
    # AGY 1.4.7 has no custom output limit, and small maxTokens values prevent
    # startup. Keep native capacity defaults rather than inventing model limits.
    return {
        model.wire_slug: {
            "modelName": _MODEL_PREFIX + model.wire_slug,
            "label": _LABEL_PREFIX + model.display_name,
            "toolFormatterType": _NATIVE_TOOLS,
            "modelFeatures": {
                "supportsImages": model.input_modalities is None
                or ModelInputModality.IMAGE in model.input_modalities,
                "supportsVideo": False,
                "supportsAudio": False,
                "supportsPdf": False,
                "supportsToolCalls": True,
                "supportsThinking": model.supports_reasoning is not False,
            },
        }
        for model in catalog.models
    }


def configure(
    path: Path, state: Path, catalog: ModelCatalog, *, connect: bool = False
) -> None:
    """Connect explicitly, or refresh a recognized connection without changing mode."""
    path, state = path.resolve(), state.resolve()
    with _locked(state):
        document = _read(path)
        models = _models(document)
        provider = _provider(document)
        connection = _connection(state, path)
        if not connect and (
            connection is None or provider.get("modelProvider") != "gemini"
        ):
            raise DisconnectedError()
        owned_keys = connection.model_keys if connection is not None else frozenset()
        generated = _catalog_models(catalog)
        for key in generated.keys() & models.keys():
            if key not in owned_keys or not _owned(key, models[key]):
                raise ValueError(
                    f"Antigravity model '{key}' belongs to your configuration. "
                    "Rename that entry before connecting or refreshing FCC."
                )
        previous = (
            connection.previous_provider
            if connection is not None and provider.get("modelProvider") == "gemini"
            else provider
        )
        retained = {
            key: value
            for key, value in models.items()
            if key not in owned_keys or not _owned(key, value)
        }
        config = document.get("customModelsConfig", {})
        assert isinstance(config, dict)
        updated: JsonObject = {
            **document,
            "modelProvider": "gemini",
            "customModelsConfig": {**config, "customModels": {**retained, **generated}},
        }
        desired_keys = frozenset(generated)
        # Keep both sets recoverable if the native write or final record write fails.
        _write_state(state, path, _Connection(previous, owned_keys | desired_keys))
        _write_native(path, document, updated)
        _write_state(state, path, _Connection(previous, desired_keys))


def status(path: Path, state: Path) -> JsonObject:
    """Recognize existing connections without a catalog or native settings writes."""
    path, state = path.resolve(), state.resolve()
    connected = can_disconnect = False
    try:
        state.stat()
    except FileNotFoundError:
        pass
    else:
        with _locked(state):
            connection = _connection(state, path)
            can_disconnect = connection is not None
            document = _read(path)
            _models(document)
            connected = (
                connection is not None
                and _provider(document).get("modelProvider") == "gemini"
            )
    return {
        "connected": connected,
        "can_disconnect": can_disconnect,
        "paths": {"settings": str(path), "connection": str(state)},
    }


def disconnect(path: Path, state: Path) -> bool:
    """Remove recognized FCC models and restore only the still-selected FCC mode."""
    path, state = path.resolve(), state.resolve()
    try:
        state.stat()
    except FileNotFoundError:
        return False
    with _locked(state):
        connection = _connection(state, path)
        if connection is None:
            return False
        document = _read(path)
        models = _models(document)
        _provider(document)
        retained = {
            key: value
            for key, value in models.items()
            if key not in connection.model_keys or not _owned(key, value)
        }
        updated = dict(document)
        if retained != models:
            config = document.get("customModelsConfig", {})
            assert isinstance(config, dict)
            config = dict(config)
            if retained:
                config["customModels"] = retained
            else:
                config.pop("customModels", None)
            if config:
                updated["customModelsConfig"] = config
            else:
                updated.pop("customModelsConfig", None)
        if updated.get("modelProvider") == "gemini":
            updated.pop("modelProvider")
            updated.update(connection.previous_provider)
        _write_native(path, document, updated)
        state.unlink()
        return True
