"""Antigravity connection ownership preserves native configuration and history."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses import antigravity_integration as integration


def catalog(*ids: str) -> ModelCatalog:
    return ModelCatalog(
        tuple(
            CatalogModel(
                name,
                name,
                name,
                False,
                frozenset({ModelInputModality.TEXT}),
                context_window_tokens=32000,
                max_output_tokens=4096,
            )
            for name in ids
        ),
        ids[-1],
    )


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def native_files(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / ".gemini/antigravity-cli/settings.json"
    state = tmp_path / ".fcc/antigravity-integration.json"
    write(
        path,
        {
            "modelProvider": "",
            "model": "my-native-choice",
            "colorScheme": "dark",
            "customModelsConfig": {
                "keep": True,
                "customModels": {"personal": {"modelName": "my-personal-model"}},
            },
            "permissions": {"allow": ["read_file"]},
        },
    )
    return path, state


def test_connect_refresh_disconnect_preserve_native_settings(native_files):
    path, state = native_files
    original = read(path)
    models = catalog("provider/first", "provider/selected:free")
    integration.configure(path, state, models, connect=True)
    connected = read(path)
    assert connected["modelProvider"] == "gemini"
    assert connected["model"] == original["model"]
    custom = connected["customModelsConfig"]["customModels"]
    assert (
        custom["personal"] == original["customModelsConfig"]["customModels"]["personal"]
    )
    selected = custom[models.default_model_id]
    assert selected["modelName"] == "gemini-api://models/provider/selected:free"
    assert selected["toolFormatterType"] == "TOOL_FORMATTER_TYPE_NONE"
    assert selected["modelFeatures"] == {
        "supportsImages": False,
        "supportsVideo": False,
        "supportsAudio": False,
        "supportsPdf": False,
        "supportsToolCalls": True,
        "supportsThinking": False,
    }
    # Native AGY cannot honor small custom contexts or configurable output limits.
    assert "maxTokens" not in selected and "maxOutputTokens" not in selected
    before = path.read_bytes(), state.read_bytes()
    integration.configure(path, state, models)
    assert (path.read_bytes(), state.read_bytes()) == before

    integration.configure(path, state, catalog("provider/replacement"))
    assert set(read(path)["customModelsConfig"]["customModels"]) == {
        "personal",
        "provider/replacement",
    }
    assert integration.disconnect(path, state)
    assert read(path) == original
    assert not state.exists()
    assert not integration.disconnect(path, state)


@pytest.mark.parametrize("previous", [{}, {"modelProvider": "gemini"}])
def test_disconnect_restores_provider_field_presence_and_existing_key_mode(
    native_files, previous
):
    path, state = native_files
    write(path, {**previous, "keep": "native"})
    integration.configure(path, state, catalog("provider/model"), connect=True)
    integration.disconnect(path, state)
    assert read(path) == {**previous, "keep": "native"}


def test_unconnected_launch_requires_explicit_connect_without_mutation(native_files):
    path, state = native_files
    before = path.read_bytes()
    with pytest.raises(ValueError, match="disconnected"):
        integration.configure(path, state, catalog("provider/model"))
    assert path.read_bytes() == before and not state.exists()


def test_status_recognizes_existing_connection_and_external_provider_changes(
    native_files,
):
    path, state = native_files
    before = path.read_bytes()
    assert integration.status(path, state)["connected"] is False
    assert integration.status(path, state)["can_disconnect"] is False
    assert path.read_bytes() == before and not state.exists()
    integration.configure(path, state, catalog("provider/model"), connect=True)
    before = path.read_bytes(), state.read_bytes()
    result = integration.status(path, state)
    assert result["connected"] is True
    assert result["can_disconnect"] is True
    assert result["paths"] == {
        "settings": str(path.resolve()),
        "connection": str(state.resolve()),
    }
    assert (path.read_bytes(), state.read_bytes()) == before
    write(path, {**read(path), "modelProvider": "personal"})
    assert integration.status(path, state)["connected"] is False
    assert integration.status(path, state)["can_disconnect"] is True
    assert state.exists()


def test_status_waits_for_native_write_before_reporting_connection(
    native_files, monkeypatch
):
    path, state = native_files
    entered, reading, release = threading.Event(), threading.Event(), threading.Event()
    write_native = integration._write_native

    def held(*args):
        entered.set()
        assert release.wait(5)
        return write_native(*args)

    def inspect():
        reading.set()
        return integration.status(path, state)

    monkeypatch.setattr(integration, "_write_native", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(
            integration.configure, path, state, catalog("provider/model"), connect=True
        )
        try:
            assert entered.wait(5)
            assert state.exists()
            reader = pool.submit(inspect)
            assert reading.wait(5)
            assert not reader.done()
        finally:
            release.set()
        writer.result(timeout=5)
        assert reader.result(timeout=5)["connected"] is True


def test_status_does_not_hide_record_access_errors(native_files, monkeypatch):
    path, state = native_files
    original_stat = Path.stat

    def denied(self, *args, **kwargs):
        if self == state:
            raise PermissionError("blocked")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(PermissionError):
        integration.status(path, state)


def test_preexisting_custom_model_collision_is_not_claimed(native_files):
    path, state = native_files
    before = path.read_bytes()
    with pytest.raises(ValueError, match="personal"):
        integration.configure(path, state, catalog("personal"), connect=True)
    assert path.read_bytes() == before and not state.exists()


def test_disconnect_preserves_externally_changed_provider_and_model(native_files):
    path, state = native_files
    integration.configure(path, state, catalog("provider/model"), connect=True)
    changed = read(path)
    changed["modelProvider"] = ""
    changed["customModelsConfig"]["customModels"]["provider/model"]["modelName"] = (
        "user-choice"
    )
    write(path, changed)
    with pytest.raises(ValueError, match="disconnected"):
        integration.configure(path, state, catalog("provider/model"))
    assert read(path) == changed
    integration.disconnect(path, state)
    assert read(path) == changed


def test_refresh_refuses_to_replace_a_user_modified_owned_model(native_files):
    path, state = native_files
    integration.configure(path, state, catalog("provider/model"), connect=True)
    changed = read(path)
    changed["customModelsConfig"]["customModels"]["provider/model"]["label"] = (
        "Personal"
    )
    write(path, changed)
    before = state.read_bytes()
    with pytest.raises(ValueError, match="provider/model"):
        integration.configure(path, state, catalog("provider/model"))
    assert read(path) == changed and state.read_bytes() == before
    integration.disconnect(path, state)
    assert read(path)["customModelsConfig"] == changed["customModelsConfig"]


@pytest.mark.parametrize(
    "source",
    [
        "{broken",
        '{"modelProvider":"",}',
        '\ufeff{"modelProvider":""}',
        '{"modelProvider":"", "modelProvider":"gemini"}',
        '{"customModelsConfig": []}',
        '{"customModelsConfig":{"customModels":[]}}',
        '{"unknown": NaN}',
        "[]",
    ],
)
def test_invalid_native_settings_are_not_replaced(native_files, source):
    path, state = native_files
    path.write_text(source, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        integration.configure(path, state, catalog("provider/model"), connect=True)
    assert path.read_bytes() == before and not state.exists()


def test_failed_native_write_retains_recoverable_ownership(native_files, monkeypatch):
    path, state = native_files
    original = read(path)
    integration.configure(path, state, catalog("provider/old"), connect=True)
    writer = integration.atomic_write_text

    def fail_native(target, content, **kwargs):
        if target == path:
            raise PermissionError("native file is locked")
        writer(target, content, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(integration, "atomic_write_text", fail_native)
        with pytest.raises(PermissionError):
            integration.configure(path, state, catalog("provider/new"))
    assert "provider/old" in read(path)["customModelsConfig"]["customModels"]
    integration.configure(path, state, catalog("provider/new"))
    integration.disconnect(path, state)
    assert read(path) == original


def test_initial_failed_native_write_can_be_disconnected(native_files, monkeypatch):
    path, state = native_files
    before = path.read_bytes()
    writer = integration.atomic_write_text

    def fail_native(target, content, **kwargs):
        if target == path:
            raise PermissionError("native file is locked")
        writer(target, content, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(integration, "atomic_write_text", fail_native)
        with pytest.raises(PermissionError):
            integration.configure(path, state, catalog("provider/model"), connect=True)
    assert path.read_bytes() == before and state.exists()
    integration.disconnect(path, state)
    assert path.read_bytes() == before and not state.exists()


@pytest.mark.parametrize("bad_state", ["{broken", "{}", '{"model_keys": [3]}'])
def test_malformed_ownership_record_blocks_mutation(native_files, bad_state):
    path, state = native_files
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(bad_state)
    before = path.read_bytes(), state.read_bytes()
    with pytest.raises(ValueError):
        integration.configure(path, state, catalog("provider/model"), connect=True)
    with pytest.raises(ValueError):
        integration.disconnect(path, state)
    assert (path.read_bytes(), state.read_bytes()) == before


def test_concurrent_connections_keep_the_original_provider_mode(native_files):
    path, state = native_files
    original = read(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                integration.configure, path, state, catalog(model), connect=True
            )
            for model in ("provider/one", "provider/two")
        ]
        for future in futures:
            future.result(timeout=5)
    keys = set(read(path)["customModelsConfig"]["customModels"])
    assert keys in ({"personal", "provider/one"}, {"personal", "provider/two"})
    integration.disconnect(path, state)
    assert read(path) == original
