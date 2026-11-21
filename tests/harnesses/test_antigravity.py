"""Antigravity session catalog and ownership-safe retirement of old connections."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses import antigravity
from tests.harnesses.antigravity_support import write_previous_connection
from tests.wire_payload import decode_object, object_payload


@pytest.fixture
def native_files(tmp_path: Path):
    path, state = (
        tmp_path / "native/settings.json",
        tmp_path / "fcc/antigravity-integration.json",
    )
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "modelProvider": "",
                "model": "my-choice",
                "colorScheme": "dark",
                "customModelsConfig": {
                    "keep": True,
                    "customModels": {"personal": {"modelName": "own"}},
                },
                "permissions": {"allow": ["read_file"]},
            }
        )
    )
    return path, state


def test_session_catalog_preserves_native_model_contract() -> None:
    model = CatalogModel(
        "provider/model:variant",
        "provider/model:variant",
        "My model",
        False,
        frozenset({ModelInputModality.TEXT}),
        context_window_tokens=32000,
        max_output_tokens=4096,
    )
    body = antigravity.session_settings(ModelCatalog((model,), model.wire_slug))
    assert body == {
        "modelProvider": "gemini",
        "customModelsConfig": {
            "customModels": {
                "provider/model:variant": {
                    "modelName": "gemini-api://models/provider/model:variant",
                    "label": "FCC: My model",
                    "toolFormatterType": "TOOL_FORMATTER_TYPE_NONE",
                    "modelFeatures": {
                        "supportsImages": False,
                        "supportsVideo": False,
                        "supportsAudio": False,
                        "supportsPdf": False,
                        "supportsToolCalls": True,
                        "supportsThinking": False,
                    },
                }
            }
        },
    }


def test_unrecorded_settings_are_not_even_read(native_files: tuple[Path, Path]) -> None:
    path, state = native_files
    path.write_text("not JSON; native owns this")
    assert not antigravity.cleanup_previous_connection(path, state)
    assert path.read_text() == "not JSON; native owns this"
    assert not state.parent.exists()


@pytest.mark.parametrize(
    "previous", [{}, {"modelProvider": ""}, {"modelProvider": "gemini"}]
)
def test_cleanup_restores_only_previous_connection_and_preserves_history(
    native_files: tuple[Path, Path], previous: JsonObject
) -> None:
    path, state = native_files
    document = decode_object(path.read_text())
    document.pop("modelProvider")
    document.update(previous)
    path.write_text(json.dumps(document))
    original = write_previous_connection(path, state)
    history = path.with_name("history.db")
    history.write_bytes(b"existing native history")
    assert antigravity.cleanup_previous_connection(path, state)
    assert load_json_value(path.read_text()) == original
    assert history.read_bytes() == b"existing native history"
    assert not state.exists()
    assert not antigravity.cleanup_previous_connection(path, state)


@pytest.mark.parametrize(
    "field,value", [("modelName", "user-model"), ("label", "Personal")]
)
def test_cleanup_preserves_external_provider_and_modified_model(
    native_files: tuple[Path, Path], field: str, value: str
) -> None:
    path, state = native_files
    write_previous_connection(path, state)
    changed = decode_object(path.read_text())
    changed["modelProvider"] = "personal"
    object_payload(
        object_payload(object_payload(changed["customModelsConfig"])["customModels"])[
            "provider/model"
        ]
    )[field] = value
    path.write_text(json.dumps(changed))
    antigravity.cleanup_previous_connection(path, state)
    assert load_json_value(path.read_text()) == changed
    assert not state.exists()


@pytest.mark.parametrize("bad_state", ["{broken", "{}", '{"model_keys": [3]}'])
def test_invalid_record_is_preserved_without_native_mutation(
    native_files: tuple[Path, Path], bad_state: str
) -> None:
    path, state = native_files
    write_previous_connection(path, state)
    state.write_text(bad_state)
    before = path.read_bytes(), state.read_bytes()
    with pytest.raises(ValueError):
        antigravity.cleanup_previous_connection(path, state)
    assert (path.read_bytes(), state.read_bytes()) == before


def test_record_for_another_path_is_not_used(native_files: tuple[Path, Path]) -> None:
    path, state = native_files
    write_previous_connection(path, state)
    record = decode_object(state.read_text())
    record["settings_path"] = str(path.with_name("other.json"))
    state.write_text(json.dumps(record))
    before = path.read_bytes(), state.read_bytes()
    with pytest.raises(ValueError, match="connection record"):
        antigravity.cleanup_previous_connection(path, state)
    assert (path.read_bytes(), state.read_bytes()) == before


@pytest.mark.parametrize(
    "source",
    [
        "{broken",
        '{"modelProvider":"",}',
        '{"modelProvider":"", "modelProvider":"gemini"}',
        '{"customModelsConfig": []}',
        '{"customModelsConfig":{"customModels":[]}}',
        '{"unknown": NaN}',
        "[]",
    ],
)
def test_invalid_native_settings_are_not_replaced(
    native_files: tuple[Path, Path], source: str
) -> None:
    path, state = native_files
    write_previous_connection(path, state)
    path.write_text(source)
    before = path.read_bytes(), state.read_bytes()
    with pytest.raises(ValueError):
        antigravity.cleanup_previous_connection(path, state)
    assert (path.read_bytes(), state.read_bytes()) == before


def test_failed_native_write_keeps_record_for_retry(
    native_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, state = native_files
    original = write_previous_connection(path, state)
    before = path.read_bytes(), state.read_bytes()
    with monkeypatch.context() as patch:

        def denied(*args: object, **kwargs: object):
            raise PermissionError("native file locked")

        patch.setattr(antigravity, "atomic_write_text", denied)
        with pytest.raises(PermissionError):
            antigravity.cleanup_previous_connection(path, state)
    assert (path.read_bytes(), state.read_bytes()) == before
    antigravity.cleanup_previous_connection(path, state)
    assert json.loads(path.read_text()) == original and not state.exists()


def test_interrupted_record_removal_is_safe_to_retry(
    native_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, state = native_files
    original = write_previous_connection(path, state)
    unlink = Path.unlink
    with monkeypatch.context() as patch:

        def denied(target: Path, missing_ok: bool = False):
            if target == state:
                raise PermissionError("record locked")
            return unlink(target, missing_ok=missing_ok)

        patch.setattr(Path, "unlink", denied)
        with pytest.raises(PermissionError):
            antigravity.cleanup_previous_connection(path, state)
    assert json.loads(path.read_text()) == original and state.exists()
    antigravity.cleanup_previous_connection(path, state)
    assert json.loads(path.read_text()) == original and not state.exists()


def test_two_first_launches_safely_share_cleanup(
    native_files: tuple[Path, Path],
) -> None:
    path, state = native_files
    original = write_previous_connection(path, state)
    gate = Barrier(2)

    def cleanup():
        gate.wait(timeout=5)
        return antigravity.cleanup_previous_connection(path, state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(timeout=8)
            for future in [pool.submit(cleanup), pool.submit(cleanup)]
        ]
    assert sorted(results) == [False, True]
    assert json.loads(path.read_text()) == original and not state.exists()
