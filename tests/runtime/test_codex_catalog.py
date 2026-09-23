import json
from pathlib import Path
from unittest.mock import patch

import pytest

from free_claude_code.application.code_sessions.models import CodeValidationError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import (
    ModelCatalogSnapshot,
    RequestRuntimeLease,
    RequestRuntimePort,
)
from free_claude_code.cli.launchers.catalog_http import catalog_models_from_response
from free_claude_code.config.settings import Settings
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.codex_model_catalog import build_codex_model_catalog
from free_claude_code.runtime.codex_app_server import CodexHarnessFactory
from free_claude_code.runtime.codex_catalog import (
    CodexModelCatalogPublisher,
    write_codex_model_catalog,
)
from tests.harnesses.test_codex_model_catalog import _models_payload


class FakeRequestRuntime(RequestRuntimePort):
    def __init__(
        self,
        *,
        settings: Settings,
        cached_infos: tuple[ProviderModelInfo, ...] = (),
    ) -> None:
        self._settings = settings
        self._cached_infos = cached_infos

    async def acquire(
        self, *, include_model_infos: bool = False
    ) -> RequestRuntimeLease:
        del include_model_infos
        raise AssertionError("Catalog publication must not acquire a provider lease.")

    async def wait_for_catalog(self) -> ModelCatalogSnapshot:
        return ModelCatalogSnapshot(self._settings, self._cached_infos)

    def current_settings(self) -> Settings:
        return self._settings

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        del provider_id, model_id
        return None

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return self._cached_infos


def _runtime() -> FakeRequestRuntime:
    settings = Settings().model_copy(update={"model": "nvidia_nim/configured"})
    return FakeRequestRuntime(
        settings=settings,
        cached_infos=(
            ProviderModelInfo("open_router/discovered", context_window_tokens=100_000),
        ),
    )


def _catalog_slugs(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [model["slug"] for model in payload["models"]]


@pytest.mark.asyncio
async def test_known_vision_survives_publication_and_browser_preparation(
    tmp_path: Path,
):
    runtime = FakeRequestRuntime(
        settings=Settings().model_copy(update={"model": "nvidia_nim/configured"}),
        cached_infos=(
            ProviderModelInfo(
                "open_router/vision",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131072,
            ),
        ),
    )
    path = tmp_path / "catalog.json"
    CodexModelCatalogPublisher(path).publish(runtime)
    entries = {row["slug"]: row for row in json.loads(path.read_text())["models"]}
    assert entries["open_router/vision"]["input_modalities"] == ["text", "image"]
    selected = await CodexHarnessFactory(runtime, binary="codex").prepare(
        "open_router/vision", None, "config"
    )
    prepared = json.loads(json.dumps(build_codex_model_catalog(selected.models)))
    assert next(
        row for row in prepared["models"] if row["slug"] == "open_router/vision"
    )["input_modalities"] == ["text", "image"]


def test_publisher_projects_the_application_catalog_without_compatibility_ids(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    publisher = CodexModelCatalogPublisher(catalog_path)

    publisher.publish(_runtime())

    assert _catalog_slugs(catalog_path) == [
        "nvidia_nim/configured",
        "open_router/discovered",
    ]


def test_empty_projection_preserves_existing_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    catalog_path.write_text("last known good\n", encoding="utf-8")
    publisher = CodexModelCatalogPublisher(catalog_path)

    with (
        patch(
            "free_claude_code.runtime.codex_catalog.build_codex_model_catalog",
            return_value={"models": []},
        ),
        pytest.raises(ValueError, match="no routable models"),
    ):
        publisher.publish(_runtime())

    assert catalog_path.read_text(encoding="utf-8") == "last known good\n"


@pytest.mark.asyncio
async def test_code_picker_and_native_selection_use_the_same_advertised_efforts():
    factory = CodexHarnessFactory(_runtime(), binary="codex")
    advertised = factory.catalog()
    assert advertised.default_model == "nvidia_nim/configured"
    model = advertised.models[1]
    assert advertised.models[0].context_window_tokens is None
    assert model.context_window_tokens == 100_000
    assert model.reasoning_efforts == ("off", "low", "medium", "high", "xhigh", "max")
    selected = await factory.prepare(model.id, None, "config")
    assert selected.model == model.id
    assert selected.reasoning_effort == model.default_reasoning_effort == "medium"
    for effort in model.reasoning_efforts:
        assert (
            await factory.prepare(model.id, effort, "config")
        ).reasoning_effort == effort
    with pytest.raises(CodeValidationError, match="effort"):
        (await factory.prepare(model.id, "unsupported", "config"))
    with pytest.raises(CodeValidationError, match="model"):
        (await factory.prepare("missing/model", None, "config"))


@pytest.mark.asyncio
async def test_non_reasoning_model_off_selection_is_available():
    runtime = FakeRequestRuntime(
        settings=Settings().model_copy(update={"model": "nvidia_nim/configured"}),
        cached_infos=(
            ProviderModelInfo("open_router/text-only", supports_thinking=False),
        ),
    )
    factory = CodexHarnessFactory(runtime, binary="codex")
    model = next(
        entry
        for entry in factory.catalog().models
        if entry.id == "open_router/text-only"
    )
    assert model.reasoning_efforts == ("off",)
    assert model.default_reasoning_effort == "off"
    assert (await factory.prepare(model.id, "off", "config")).reasoning_effort == "off"


def test_catalog_writer_skips_identical_content_and_replaces_changes(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    first = build_codex_model_catalog(
        catalog_models_from_response(_models_payload("nvidia_nim/first"))
    )
    second = build_codex_model_catalog(
        catalog_models_from_response(_models_payload("nvidia_nim/second"))
    )

    assert write_codex_model_catalog(catalog_path, first) is True
    assert write_codex_model_catalog(catalog_path, first) is False
    assert list(tmp_path.glob(".codex-model-catalog.json.*.tmp")) == []

    assert write_codex_model_catalog(catalog_path, second) is True
    assert json.loads(catalog_path.read_text(encoding="utf-8")) == second
    assert list(tmp_path.glob(".codex-model-catalog.json.*.tmp")) == []


def test_catalog_writer_cleans_temporary_file_after_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    catalog_path.write_text("previous\n", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> Path:
        raise PermissionError("destination is locked")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(PermissionError, match="locked"):
        write_codex_model_catalog(
            catalog_path,
            build_codex_model_catalog(
                catalog_models_from_response(_models_payload("nvidia_nim/replacement"))
            ),
        )

    assert catalog_path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".codex-model-catalog.json.*.tmp")) == []


@pytest.mark.asyncio
async def test_catalog_refresh_preserves_selection_and_behavior_fingerprint():
    runtime = FakeRequestRuntime(
        settings=Settings().model_copy(update={"model": "open_router/Zulu"}),
        cached_infos=(
            ProviderModelInfo("open_router/Zulu", context_window_tokens=32000),
        ),
    )
    factory = CodexHarnessFactory(runtime, binary="codex")
    before = await factory.prepare("open_router/Zulu", "high", "config")
    runtime._cached_infos = (
        ProviderModelInfo("deepseek/alpha"),
        *runtime._cached_infos,
    )
    after = await factory.prepare("open_router/Zulu", "high", "config")
    assert after.model == before.model
    assert after.reasoning_effort == before.reasoning_effort
    assert after.configuration_key == before.configuration_key
    assert factory.catalog().default_model == "open_router/Zulu"
    assert factory.catalog().models[0].id == "deepseek/alpha"
    runtime._cached_infos = (
        ProviderModelInfo("open_router/Zulu", context_window_tokens=64000),
    )
    changed = await factory.prepare("open_router/Zulu", "high", "config")
    assert changed.configuration_key != before.configuration_key


@pytest.mark.parametrize(
    "modalities", [frozenset(), frozenset({ModelInputModality.IMAGE})]
)
def test_non_text_capabilities_retain_the_native_text_fallback(modalities):
    from free_claude_code.application.model_catalog import read_model_catalog

    snapshot = ModelCatalogSnapshot(
        Settings().model_copy(update={"model": "open_router/model"}),
        (ProviderModelInfo("open_router/model", input_modalities=modalities),),
    )
    catalog = read_model_catalog(snapshot)
    assert catalog.models[0].input_modalities == modalities
    payload = json.loads(json.dumps(build_codex_model_catalog(catalog.models)))
    assert payload["models"][0]["input_modalities"] == ["text"]
