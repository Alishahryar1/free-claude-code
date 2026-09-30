from dataclasses import replace

from free_claude_code.application.model_catalog import (
    CatalogModel,
    catalog_wire_slug_for_ref,
    read_model_catalog,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ModelCatalogSnapshot
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.settings import Settings


def test_custom_display_names_sort_as_labels_without_changing_model_identity():
    names = ["Team/West", "/Gateway", "Team", "z"]
    definitions = tuple(
        CustomProviderDefinition(
            provider_id=f"custom_{index:032x}",
            display_name=name,
            base_url="https://gateway.example/v1",
        )
        for index, name in enumerate(names)
    )
    settings = Settings(
        custom_providers=definitions, model=f"{definitions[0].provider_id}/Z"
    )
    infos = tuple(
        ProviderModelInfo(f"{definition.provider_id}/{model}")
        for definition in definitions
        for model in ("Z", "org/m")
    )
    snapshot = ModelCatalogSnapshot(settings, infos)
    catalog = read_model_catalog(snapshot)
    assert [model.display_name for model in catalog.models] == [
        f"{name}/{model}"
        for name in ("/Gateway", "Team", "Team/West", "z")
        for model in ("org/m", "Z")
    ]
    assert {model.provider_model_ref for model in catalog.models} == {
        info.model_id for info in infos
    }
    assert catalog.default_model_id == settings.model
    assert (
        read_model_catalog(replace(snapshot, model_infos=tuple(reversed(infos))))
        == catalog
    )


def test_catalog_sorts_components_and_preserves_exact_identity_and_default():
    settings = Settings.model_construct(
        model="open_router/Zulu",
        model_fable=None,
        model_opus=None,
        model_sonnet=None,
        model_haiku=None,
        model_fallbacks=("wafer/z", "deepseek/a"),
    )
    infos = (
        ProviderModelInfo("alpha-2/model"),
        ProviderModelInfo("alpha/z"),
        ProviderModelInfo("alpha/ß"),
        ProviderModelInfo("alpha/ss"),
        ProviderModelInfo("alpha/Apple"),
        ProviderModelInfo("alpha/apple"),
        ProviderModelInfo("alpha/vendor/nested"),
        ProviderModelInfo(
            "open_router/Zulu", supports_thinking=False, context_window_tokens=32000
        ),
    )
    snapshot = ModelCatalogSnapshot(settings, infos)
    catalog = read_model_catalog(snapshot)
    assert [model.provider_model_ref for model in catalog.models] == [
        "alpha/Apple",
        "alpha/apple",
        "alpha/ss",
        "alpha/ß",
        "alpha/vendor/nested",
        "alpha/z",
        "alpha-2/model",
        "deepseek/a",
        "open_router/Zulu",
        "wafer/z",
    ]
    assert (
        read_model_catalog(replace(snapshot, model_infos=tuple(reversed(infos))))
        == catalog
    )
    assert catalog.default_model_id == "claude-3-freecc-no-thinking/open_router/Zulu"
    selected = next(
        model for model in catalog.models if model.wire_slug == catalog.default_model_id
    )
    assert selected.context_window_tokens == 32000
    assert settings.model_fallbacks == ("wafer/z", "deepseek/a")


def test_catalog_retains_configured_models_and_enriches_each_reference_once():
    settings = Settings.model_construct(
        model="open_router/shared",
        model_opus="open_router/shared",
        model_fable=None,
        model_sonnet=None,
        model_haiku="deepseek/undiscovered",
        model_fallbacks=("open_router/shared",),
    )
    info = ProviderModelInfo("open_router/shared", context_window_tokens=65536)
    catalog = read_model_catalog(ModelCatalogSnapshot(settings, (info,)))
    assert [model.provider_model_ref for model in catalog.models] == [
        "deepseek/undiscovered",
        "open_router/shared",
    ]
    assert catalog.models[0].supports_reasoning is None
    assert catalog.models[0].context_window_tokens is None
    assert catalog.models[1].context_window_tokens == 65536


def _client_model(wire_slug: str, provider_model_ref: str) -> CatalogModel:
    return CatalogModel(
        wire_slug=wire_slug,
        provider_model_ref=provider_model_ref,
        display_name=provider_model_ref,
        supports_reasoning=wire_slug == provider_model_ref,
    )


def test_catalog_wire_slug_prefers_the_advertised_no_thinking_slug() -> None:
    models = (
        _client_model(
            "claude-3-freecc-no-thinking/open_router/vendor/chat-model",
            "open_router/vendor/chat-model",
        ),
    )

    assert (
        catalog_wire_slug_for_ref(models, "open_router/vendor/chat-model")
        == "claude-3-freecc-no-thinking/open_router/vendor/chat-model"
    )


def test_catalog_wire_slug_keeps_a_directly_advertised_ref() -> None:
    models = (
        _client_model("open_router/vendor/chat-model", "open_router/vendor/chat-model"),
    )

    assert (
        catalog_wire_slug_for_ref(models, "open_router/vendor/chat-model")
        == "open_router/vendor/chat-model"
    )


def test_catalog_wire_slug_falls_back_when_the_catalog_omits_the_ref() -> None:
    models = (_client_model("open_router/vendor/other", "open_router/vendor/other"),)

    assert (
        catalog_wire_slug_for_ref(models, "open_router/vendor/chat-model")
        == "open_router/vendor/chat-model"
    )
    assert catalog_wire_slug_for_ref((), "open_router/vendor/chat-model") == (
        "open_router/vendor/chat-model"
    )


def test_catalog_wire_slug_passes_through_an_unset_model() -> None:
    assert catalog_wire_slug_for_ref((), None) is None
    assert catalog_wire_slug_for_ref((), "") == ""
