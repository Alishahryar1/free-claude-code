"""Tests for OpenRouter provider-routing preference configuration."""

import json

import pytest
from pydantic import ValidationError

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import prepare_admin_update
from free_claude_code.config.admin.values import normalize_for_env
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.open_router_preferences import (
    OpenRouterProviderPreferences,
)
from free_claude_code.config.settings import Settings

PREFERENCES_JSON = json.dumps({"only": ["DeepInfra"], "allow_fallbacks": False})


def test_settings_parse_json_preferences():
    settings = Settings.model_validate(
        {
            "OPENROUTER_API_KEY": "key",
            "OPENROUTER_PROVIDER_PREFERENCES": json.dumps(
                {
                    "only": ["DeepInfra"],
                    "quantizations": ["fp8"],
                    "allow_fallbacks": False,
                }
            ),
        }
    )

    preferences = settings.open_router_provider_preferences

    assert preferences is not None
    assert preferences.to_request_field() == {
        "only": ["DeepInfra"],
        "quantizations": ["fp8"],
        "allow_fallbacks": False,
    }


def test_settings_blank_preferences_are_unset():
    assert Settings().open_router_provider_preferences is None
    assert (
        Settings.model_validate(
            {"OPENROUTER_PROVIDER_PREFERENCES": "  "}
        ).open_router_provider_preferences
        is None
    )


def test_settings_reject_malformed_json():
    with pytest.raises(ValidationError):
        Settings.model_validate({"OPENROUTER_PROVIDER_PREFERENCES": "{not json"})


def test_preferences_reject_unknown_fields():
    with pytest.raises(ValidationError):
        OpenRouterProviderPreferences.model_validate({"nonsense": True})


def test_preferences_reject_empty_provider_lists():
    with pytest.raises(ValidationError):
        OpenRouterProviderPreferences.model_validate({"only": []})


def test_preferences_reject_unknown_quantization():
    with pytest.raises(ValidationError):
        OpenRouterProviderPreferences.model_validate({"quantizations": ["fp9"]})


def test_preferences_reject_empty_object():
    with pytest.raises(ValidationError):
        OpenRouterProviderPreferences.model_validate({})


def test_sort_object_and_price_limits_serialize_for_the_request():
    preferences = OpenRouterProviderPreferences.model_validate(
        {
            "sort": {"by": "price", "partition": "none"},
            "max_price": {"prompt": 0.25, "completion": 1.25},
            "preferred_max_latency": {"p90": 5.0},
        }
    )

    assert preferences.to_request_field() == {
        "sort": {"by": "price", "partition": "none"},
        "max_price": {"prompt": 0.25, "completion": 1.25},
        "preferred_max_latency": {"p90": 5.0},
    }


def test_sort_string_and_plain_thresholds_serialize_for_the_request():
    preferences = OpenRouterProviderPreferences.model_validate(
        {"sort": "price", "preferred_min_throughput": 100, "data_collection": "deny"}
    )

    assert preferences.to_request_field() == {
        "sort": "price",
        "preferred_min_throughput": 100,
        "data_collection": "deny",
    }


def test_percentile_thresholds_require_one_cutoff():
    with pytest.raises(ValidationError):
        OpenRouterProviderPreferences.model_validate({"preferred_min_throughput": {}})


def test_max_price_requires_one_limit():
    with pytest.raises(ValidationError):
        OpenRouterProviderPreferences.model_validate({"max_price": {}})


def test_admin_exposes_the_preferences_textarea():
    entry = FIELD_BY_KEY["OPENROUTER_PROVIDER_PREFERENCES"]

    assert entry.field_type == "textarea"
    assert entry.section_id == "providers"
    assert entry.settings_attr == "open_router_provider_preferences"
    assert entry.restart_required is True
    assert "allow_fallbacks" in entry.description


def test_admin_displays_preferences_as_json():
    preferences = OpenRouterProviderPreferences.model_validate(
        {"only": ["DeepInfra"], "allow_fallbacks": False}
    )

    displayed = normalize_for_env(preferences, key="OPENROUTER_PROVIDER_PREFERENCES")
    assert displayed is not None
    assert json.loads(displayed) == {"only": ["DeepInfra"], "allow_fallbacks": False}
    assert normalize_for_env(None, key="OPENROUTER_PROVIDER_PREFERENCES") is None


def test_admin_update_persists_and_reloads_preferences():
    store = ManagedConfigStore()
    store.initialize()
    snapshot = store.read()

    prepared = prepare_admin_update(
        {"OPENROUTER_PROVIDER_PREFERENCES": PREFERENCES_JSON},
        snapshot,
        snapshot.settings,
        revision_key=b"test-revision-key",
    )

    assert prepared.valid, prepared.errors
    assert prepared.changed_keys == ("OPENROUTER_PROVIDER_PREFERENCES",)
    assert prepared.settings is not None
    assert prepared.settings.open_router_provider_preferences is not None

    store.commit(prepared.target_values)

    reloaded = store.read().settings
    assert reloaded.open_router_provider_preferences is not None
    assert reloaded.open_router_provider_preferences.to_request_field() == {
        "only": ["DeepInfra"],
        "allow_fallbacks": False,
    }


def test_settings_decode_quantizations_from_comma_list():
    settings = Settings.model_validate({"OPENROUTER_QUANTIZATIONS": "fp8, bf16"})

    assert settings.open_router_quantizations == ("fp8", "bf16")


def test_settings_blank_quantizations_mean_any():
    assert Settings().open_router_quantizations is None
    assert (
        Settings.model_validate(
            {"OPENROUTER_QUANTIZATIONS": ""}
        ).open_router_quantizations
        is None
    )
    assert (
        Settings.model_validate(
            {"OPENROUTER_QUANTIZATIONS": ", ,"}
        ).open_router_quantizations
        is None
    )


def test_settings_reject_unknown_quantization():
    with pytest.raises(ValidationError):
        Settings.model_validate({"OPENROUTER_QUANTIZATIONS": "fp9"})


def test_settings_decode_quantizations_from_list():
    settings = Settings.model_validate({"OPENROUTER_QUANTIZATIONS": ["fp8"]})

    assert settings.open_router_quantizations == ("fp8",)


def test_settings_cheapest_provider_defaults_off():
    assert Settings().open_router_use_cheapest_provider is False
    assert (
        Settings.model_validate(
            {"OPENROUTER_USE_CHEAPEST_PROVIDER": "true"}
        ).open_router_use_cheapest_provider
        is True
    )


def test_admin_exposes_the_cheapest_provider_checkbox():
    entry = FIELD_BY_KEY["OPENROUTER_USE_CHEAPEST_PROVIDER"]

    assert entry.field_type == "boolean"
    assert entry.section_id == "providers"
    assert entry.settings_attr == "open_router_use_cheapest_provider"
    assert entry.restart_required is True
    assert "cheapest" in entry.description


def test_admin_exposes_the_quantizations_multiselect():
    entry = FIELD_BY_KEY["OPENROUTER_QUANTIZATIONS"]

    assert entry.field_type == "multiselect"
    assert entry.section_id == "providers"
    assert entry.settings_attr == "open_router_quantizations"
    assert entry.restart_required is True
    assert {getattr(option, "value", option) for option in entry.options} >= {
        "fp8",
        "bf16",
        "unknown",
    }
    assert "any quantization" in entry.description


def test_admin_displays_quantizations_as_comma_list():
    settings = Settings.model_validate({"OPENROUTER_QUANTIZATIONS": "fp8,bf16"})

    displayed = normalize_for_env(
        settings.open_router_quantizations, key="OPENROUTER_QUANTIZATIONS"
    )

    assert displayed == "fp8,bf16"
    assert normalize_for_env(None, key="OPENROUTER_QUANTIZATIONS") is None


def test_admin_update_persists_and_reloads_cheapest_settings():
    store = ManagedConfigStore()
    store.initialize()
    snapshot = store.read()

    prepared = prepare_admin_update(
        {
            "OPENROUTER_USE_CHEAPEST_PROVIDER": "true",
            "OPENROUTER_QUANTIZATIONS": "fp8,bf16",
        },
        snapshot,
        snapshot.settings,
        revision_key=b"test-revision-key",
    )

    assert prepared.valid, prepared.errors
    assert set(prepared.changed_keys) == {
        "OPENROUTER_USE_CHEAPEST_PROVIDER",
        "OPENROUTER_QUANTIZATIONS",
    }

    store.commit(prepared.target_values)

    reloaded = store.read().settings
    assert reloaded.open_router_use_cheapest_provider is True
    assert reloaded.open_router_quantizations == ("fp8", "bf16")


def test_admin_update_clears_quantizations_when_unselected():
    store = ManagedConfigStore()
    store.initialize()
    snapshot = store.read()
    prepared = prepare_admin_update(
        {"OPENROUTER_QUANTIZATIONS": "fp8"},
        snapshot,
        snapshot.settings,
        revision_key=b"test-revision-key",
    )
    assert prepared.valid, prepared.errors
    store.commit(prepared.target_values)

    cleared = prepare_admin_update(
        {"OPENROUTER_QUANTIZATIONS": None},
        store.read(),
        store.read().settings,
        revision_key=b"test-revision-key",
    )

    assert cleared.valid, cleared.errors
    store.commit(cleared.target_values)

    assert store.read().settings.open_router_quantizations is None
