"""Fallback key numbers refer to each provider's effective numbered list."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from free_claude_code.config.admin.persistence import prepare_admin_update
from free_claude_code.config.admin.values import load_config_response
from free_claude_code.config.loader import (
    ManagedConfigSnapshot,
    compose_settings_snapshot,
)
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.settings import Settings

KEYS = ["synthetic-first", "synthetic-second", "synthetic-third"]
REVISION_KEY = b"test-fallback-revision"
CUSTOM_ID = "custom_" + "a" * 32


def snapshot(managed, process=None):
    process = process or {}
    effective = compose_settings_snapshot(managed, process)
    return ManagedConfigSnapshot(
        effective.settings, effective.sources, managed, process, Path("unused.env")
    )


def admin_field(state, key):
    fields = load_config_response(state, revision_key=REVISION_KEY)["fields"]
    assert isinstance(fields, list)
    for field in fields:
        assert isinstance(field, dict)
        if field["key"] == key:
            return field
    raise AssertionError(f"Missing Admin field: {key}")


@pytest.mark.parametrize(
    "value",
    ["groq/one", ["groq/one"], [{"model": "groq/one", "reasoning": "high"}]],
)
def test_existing_fallback_formats_use_key_one(value):
    settings = Settings(MODEL_FALLBACKS=value)
    assert settings.model_fallbacks is not None
    assert settings.model_fallbacks[0].api_key_number == 1


def test_explicit_key_number_and_effort_survive_json_input():
    settings = Settings.model_validate(
        {
            "GROQ_API_KEY": KEYS,
            "MODEL_FALLBACKS": '[{"model":"groq/one","reasoning":"high","api_key_number":3}]',
        }
    )
    assert settings.model_fallbacks is not None
    assert settings.model_fallbacks[0].model_dump(mode="json") == {
        "model": "groq/one",
        "reasoning": "high",
        "api_key_number": 3,
    }


def test_same_model_can_use_distinct_keys_without_duplicate_discovery_refs():
    settings = Settings(
        MODEL="groq/one",
        GROQ_API_KEY=KEYS,
        MODEL_FALLBACKS=[
            {"model": "groq/one", "api_key_number": 1},
            {"model": "groq/one", "api_key_number": 2},
            {"model": "groq/one", "api_key_number": 3},
        ],
    )
    assert settings.model_fallbacks is not None
    assert [entry.api_key_number for entry in settings.model_fallbacks] == [1, 2, 3]
    assert [ref.model_ref for ref in configured_chat_model_refs(settings)] == [
        "groq/one"
    ]


def test_different_efforts_do_not_allow_duplicate_model_key_pairs():
    with pytest.raises(ValidationError, match="duplicate"):
        Settings(
            GROQ_API_KEY=KEYS,
            MODEL_FALLBACKS=[
                {"model": "groq/one", "reasoning": "off", "api_key_number": 2},
                {"model": "groq/one", "reasoning": "high", "api_key_number": 2},
            ],
        )


@pytest.mark.parametrize("number", [0, -1, True, 1.5, "2"])
def test_key_number_requires_a_positive_integer(number):
    with pytest.raises(ValidationError) as rejected:
        Settings(
            GROQ_API_KEY=KEYS,
            MODEL_FALLBACKS=[{"model": "groq/one", "api_key_number": number}],
        )
    errors = rejected.value.errors(include_input=False)
    assert errors[0]["loc"] == ("MODEL_FALLBACKS", 0, "api_key_number")
    assert errors[0]["type"] in {"greater_than_equal", "int_type"}


def test_unavailable_key_is_rejected_without_disclosing_keys():
    with pytest.raises(ValidationError, match="Key 4") as rejected:
        Settings(
            GROQ_API_KEY=KEYS,
            MODEL_FALLBACKS=[{"model": "groq/one", "api_key_number": 4}],
        )
    assert all(key not in str(rejected.value) for key in KEYS)


@pytest.mark.parametrize("provider", ["openai", "github_copilot", "vertex", "ollama"])
def test_other_authentication_has_only_the_default_selection(provider):
    default = Settings(
        MODEL_FALLBACKS=[{"model": f"{provider}/one", "api_key_number": 1}]
    )
    assert default.model_fallbacks is not None
    assert default.model_fallbacks[0].api_key_number == 1
    with pytest.raises(ValidationError, match="API key list"):
        Settings(MODEL_FALLBACKS=[{"model": f"{provider}/one", "api_key_number": 2}])


@pytest.mark.parametrize("provider", ["zai", "zai_api", "opencode_zen", "opencode_go"])
def test_shared_credential_owners_supply_the_same_numbered_keys(provider):
    settings = Settings(
        ZAI_API_KEY=KEYS,
        OPENCODE_API_KEY=KEYS,
        MODEL_FALLBACKS=[{"model": f"{provider}/one", "api_key_number": 3}],
    )
    assert settings.model_fallbacks is not None
    assert settings.model_fallbacks[0].api_key_number == 3


def test_custom_provider_key_selection_and_unauthenticated_default():
    definition = {
        "provider_id": CUSTOM_ID,
        "display_name": "Gateway",
        "base_url": "https://gateway.example/v1",
        "api_keys": KEYS,
    }
    selected = Settings(
        FCC_CUSTOM_PROVIDERS=[definition],
        MODEL_FALLBACKS=[{"model": f"{CUSTOM_ID}/one", "api_key_number": 2}],
    )
    assert selected.model_fallbacks is not None
    assert selected.model_fallbacks[0].api_key_number == 2
    anonymous = Settings(
        FCC_CUSTOM_PROVIDERS=[definition | {"api_keys": []}],
        MODEL_FALLBACKS=[{"model": f"{CUSTOM_ID}/one"}],
    )
    assert anonymous.model_fallbacks is not None
    assert anonymous.model_fallbacks[0].api_key_number == 1
    with pytest.raises(ValidationError, match="Key 2"):
        Settings(
            FCC_CUSTOM_PROVIDERS=[definition | {"api_keys": []}],
            MODEL_FALLBACKS=[{"model": f"{CUSTOM_ID}/one", "api_key_number": 2}],
        )


def test_admin_save_reload_preserves_the_selected_number_and_clean_state():
    entries = [{"model": "groq/one", "reasoning": "low", "api_key_number": 3}]
    original = snapshot({"GROQ_API_KEY": json.dumps(KEYS)})
    saved = prepare_admin_update(
        {"MODEL_FALLBACKS": entries},
        original,
        original.settings,
        revision_key=REVISION_KEY,
    )
    assert saved.valid, saved.errors
    assert json.loads(saved.target_values["MODEL_FALLBACKS"]) == entries
    loaded = snapshot(saved.target_values)
    field = admin_field(loaded, "MODEL_FALLBACKS")
    value = field["value"]
    assert isinstance(value, str)
    assert json.loads(value) == entries
    unchanged = prepare_admin_update(
        {"MODEL_FALLBACKS": field["value"]},
        loaded,
        loaded.settings,
        revision_key=REVISION_KEY,
    )
    assert unchanged.valid and unchanged.changed_keys == ()
    assert unchanged.pending_fields == ()


def test_process_fallbacks_select_from_process_owned_keys():
    state = snapshot(
        {
            "GROQ_API_KEY": json.dumps(KEYS),
            "MODEL_FALLBACKS": '[{"model":"groq/one","api_key_number":3}]',
        },
        {
            "GROQ_API_KEY": '["process-first","process-second"]',
            "MODEL_FALLBACKS": '[{"model":"groq/one","api_key_number":2}]',
        },
    )
    assert state.settings.model_fallbacks is not None
    assert state.settings.model_fallbacks[0].api_key_number == 2
    field = admin_field(state, "MODEL_FALLBACKS")
    assert field["locked"] is True


def test_removing_a_selected_key_fails_the_prospective_update():
    original = snapshot(
        {
            "GROQ_API_KEY": json.dumps(KEYS),
            "MODEL_FALLBACKS": '[{"model":"groq/one","api_key_number":3}]',
        }
    )
    field = admin_field(original, "GROQ_API_KEY")
    value = field["value"]
    assert isinstance(value, dict)
    prepared = prepare_admin_update(
        {
            "GROQ_API_KEY": {
                "revision": value["revision"],
                "entries": [{"retain": 0}, {"retain": 1}],
            }
        },
        original,
        original.settings,
        revision_key=REVISION_KEY,
    )
    assert not prepared.valid
    assert "Key 3" in " ".join(prepared.errors)
    assert len(original.settings.groq_api_keys) == 3
