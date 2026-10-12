"""Fallback model/effort configuration and persistence contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from free_claude_code.config.admin.persistence import prepare_admin_update
from free_claude_code.config.admin.values import load_config_response
from free_claude_code.config.loader import (
    ConfigSource,
    ManagedConfigSnapshot,
    compose_settings_snapshot,
)
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings


def _snapshot(managed=None, process=None):
    managed, process = managed or {}, process or {}
    result = compose_settings_snapshot(managed, process)
    return ManagedConfigSnapshot(
        result.settings, result.sources, managed, process, Path("unused.env")
    )


def _fallback_field(state):
    fields = load_config_response(state, revision_key=b"test-revision-key")["fields"]
    assert isinstance(fields, list)
    for field in fields:
        assert isinstance(field, dict)
        if field["key"] == "MODEL_FALLBACKS":
            return field
    raise AssertionError("Fallback field is missing")


@pytest.mark.parametrize(
    "effort", ["inherit", "off", "client", "low", "medium", "high", "xhigh", "max"]
)
@pytest.mark.parametrize("as_json", [False, True])
def test_fallback_records_preserve_model_and_effort(effort, as_json):
    value = [
        {"model": " groq/vendor/Model-A ", "reasoning": effort},
        {"model": "open_router/vendor/model-b"},
    ]
    settings = Settings.model_validate(
        {"MODEL_FALLBACKS": json.dumps(value) if as_json else value}
    )
    assert settings.model_fallbacks is not None
    assert [
        (entry.model, entry.reasoning.value) for entry in settings.model_fallbacks
    ] == [
        ("groq/vendor/Model-A", effort),
        ("open_router/vendor/model-b", "inherit"),
    ]
    with pytest.raises(ValidationError):
        settings.model_fallbacks[0].__setattr__("reasoning", ReasoningPreference.OFF)


@pytest.mark.parametrize(
    "value",
    [
        "groq/one, open_router/vendor/two ",
        ["groq/one", "open_router/vendor/two"],
        ("groq/one", "open_router/vendor/two"),
    ],
)
def test_legacy_fallbacks_are_inherit_records(value):
    settings = Settings.model_validate({"MODEL_FALLBACKS": value})
    assert settings.model_fallbacks is not None
    assert [
        (entry.model, entry.reasoning.value) for entry in settings.model_fallbacks
    ] == [("groq/one", "inherit"), ("open_router/vendor/two", "inherit")]


@pytest.mark.parametrize(
    "value",
    [
        [{"model": "groq/one", "reasoning": "minimal"}],
        [{"model": "groq/one", "reasoning": True}],
        [{"model": "groq/one", "reasoning": "low", "unknown": 1}],
        [{"model": "groq/"}],
        [{"model": "unknown/one"}],
        [
            {"model": "groq/one", "reasoning": "low"},
            {"model": "groq/one", "reasoning": "high"},
        ],
        '[{"model":"groq/one"}',
    ],
)
def test_invalid_fallback_records_are_rejected(value):
    with pytest.raises(ValidationError):
        Settings.model_validate({"MODEL_FALLBACKS": value})


def test_fallback_records_retain_primary_overlap_and_discovery_identity():
    settings = Settings(
        MODEL="groq/one",
        MODEL_OPUS="groq/one",
        MODEL_FALLBACKS=[
            {"model": "groq/one", "reasoning": "off"},
            {"model": "open_router/vendor/two", "reasoning": "max"},
        ],
    )
    assert [entry.model_ref for entry in configured_chat_model_refs(settings)] == [
        "groq/one",
        "open_router/vendor/two",
    ]


@pytest.mark.parametrize("as_json", [False, True])
def test_admin_fallback_records_round_trip_without_restart_or_false_changes(as_json):
    entries = [
        {"model": "groq/one", "reasoning": "low"},
        {"model": "open_router/vendor/two", "reasoning": "inherit"},
    ]
    canonical = [
        {"model": "groq/one", "reasoning": "low", "api_key_number": 1},
        {
            "model": "open_router/vendor/two",
            "reasoning": "inherit",
            "api_key_number": 1,
        },
    ]
    original = _snapshot()
    saved = prepare_admin_update(
        {"MODEL_FALLBACKS": json.dumps(entries) if as_json else entries},
        original,
        original.settings,
        revision_key=b"test-revision-key",
    )
    assert saved.valid, saved.errors
    assert saved.pending_fields == ()
    assert json.loads(saved.target_values["MODEL_FALLBACKS"]) == canonical
    loaded = _snapshot(saved.target_values)
    field = _fallback_field(loaded)
    value = field["value"]
    assert isinstance(value, str) and json.loads(value) == canonical
    assert field["source"] == ConfigSource.MANAGED.value
    options = field["options"]
    assert isinstance(options, list)
    values = []
    for option in options:
        assert isinstance(option, dict)
        values.append(option["value"])
    assert values == [
        "inherit",
        "off",
        "client",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    unchanged = prepare_admin_update(
        {"MODEL_FALLBACKS": field["value"]},
        loaded,
        loaded.settings,
        revision_key=b"test-revision-key",
    )
    assert unchanged.valid and unchanged.changed_keys == ()


@pytest.mark.parametrize("empty", ["", "[]", []])
def test_empty_process_fallbacks_override_managed_records_and_remain_locked(empty):
    managed = {"MODEL_FALLBACKS": '[{"model":"groq/one","reasoning":"max"}]'}
    state = _snapshot(
        managed,
        {"MODEL_FALLBACKS": json.dumps(empty) if isinstance(empty, list) else empty},
    )
    assert state.settings.model_fallbacks is None
    field = _fallback_field(state)
    assert field["locked"] is True and field["value"] is None
    result = prepare_admin_update(
        {"MODEL_FALLBACKS": [{"model": "groq/two", "reasoning": "off"}]},
        state,
        state.settings,
        revision_key=b"test-revision-key",
    )
    assert result.valid and result.settings is not None
    assert result.settings.model_fallbacks is None
    assert json.loads(result.target_values["MODEL_FALLBACKS"]) == [
        {"model": "groq/one", "reasoning": "max", "api_key_number": 1}
    ]


def test_retired_provider_normalization_preserves_surviving_efforts():
    state = _snapshot(
        {
            "MODEL_FALLBACKS": json.dumps(
                [
                    {"model": "groq/one", "reasoning": "low"},
                    {"model": "github_models/retired", "reasoning": "high"},
                    {"model": "deepseek/two", "reasoning": "max"},
                ]
            )
        }
    )
    assert state.settings.model_fallbacks is not None
    assert [
        (entry.model, entry.reasoning.value) for entry in state.settings.model_fallbacks
    ] == [("groq/one", "low"), ("deepseek/two", "max")]
