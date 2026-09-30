import json
from pathlib import Path

from free_claude_code.config.admin.custom_providers import CustomProviderMutation
from free_claude_code.config.admin.persistence import prepare_admin_update
from free_claude_code.config.admin.values import load_config_response
from free_claude_code.config.custom_providers import CUSTOM_PROVIDERS_ENV
from free_claude_code.config.loader import (
    ManagedConfigSnapshot,
    compose_settings_snapshot,
)
from free_claude_code.runtime.diagnostics import settings_report


def snapshot(values=None, process=None):
    values, process = values or {}, process or {}
    result = compose_settings_snapshot(values, process)
    return ManagedConfigSnapshot(
        result.settings, result.sources, values, process, Path("unused.env")
    )


def mutate(state, action, provider_id=None, values=None, routes=None):
    return prepare_admin_update(
        routes or {},
        state,
        state.settings,
        CustomProviderMutation(
            action=action, provider_id=provider_id, values=values or {}
        ),
    )


def created():
    original = snapshot()
    result = mutate(
        original,
        "create",
        values={
            "display_name": "Gateway",
            "base_url": "https://example.org/private/path",
            "api_key": "private-api-key",
            "model_ids": ["one"],
        },
    )
    assert result.valid
    return snapshot(result.target_values), result.custom_provider_id


def test_mutations_preserve_other_providers_secrets_and_routing_identity():
    state, first = created()
    second = mutate(
        state,
        "create",
        values={"display_name": "Second", "base_url": "http://localhost:9999/api"},
    )
    state = snapshot(second.target_values)
    edited = mutate(
        state, "update", first, {"display_name": "Renamed", "api_key": "********"}
    )
    assert edited.valid and len(edited.settings.custom_providers) == 2
    original = edited.settings.custom_provider(first)
    assert original.display_name == "Renamed"
    assert original.api_key.get_secret_value() == "private-api-key"
    cleared = mutate(snapshot(edited.target_values), "update", first, {"api_key": None})
    assert cleared.settings.custom_provider(first).api_key is None


def test_deletion_requires_reassignment_and_is_idempotent():
    state, provider_id = created()
    selected = snapshot(state.managed | {"MODEL": f"{provider_id}/one"})
    rejected = mutate(selected, "delete", provider_id)
    assert not rejected.valid and "MODEL" in str(rejected.errors)
    removed = mutate(
        selected, "delete", provider_id, routes={"MODEL": state.settings.model}
    )
    assert removed.valid and not removed.settings.custom_providers
    assert mutate(snapshot(removed.target_values), "delete", provider_id).valid
    assert not mutate(
        snapshot(removed.target_values),
        "update",
        provider_id,
        {"display_name": "stale"},
    ).valid


def test_invalid_duplicate_or_locked_mutation_does_not_expose_keys():
    state, provider_id = created()
    duplicate = mutate(
        state,
        "create",
        values={"display_name": "GATEWAY", "base_url": "http://localhost"},
    )
    assert not duplicate.valid
    invalid = mutate(
        state,
        "update",
        provider_id,
        {"base_url": "https://user:private-api-key@example.org"},
    )
    assert not invalid.valid
    assert "private-api-key" not in json.dumps(invalid.applied_response())
    locked = snapshot(
        state.managed, {CUSTOM_PROVIDERS_ENV: state.managed[CUSTOM_PROVIDERS_ENV]}
    )
    assert not mutate(locked, "delete", provider_id).valid
    public = load_config_response(state)
    rows = public["custom_providers"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["api_key"] == "********"
    assert "private-api-key" not in json.dumps(public)
    doctor = json.dumps(settings_report(state.settings, {}))
    assert (
        "Gateway" in doctor
        and "private-api-key" not in doctor
        and "/private/path" not in doctor
    )
