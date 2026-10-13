"""Masked row references retain exact secrets and reject stale/invalid edits."""

import json

import pytest
from pydantic import SecretStr

from free_claude_code.config.admin.custom_providers import CustomProviderMutation
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.runtime.configuration import ConfigurationService


@pytest.fixture
def service(monkeypatch):
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_env:
            monkeypatch.delenv(descriptor.credential_env, raising=False)
    store = ManagedConfigStore()
    store.initialize()
    return ConfigurationService(store)


async def key_edit(service, key, entries):
    public = await service.admin_config()
    field = next(item for item in public["fields"] if item["key"] == key)
    return {"revision": field["value"]["revision"], "entries": entries}


async def save_keys(service, values):
    edit = await key_edit(
        service, "OPENAI_API_KEY", [{"value": value} for value in values]
    )
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    assert prepared.valid
    await service.commit(prepared)
    assert "private-first" not in repr(service._store.read())


@pytest.mark.asyncio
async def test_reorder_replace_and_clear_masked_keys_round_trip(service):
    await save_keys(service, ["private-first", "private-second", "private-third"])
    edit = await key_edit(
        service,
        "OPENAI_API_KEY",
        [{"retain": 2}, {"value": "replacement"}, {"retain": 0}],
    )
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    assert prepared.valid
    assert prepared.settings.openai_api_keys == tuple(
        map(SecretStr, ("private-third", "replacement", "private-first"))
    )
    await service.commit(prepared)
    public = await service.admin_config()
    assert "private-first" not in json.dumps(public)
    assert "private-third" not in json.dumps(public)
    assert "replacement" not in json.dumps(public)
    edit = await key_edit(service, "OPENAI_API_KEY", [])
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    assert prepared.valid
    await service.commit(prepared)
    assert service._store.read().settings.openai_api_keys == ()
    assert "OPENAI_API_KEY" not in service._store.read().managed


@pytest.mark.asyncio
async def test_stale_masked_reference_cannot_replace_another_saved_key(service):
    await save_keys(service, ["one", "two"])
    stale = await key_edit(service, "OPENAI_API_KEY", [{"retain": 1}])
    await save_keys(service, ["new-one", "new-two"])
    before = service._store.path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        await service.prepare({"OPENAI_API_KEY": stale}, service._store.read().settings)
    assert service._store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_references_are_bound_to_owner_and_server_lifetime(service):
    edit = await key_edit(service, "OPENAI_API_KEY", [{"value": "one"}])
    with pytest.raises(ValueError, match="changed"):
        await service.prepare({"GROQ_API_KEY": edit}, service._store.read().settings)
    replacement_service = ConfigurationService(service._store)
    with pytest.raises(ValueError, match="changed"):
        await replacement_service.prepare(
            {"OPENAI_API_KEY": edit}, service._store.read().settings
        )


@pytest.mark.asyncio
async def test_duplicate_retained_and_new_keys_highlights_both_rows(service):
    await save_keys(service, ["private-first"])
    edit = await key_edit(
        service,
        "OPENAI_API_KEY",
        [{"retain": 0}, {"value": " private-first "}, {"value": " "}],
    )
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    assert not prepared.valid
    response = prepared.applied_response()
    assert {row["index"] for row in response["credential_row_errors"]} == {0, 1, 2}
    assert "private-first" not in json.dumps(response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        {"retain": True},
        {"retain": -1},
        {"retain": 2},
        {"retain": 0, "value": "one"},
        {"value": 3},
    ],
)
async def test_invalid_entries_do_not_change_managed_credentials(service, entry):
    await save_keys(service, ["one"])
    edit = await key_edit(service, "OPENAI_API_KEY", [entry])
    before = service._store.path.read_bytes()
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    assert not prepared.valid
    assert prepared.applied_response()["credential_row_errors"][0]["index"] == 0
    assert service._store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_locked_lists_allow_check_submission_but_reject_mutation(
    service, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", '["one","two"]')
    edit = await key_edit(service, "OPENAI_API_KEY", [{"retain": 0}, {"retain": 1}])
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit, "PORT": "8123"}, service._store.read().settings
    )
    assert prepared.valid
    assert "OPENAI_API_KEY" not in prepared.target_values
    assert prepared.credential_submissions[0].credentials == (
        SecretStr("one"),
        SecretStr("two"),
    )
    edit["entries"].reverse()
    rejected = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    assert not rejected.valid


@pytest.mark.asyncio
async def test_commit_rejects_other_prepared_write_and_preserves_newer_settings(
    service,
):
    active = service._store.read().settings
    first = await service.prepare({"PORT": "8123"}, active)
    second = await service.prepare({"HOST": "localhost"}, active)
    await service.commit(first)
    with pytest.raises(ValueError, match="changed"):
        await service.commit(second)
    assert service._store.read().settings.port == 8123


@pytest.mark.asyncio
async def test_commit_rejects_changed_process_ownership(service, monkeypatch):
    await save_keys(service, ["managed"])
    edit = await key_edit(service, "OPENAI_API_KEY", [{"value": "replacement"}])
    prepared = await service.prepare(
        {"OPENAI_API_KEY": edit}, service._store.read().settings
    )
    before = service._store.path.read_bytes()
    monkeypatch.setenv("OPENAI_API_KEY", "process")
    with pytest.raises(ValueError, match="changed"):
        await service.commit(prepared)
    assert service._store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_custom_list_uses_same_masked_protocol(service):
    mutation = CustomProviderMutation(
        action="create",
        values={
            "display_name": "Custom",
            "base_url": "https://custom.example/v1",
            "api_keys": {
                "revision": None,
                "entries": [{"value": "custom-one"}, {"value": "custom-two"}],
            },
        },
    )
    prepared = await service.prepare({}, service._store.read().settings, mutation)
    assert prepared.valid
    await service.commit(prepared)
    public = await service.admin_config()
    definition = public["custom_providers"][0]
    assert "custom-one" not in json.dumps(public)
    edit = {
        "revision": definition["api_keys"]["revision"],
        "entries": [{"retain": 1}, {"retain": 0}],
    }
    prepared = await service.prepare(
        {},
        service._store.read().settings,
        CustomProviderMutation(
            action="update",
            provider_id=definition["provider_id"],
            values={"api_keys": edit},
        ),
    )
    assert prepared.valid
    assert prepared.settings.custom_providers[0].api_keys == (
        SecretStr("custom-two"),
        SecretStr("custom-one"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "device,key", [("nvidia_nim", "NVIDIA_NIM_API_KEY"), ("cpu", "HUGGINGFACE_API_KEY")]
)
async def test_voice_restart_uses_first_key_and_ignores_tail_edits(
    service, device, key
):
    edit = await key_edit(
        service, key, [{"value": "voice-first"}, {"value": "voice-tail"}]
    )
    prepared = await service.prepare(
        {key: edit, "VOICE_NOTE_ENABLED": "true", "WHISPER_DEVICE": device},
        service._store.read().settings,
    )
    assert prepared.valid
    await service.commit(prepared)
    active = service._store.read().settings
    edit = await key_edit(service, key, [{"retain": 0}, {"value": "new-tail"}])
    prepared = await service.prepare({key: edit}, active)
    assert prepared.valid and key not in prepared.pending_fields
    await service.commit(prepared)
    edit = await key_edit(service, key, [{"retain": 1}, {"retain": 0}])
    prepared = await service.prepare({key: edit}, active)
    assert prepared.valid and key in prepared.pending_fields
