"""Submitted lists request checks even when the stored credentials are unchanged."""

from pydantic import SecretStr

from free_claude_code.config.admin.persistence import (
    prepare_admin_update,
)
from free_claude_code.config.admin.values import MASKED_SECRET, load_config_response
from free_claude_code.config.loader import ManagedConfigStore
from tests.config.credential_support import credential_edit


def prepare(updates, snapshot, active):
    public = load_config_response(snapshot, revision_key=b"test-revision-key")
    rows = public["fields"]
    assert isinstance(rows, list)
    fields = {field["key"]: field for field in rows if isinstance(field, dict)}
    updates = {
        key: credential_edit(fields[key]["value"], value)
        if fields[key]["type"] == "api_key_list"
        else value
        for key, value in updates.items()
    }
    return prepare_admin_update(
        updates, snapshot, active, revision_key=b"test-revision-key"
    )


def test_changed_keys_exclude_masks_noops_and_removals(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    store = ManagedConfigStore()
    store.initialize()
    active = store.read().settings
    first = prepare({"GROQ_API_KEY": " saved-key "}, store.read(), active)
    assert first.valid
    assert first.changed_keys == ("GROQ_API_KEY",)
    store.commit(first.target_values)
    # Compare to the saved value even if a running generation has not restarted.
    for value in ("saved-key", " saved-key ", MASKED_SECRET, "", "   ", None):
        prepared = prepare({"GROQ_API_KEY": value}, store.read(), active)
        assert prepared.valid
        assert prepared.changed_keys == ()
    changed = prepare(
        {"GROQ_API_KEY": "new-key", "GROQ_PROXY": "http://localhost:8080"},
        store.read(),
        active,
    )
    assert "GROQ_API_KEY" in changed.changed_keys
    assert changed.settings is not None
    assert changed.settings.groq_api_keys == (SecretStr("new-key"),)
    assert changed.settings.groq_proxy == "http://localhost:8080"


def test_process_locked_key_is_not_a_change(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "process-key")
    store = ManagedConfigStore()
    store.initialize()
    active = store.read().settings
    prepared = prepare({"GROQ_API_KEY": MASKED_SECRET}, store.read(), active)
    assert prepared.valid
    assert prepared.changed_keys == ()
    assert prepared.settings is not None
    assert prepared.settings.groq_api_keys == (SecretStr("process-key"),)
    assert prepared.credential_submissions[0].credentials == (SecretStr("process-key"),)
