"""Only effective edits, against saved settings, request credential validation."""

from free_claude_code.config.admin.persistence import (
    prepare_admin_update,
)
from free_claude_code.config.admin.values import MASKED_SECRET
from free_claude_code.config.loader import ManagedConfigStore


def test_changed_keys_exclude_masks_noops_and_removals(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    store = ManagedConfigStore()
    store.initialize()
    active = store.read().settings
    first = prepare_admin_update({"GROQ_API_KEY": " saved-key "}, store.read(), active)
    assert first.valid
    assert first.changed_keys == ("GROQ_API_KEY",)
    store.commit(first.target_values)
    # Compare to the saved value even if a running generation has not restarted.
    for value in ("saved-key", " saved-key ", MASKED_SECRET, "", "   ", None):
        prepared = prepare_admin_update({"GROQ_API_KEY": value}, store.read(), active)
        assert prepared.valid
        assert prepared.changed_keys == ()
    changed = prepare_admin_update(
        {"GROQ_API_KEY": "new-key", "GROQ_PROXY": "http://localhost:8080"},
        store.read(),
        active,
    )
    assert "GROQ_API_KEY" in changed.changed_keys
    assert changed.settings is not None
    assert changed.settings.groq_api_key == "new-key"
    assert changed.settings.groq_proxy == "http://localhost:8080"


def test_process_locked_key_is_not_a_change(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "process-key")
    store = ManagedConfigStore()
    store.initialize()
    active = store.read().settings
    prepared = prepare_admin_update(
        {"GROQ_API_KEY": "submitted-key"}, store.read(), active
    )
    assert prepared.valid
    assert prepared.changed_keys == ()
    assert prepared.settings is not None
    assert prepared.settings.groq_api_key == "process-key"
