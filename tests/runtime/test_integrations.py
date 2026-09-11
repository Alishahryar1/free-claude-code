"""Lifecycle tests use real files, synthetic installations, and no live clients."""

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from free_claude_code.application.integrations import IntegrationAction as Action
from free_claude_code.application.integrations import IntegrationError
from free_claude_code.application.integrations import IntegrationId as Item
from free_claude_code.runtime.integrations import service as service_module
from free_claude_code.runtime.integrations import storage as storage_module
from free_claude_code.runtime.integrations.documents import JsonDocument
from free_claude_code.runtime.integrations.service import IntegrationService
from tests.integration_support import connection, installed_clients, write_json


@pytest.fixture
def setup(tmp_path):
    locator = installed_clients(tmp_path)
    return IntegrationService(locator), locator, connection(tmp_path)


def perform(service, ctx, item, action):
    preview = service.preview(item, action, ctx)
    return service.apply(item, action, preview["revision"], ctx)


def card(service, ctx, item):
    return next(entry for entry in service.inspect(ctx)["items"] if entry["id"] == item)


def test_inspection_and_preview_do_not_write_configuration_or_ownership(setup):
    service, locator, ctx = setup
    before = set(locator.home.rglob("*"))
    assert len(service.inspect(ctx)["items"]) == 4
    preview = service.preview(Item.CLAUDE_VSCODE, Action.SETUP, ctx)
    assert preview["changes"]
    assert set(locator.home.rglob("*")) == before
    assert ctx.settings.proxy_auth_token not in json.dumps(preview)


def test_vscode_setup_update_disconnect_restore_originals_and_unrelated_edits(setup):
    service, locator, ctx = setup
    original = b'{\n // keep\n "editor.fontSize":15,\n "claudeCode.disableLoginPrompt":false,\n "claudeCode.environmentVariables":[{"name":"KEEP","value":"mine"},{"name":"ANTHROPIC_BASE_URL","value":"https://previous.example"}]\n}'
    locator.vscode_settings.parent.mkdir(parents=True)
    locator.vscode_settings.write_bytes(original)
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    document = JsonDocument(locator.vscode_settings.read_bytes(), jsonc=True)
    assert document.get(("claudeCode.disableLoginPrompt",)) is True
    document.set(("editor.fontSize",), 20)
    locator.vscode_settings.write_bytes(document.render())
    changed = connection(locator.home, port=8182, token="new-token")
    assert card(service, changed, Item.CLAUDE_VSCODE)["status"] == "update_available"
    perform(service, changed, Item.CLAUDE_VSCODE, Action.UPDATE)
    result = perform(service, changed, Item.CLAUDE_VSCODE, Action.DISCONNECT)
    restored = JsonDocument(locator.vscode_settings.read_bytes(), jsonc=True)
    assert restored.get(("claudeCode.environmentVariables",)) == [
        {"name": "KEEP", "value": "mine"},
        {"name": "ANTHROPIC_BASE_URL", "value": "https://previous.example"},
    ]
    assert restored.get(("claudeCode.disableLoginPrompt",)) is False
    assert restored.get(("editor.fontSize",)) == 20
    assert b"// keep" in locator.vscode_settings.read_bytes()
    assert result["applied"]


def test_codex_shared_setup_and_restore_existing_provider_model_catalog(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    original = b'# user\nmodel="old"\nmodel_provider="other"\nmodel_catalog_json="custom.json"\n[model_providers.other]\nname="Keep"\n'
    path.write_bytes(original)
    perform(service, ctx, Item.CODEX, Action.SETUP)
    data = tomllib.loads(path.read_text())
    assert data["model"] == "test/model"
    assert data["model_providers"]["fcc"]["auth"]["command"] == str(
        locator.home / "bin/fcc-codex"
    )
    assert data["model_providers"]["other"]["name"] == "Keep"
    perform(service, connection(locator.home, port=8182), Item.CODEX, Action.UPDATE)
    perform(service, ctx, Item.CODEX, Action.DISCONNECT)
    data = tomllib.loads(path.read_text())
    assert data["model"] == "old" and data["model_provider"] == "other"
    assert data["model_catalog_json"] == "custom.json"
    assert b"# user" in path.read_bytes()


def test_codex_update_keeps_user_model_and_disconnect_stops_on_that_edit(setup):
    service, locator, ctx = setup
    perform(service, ctx, Item.CODEX, Action.SETUP)
    path = locator.home / ".codex/config.toml"
    path.write_bytes(path.read_bytes().replace(b'"test/model"', b'"user/model"'))
    next_ctx = connection(locator.home, port=8182)
    perform(service, next_ctx, Item.CODEX, Action.UPDATE)
    assert tomllib.loads(path.read_text())["model"] == "user/model"
    before = path.read_bytes()
    with pytest.raises(IntegrationError, match="model"):
        service.preview(Item.CODEX, Action.DISCONNECT, next_ctx)
    assert path.read_bytes() == before


def test_later_edit_blocks_whole_update_and_setup_cannot_bypass_it(setup):
    service, locator, ctx = setup
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    path = locator.vscode_settings
    path.write_bytes(path.read_bytes().replace(b"fixture-proxy-token", b"user-token"))
    before = path.read_bytes()
    changed = connection(locator.home, port=8182, token="new-token")
    assert card(service, changed, Item.CLAUDE_VSCODE)["status"] == "needs_attention"
    for action in (Action.UPDATE, Action.DISCONNECT, Action.SETUP):
        with pytest.raises(IntegrationError):
            service.preview(Item.CLAUDE_VSCODE, action, changed)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["file", "connection", "dependency"])
def test_stale_preview_cannot_write(setup, change):
    service, locator, ctx = setup
    preview = service.preview(Item.CLAUDE_VSCODE, Action.SETUP, ctx)
    if change == "file":
        write_json(locator.vscode_settings, {"keep": "latest"})
    elif change == "connection":
        ctx = connection(locator.home, port=9090)
    else:
        (locator.home / ".vscode/extensions/extensions.json").unlink()
    before = (
        locator.vscode_settings.read_bytes()
        if locator.vscode_settings.exists()
        else None
    )
    with pytest.raises(IntegrationError):
        service.apply(Item.CLAUDE_VSCODE, Action.SETUP, preview["revision"], ctx)
    assert (
        locator.vscode_settings.read_bytes()
        if locator.vscode_settings.exists()
        else None
    ) == before


def test_duplicate_confirmation_keeps_original_undo_baseline(setup):
    service, locator, ctx = setup
    preview = service.preview(Item.CLAUDE_VSCODE, Action.SETUP, ctx)
    service.apply(Item.CLAUDE_VSCODE, Action.SETUP, preview["revision"], ctx)
    before = locator.vscode_settings.read_bytes()
    with pytest.raises(IntegrationError):
        service.apply(Item.CLAUDE_VSCODE, Action.SETUP, preview["revision"], ctx)
    assert locator.vscode_settings.read_bytes() == before
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.DISCONNECT)
    assert JsonDocument(locator.vscode_settings.read_bytes(), jsonc=True).data == {}


def test_manual_connection_removal_does_not_invent_prior_settings(setup):
    service, locator, ctx = setup
    write_json(
        locator.vscode_settings,
        {
            "keep": 1,
            "claudeCode.disableLoginPrompt": True,
            "claudeCode.environmentVariables": [
                {"name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082"},
                {"name": "ANTHROPIC_AUTH_TOKEN", "value": "old-token"},
                {"name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "value": "1"},
                {"name": "KEEP", "value": "mine"},
            ],
        },
    )
    preview = service.preview(Item.CLAUDE_VSCODE, Action.DISCONNECT, ctx)
    assert "previous" in " ".join(preview["notes"]).lower()
    service.apply(Item.CLAUDE_VSCODE, Action.DISCONNECT, preview["revision"], ctx)
    data = json.loads(locator.vscode_settings.read_bytes())
    assert data["keep"] == 1
    assert data["claudeCode.environmentVariables"] == [
        {"name": "KEEP", "value": "mine"}
    ]


def test_manual_update_adoption_disconnects_instead_of_restoring_old_fcc_url(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        'model_provider="fcc"\n[model_providers.fcc]\nname="Free Claude Code"\nbase_url="http://localhost:7777/v1"\n'
    )
    perform(service, ctx, Item.CODEX, Action.UPDATE)
    perform(service, ctx, Item.CODEX, Action.DISCONNECT)
    data = tomllib.loads(path.read_text())
    assert "model_provider" not in data
    assert "7777" not in path.read_text()


def test_jetbrains_custom_agent_preserves_other_agents_and_added_fields_block_removal(
    setup,
):
    service, locator, ctx = setup
    path = locator.home / ".jetbrains/acp.json"
    write_json(path, {"agent_servers": {"Other": {"command": "keep"}}})
    perform(service, ctx, Item.CLAUDE_JETBRAINS, Action.SETUP)
    data = json.loads(path.read_bytes())
    assert data["agent_servers"]["Other"] == {"command": "keep"}
    data["agent_servers"]["Claude Code (FCC)"]["custom"] = True
    write_json(path, data)
    perform(
        service,
        connection(locator.home, port=8182),
        Item.CLAUDE_JETBRAINS,
        Action.UPDATE,
    )
    with pytest.raises(IntegrationError):
        service.preview(Item.CLAUDE_JETBRAINS, Action.DISCONNECT, ctx)
    assert (
        json.loads(path.read_bytes())["agent_servers"]["Claude Code (FCC)"]["custom"]
        is True
    )


@pytest.mark.parametrize(
    "original",
    [None, {}, {"hasCompletedOnboarding": False, "private": "do-not-return"}],
)
def test_separate_onboarding_repair_changes_only_its_flag(setup, original):
    service, locator, ctx = setup
    path = locator.home / ".claude.json"
    if original is not None:
        write_json(path, original)
    preview = service.preview(Item.CLAUDE_LOGIN, Action.REPAIR, ctx)
    assert "do-not-return" not in json.dumps(preview)
    service.apply(Item.CLAUDE_LOGIN, Action.REPAIR, preview["revision"], ctx)
    assert json.loads(path.read_bytes()) == {
        **(original or {}),
        "hasCompletedOnboarding": True,
    }
    before = path.read_bytes()
    assert service.preview(Item.CLAUDE_LOGIN, Action.REPAIR, ctx)["changes"] == []
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.DISCONNECT)
    assert path.read_bytes() == before


def test_onboarding_does_not_repair_other_login_failures_or_invalid_flags(setup):
    service, locator, ctx = setup
    for flag in ("false", 1):
        write_json(locator.home / ".claude.json", {"hasCompletedOnboarding": flag})
        with pytest.raises(IntegrationError):
            service.preview(Item.CLAUDE_LOGIN, Action.REPAIR, ctx)


def test_missing_client_and_catalog_do_not_trigger_installation_or_fallback_model(
    setup,
):
    service, locator, ctx = setup
    (locator.home / "bin/fcc-codex").unlink()
    with pytest.raises(IntegrationError):
        service.preview(Item.CODEX, Action.SETUP, ctx)
    with pytest.raises(IntegrationError):
        service.preview(
            Item.CODEX,
            Action.SETUP,
            replace(ctx, models=()),
        )


def test_codex_auth_cleanup_is_masked_and_does_not_touch_other_providers(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        '[model_providers.fcc]\nname="Old"\nenv_key="PRIVATE_ENV"\nexperimental_bearer_token="SECRET_BEARER"\nrequires_openai_auth=true\n[model_providers.other]\nname="Keep"\n'
    )
    preview = service.preview(Item.CODEX, Action.SETUP, ctx)
    assert "SECRET_BEARER" not in json.dumps(preview)
    service.apply(Item.CODEX, Action.SETUP, preview["revision"], ctx)
    assert (
        "experimental_bearer_token"
        not in tomllib.loads(path.read_text())["model_providers"]["fcc"]
    )
    perform(service, ctx, Item.CODEX, Action.DISCONNECT)
    assert (
        tomllib.loads(path.read_text())["model_providers"]["fcc"][
            "experimental_bearer_token"
        ]
        == "SECRET_BEARER"
    )


def test_codex_token_rotation_does_not_offer_update_but_known_auth_mismatch_blocks_setup(
    setup,
):
    service, locator, ctx = setup
    perform(service, ctx, Item.CODEX, Action.SETUP)
    assert (
        card(service, connection(locator.home, token="changed"), Item.CODEX)["status"]
        == "configured"
    )
    other = IntegrationService(installed_clients(locator.home / "other"))
    bad = connection(locator.home / "other")
    bad = replace(
        bad,
        settings=bad.settings.model_copy(update={"proxy_auth_enabled": True}),
        saved_auth_token="different",
    )
    with pytest.raises(IntegrationError):
        other.preview(Item.CODEX, Action.SETUP, bad)


def test_known_conflicting_claude_settings_are_reported_without_modification(setup):
    service, locator, ctx = setup
    path = locator.home / ".claude/settings.json"
    write_json(path, {"env": {"ANTHROPIC_BASE_URL": "https://other.example"}})
    before = path.read_bytes()
    with pytest.raises(IntegrationError):
        service.preview(Item.CLAUDE_VSCODE, Action.SETUP, ctx)
    assert path.read_bytes() == before


def test_disconnect_works_after_app_and_helper_are_removed(setup):
    service, locator, ctx = setup
    perform(service, ctx, Item.CODEX, Action.SETUP)
    (locator.home / "bin/fcc-codex").unlink()
    (locator.home / "bin/chatgpt").unlink()
    (locator.home / ".vscode/extensions/extensions.json").unlink()
    perform(service, ctx, Item.CODEX, Action.DISCONNECT)
    assert "model_provider" not in tomllib.loads(
        (locator.home / ".codex/config.toml").read_text()
    )


def test_read_only_or_malformed_file_is_never_overwritten(setup):
    service, locator, ctx = setup
    path = locator.home / ".claude.json"
    path.write_bytes(b'{"broken":')
    with pytest.raises(IntegrationError):
        service.preview(Item.CLAUDE_LOGIN, Action.REPAIR, ctx)
    path.write_bytes(b"{}")
    path.chmod(0o400)
    try:
        with pytest.raises(IntegrationError):
            perform(service, ctx, Item.CLAUDE_LOGIN, Action.REPAIR)
        assert path.read_bytes() == b"{}"
    finally:
        path.chmod(0o600)


def test_symlink_configuration_requires_manual_setup(setup):
    service, locator, ctx = setup
    target = locator.home / "other.json"
    target.write_bytes(b"{}")
    try:
        (locator.home / ".claude.json").symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this host")
    with pytest.raises(IntegrationError):
        service.preview(Item.CLAUDE_LOGIN, Action.REPAIR, ctx)
    assert target.read_bytes() == b"{}"


@pytest.mark.parametrize("stage", ["pending", "external", "committed"])
def test_failed_write_keeps_original_settings_recoverable(setup, monkeypatch, stage):
    service, locator, ctx = setup
    original = {"claudeCode.disableLoginPrompt": False, "editor.fontSize": 16}
    write_json(locator.vscode_settings, original)
    actual_record = service_module.write_record
    actual_write = service_module.atomic_write

    def fail_record(path, record):
        pending = isinstance(record, service_module.PendingWrite)
        if (stage == "pending" and pending) or (stage == "committed" and not pending):
            raise OSError("private error must not escape")
        actual_record(path, record)

    def fail_external(path, data, **kwargs):
        if stage == "external" and path == locator.vscode_settings:
            raise OSError("private error must not escape")
        actual_write(path, data, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "write_record", fail_record)
        patch.setattr(service_module, "atomic_write", fail_external)
        with pytest.raises(IntegrationError, match="Could not save"):
            perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    if stage != "committed":
        assert json.loads(locator.vscode_settings.read_bytes()) == original
        perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    else:
        assert card(service, ctx, Item.CLAUDE_VSCODE)["status"] == "needs_attention"
        assert card(service, ctx, Item.CLAUDE_VSCODE)["actions"] == ["update"]
        perform(service, ctx, Item.CLAUDE_VSCODE, Action.UPDATE)
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.DISCONNECT)
    assert json.loads(locator.vscode_settings.read_bytes()) == original
    assert not (service.state_dir / "claude-vscode.json").exists()


@pytest.mark.parametrize(
    "item,action",
    [(Item.CLAUDE_LOGIN, Action.REPAIR), (Item.CLAUDE_VSCODE, Action.DISCONNECT)],
)
def test_completed_write_with_failed_record_cleanup_can_be_finished(
    setup, monkeypatch, item, action
):
    service, _locator, ctx = setup
    if action == Action.DISCONNECT:
        perform(service, ctx, item, Action.SETUP)
    actual_record = service_module.write_record

    def fail_cleanup(path, record):
        if record is None:
            raise OSError("cleanup failure")
        actual_record(path, record)

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "write_record", fail_cleanup)
        with pytest.raises(IntegrationError):
            perform(service, ctx, item, action)
    assert action in card(service, ctx, item)["actions"]
    result = perform(service, ctx, item, action)
    assert result["applied"] is False
    assert not (service.state_dir / f"{item}.json").exists()


def test_interference_after_replace_retains_pending_record_and_refuses_more_writes(
    setup, monkeypatch
):
    service, locator, ctx = setup
    actual_write = service_module.atomic_write

    def interfere(path, data, **kwargs):
        actual_write(path, data, **kwargs)
        if path == locator.vscode_settings:
            write_json(path, {"user": "later edit"})

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "atomic_write", interfere)
        with pytest.raises(IntegrationError, match="verification"):
            perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    assert card(service, ctx, Item.CLAUDE_VSCODE)["status"] == "needs_attention"
    with pytest.raises(IntegrationError, match="interrupted write"):
        service.preview(Item.CLAUDE_VSCODE, Action.SETUP, ctx)
    assert json.loads(locator.vscode_settings.read_bytes()) == {"user": "later edit"}
    assert (service.state_dir / "claude-vscode.json").exists()


def test_concurrent_confirmations_across_services_have_one_winner(setup):
    service, locator, ctx = setup
    other = IntegrationService(locator)
    barrier = Barrier(2)

    def confirm(owner):
        preview = owner.preview(Item.CLAUDE_VSCODE, Action.SETUP, ctx)
        barrier.wait(timeout=5)
        try:
            return owner.apply(
                Item.CLAUDE_VSCODE, Action.SETUP, preview["revision"], ctx
            )["applied"]
        except IntegrationError as error:
            assert error.status_code == 409
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(confirm, [service, other])) == [False, True]
    perform(service, ctx, Item.CLAUDE_VSCODE, Action.DISCONNECT)
    assert json.loads(locator.vscode_settings.read_bytes()) == {}


def test_manual_jetbrains_disconnect_removes_exact_recipe_entry(setup):
    service, locator, ctx = setup
    perform(service, ctx, Item.CLAUDE_JETBRAINS, Action.SETUP)
    (service.state_dir / "claude-jetbrains.json").unlink()
    perform(service, ctx, Item.CLAUDE_JETBRAINS, Action.DISCONNECT)
    assert "Claude Code (FCC)" not in json.loads(
        (locator.home / ".jetbrains/acp.json").read_bytes()
    ).get("agent_servers", {})


def test_missing_codex_default_is_explained_on_the_card(setup):
    service, _locator, ctx = setup
    missing_default = replace(ctx, models=())
    entry = card(service, missing_default, Item.CODEX)
    assert entry["status"] == "needs_attention"
    assert "catalog/default" in entry["message"]
    assert "setup" not in entry["actions"]


def test_codex_wsl_extension_badge_explains_manual_scope_and_native_app_can_setup(
    setup,
):
    service, locator, ctx = setup
    write_json(
        locator.vscode_settings, {"chatgpt.runCodexInWindowsSubsystemForLinux": True}
    )
    entry = card(service, ctx, Item.CODEX)
    assert any("WSL" in badge for badge in entry["badges"])
    perform(service, ctx, Item.CODEX, Action.SETUP)
    assert (
        json.loads(locator.vscode_settings.read_bytes())[
            "chatgpt.runCodexInWindowsSubsystemForLinux"
        ]
        is True
    )


def test_symlink_state_directory_does_not_even_create_a_lock_in_another_location(setup):
    service, locator, ctx = setup
    other = locator.home / "other-state"
    other.mkdir()
    try:
        service.state_dir.symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this host")
    with pytest.raises(IntegrationError):
        service.apply(Item.CLAUDE_LOGIN, Action.REPAIR, "a" * 64, ctx)
    assert list(other.iterdir()) == []


def test_editor_write_during_temporary_file_flush_is_preserved(setup, monkeypatch):
    service, locator, ctx = setup
    actual_write = service_module.atomic_write
    actual_fsync = storage_module.os.fsync
    later = {"editor.fontSize": 24}

    def interfere_during_write(path, data, **kwargs):
        if path != locator.vscode_settings:
            return actual_write(path, data, **kwargs)

        def fsync_then_edit(descriptor):
            actual_fsync(descriptor)
            write_json(path, later)

        with monkeypatch.context() as patch:
            patch.setattr(storage_module.os, "fsync", fsync_then_edit)
            return actual_write(path, data, **kwargs)

    monkeypatch.setattr(service_module, "atomic_write", interfere_during_write)
    with pytest.raises(IntegrationError, match="changed since this preview"):
        perform(service, ctx, Item.CLAUDE_VSCODE, Action.SETUP)
    assert json.loads(locator.vscode_settings.read_bytes()) == later
