"""Lifecycle tests use real files, synthetic installations, and no live clients."""

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
import tomlkit
from tomlkit.exceptions import TOMLKitError

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


def perform(service, ctx, item):
    preview = service.preview(item, ctx)
    return service.apply(item, preview["revision"], ctx)


def card(service, ctx, item):
    return next(entry for entry in service.inspect(ctx)["items"] if entry["id"] == item)


def test_reapply_overwrites_edited_connection_leaves_and_keeps_other_settings(setup):
    service, locator, ctx = setup
    perform(service, ctx, Item.CLAUDE_VSCODE)
    document = JsonDocument(locator.vscode_settings.read_bytes(), jsonc=True)
    document.set(("claudeCode.disableLoginPrompt",), "wrong-type")
    document.set(("editor.fontSize",), 19)
    locator.vscode_settings.write_bytes(
        document.render().replace(b"fixture-proxy-token", b"manual-token")
    )
    changed = connection(locator.home, port=8182, token="new-token")
    perform(service, changed, Item.CLAUDE_VSCODE)
    data = json.loads(locator.vscode_settings.read_bytes())
    assert data["editor.fontSize"] == 19
    assert data["claudeCode.disableLoginPrompt"] is True
    environment = {
        entry["name"]: entry["value"]
        for entry in data["claudeCode.environmentVariables"]
    }
    assert environment["ANTHROPIC_AUTH_TOKEN"] == "new-token"
    assert environment["ANTHROPIC_BASE_URL"] == changed.url


def test_apply_ignores_old_journal_and_never_writes_history(setup):
    service, locator, ctx = setup
    journal = locator.home / ".fcc/integrations/claude-vscode.json"
    write_json(journal, {"unreadable": "old development history"})
    original = journal.read_bytes()
    perform(service, ctx, Item.CLAUDE_VSCODE)
    assert journal.read_bytes() == original
    perform(service, ctx, Item.CODEX)
    assert not (journal.parent / "codex.json").exists()


@pytest.mark.parametrize(
    "source,expected",
    [
        ('model_provider="fcc"\nmodel="chosen/model"\n', "chosen/model"),
        ('model_provider="fcc"\n', "test/model"),
        ('model_provider="fcc"\nmodel=""\n', "test/model"),
        ('model_provider="fcc"\nmodel=123\n', "test/model"),
        ('model_provider="other"\nmodel="old/model"\n', "test/model"),
    ],
)
def test_apply_uses_current_provider_to_choose_model(setup, source, expected):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(source)
    perform(service, ctx, Item.CODEX)
    assert tomllib.loads(path.read_text())["model"] == expected


@pytest.mark.parametrize(
    "source", [b"{broken", b'{"chatgpt.runCodexInWindowsSubsystemForLinux":"false"}']
)
def test_unknown_extension_mode_blocks_extension_only_apply(setup, source):
    service, locator, ctx = setup
    (locator.home / "bin/chatgpt").unlink()
    locator.vscode_settings.parent.mkdir(parents=True)
    locator.vscode_settings.write_bytes(source)
    with pytest.raises(IntegrationError):
        service.preview(Item.CODEX, ctx)


def test_independent_codex_app_does_not_read_optional_vscode_settings(
    setup, monkeypatch
):
    service, locator, ctx = setup
    installed = replace(locator.scan(), codex_app=True)
    monkeypatch.setattr(type(locator), "scan", lambda _self: installed)
    actual_read = service_module.FileSnapshot.read

    def required_read(path):
        if path == locator.vscode_settings:
            raise IntegrationError("Optional VS Code settings are unreadable")
        return actual_read(path)

    monkeypatch.setattr(service_module.FileSnapshot, "read", required_read)
    perform(service, ctx, Item.CODEX)
    entry = card(service, ctx, Item.CODEX)
    assert entry["can_apply"] is True
    assert entry["status"] == "configured"
    assert str(locator.vscode_settings) in entry["message"]
    assert "unavailable" in entry["message"]


@pytest.mark.parametrize("item", list(Item))
def test_matching_file_is_a_noop_without_atomic_replacement(setup, monkeypatch, item):
    service, locator, ctx = setup
    perform(service, ctx, item)
    preview = service.preview(item, ctx)
    path = Path(preview["path"])
    before = storage_module.FileSnapshot.read(path)
    assert preview["writes_file"] is False
    assert card(service, ctx, item)["status"] == "configured"
    assert card(service, ctx, item)["can_apply"] is True
    monkeypatch.setattr(
        service_module,
        "atomic_write",
        lambda *_args, **_kwargs: pytest.fail("No-op replaced a file"),
    )
    assert perform(service, ctx, item)["applied"] is False
    assert storage_module.FileSnapshot.read(path) == before
    assert not (locator.home / f".fcc/integrations/{item}.json").exists()


@pytest.mark.parametrize("has_app", [True, False])
def test_only_required_vscode_changes_stale_codex_confirmation(setup, has_app):
    service, locator, ctx = setup
    if not has_app:
        (locator.home / "bin/chatgpt").unlink()
    write_json(locator.vscode_settings, {"editor.fontSize": 16})
    preview = service.preview(Item.CODEX, ctx)
    write_json(locator.vscode_settings, {"editor.fontSize": 20})
    if has_app:
        assert service.apply(Item.CODEX, preview["revision"], ctx)["applied"] is True
    else:
        with pytest.raises(IntegrationError, match="changed since this preview"):
            service.apply(Item.CODEX, preview["revision"], ctx)
        assert not (locator.home / ".codex/config.toml").exists()


@pytest.mark.parametrize(
    "source", [None, b"{}", b'{"chatgpt.runCodexInWindowsSubsystemForLinux":false}']
)
def test_extension_only_native_evidence_allows_apply(setup, source):
    service, locator, ctx = setup
    (locator.home / "bin/chatgpt").unlink()
    if source is not None:
        locator.vscode_settings.parent.mkdir(parents=True)
        locator.vscode_settings.write_bytes(source)
    assert perform(service, ctx, Item.CODEX)["applied"] is True


def test_extension_only_unreadable_settings_never_imply_native_support(
    setup, monkeypatch
):
    service, locator, ctx = setup
    (locator.home / "bin/chatgpt").unlink()
    actual_read = service_module.FileSnapshot.read

    def unreadable_settings(path):
        if path == locator.vscode_settings:
            raise IntegrationError("Could not read settings")
        return actual_read(path)

    monkeypatch.setattr(service_module.FileSnapshot, "read", unreadable_settings)
    entry = card(service, ctx, Item.CODEX)
    assert entry["can_apply"] is False
    assert str(locator.vscode_settings) in entry["message"]
    with pytest.raises(IntegrationError, match="Could not read"):
        service.preview(Item.CODEX, ctx)


def test_app_with_malformed_optional_settings_can_reapply(setup):
    service, locator, ctx = setup
    assert locator.scan().codex_app is True
    locator.vscode_settings.parent.mkdir(parents=True)
    locator.vscode_settings.write_bytes(b'{"editor.fontSize":}')
    preview = service.preview(Item.CODEX, ctx)
    locator.vscode_settings.write_bytes(b'{"different-malformed":}')
    assert service.apply(Item.CODEX, preview["revision"], ctx)["applied"] is True
    assert card(service, ctx, Item.CODEX)["can_apply"] is True


def test_jetbrains_apply_merges_existing_named_agent_without_ownership_gate(setup):
    service, locator, ctx = setup
    path = locator.home / ".jetbrains/acp.json"
    write_json(
        path,
        {
            "agent_servers": {
                "Other": {"command": "keep"},
                "Claude Code (FCC)": {
                    "command": 3,
                    "args": False,
                    "custom": "keep",
                    "env": {"EXTRA": "keep", "ANTHROPIC_AUTH_TOKEN": 123},
                },
            }
        },
    )
    perform(service, ctx, Item.CLAUDE_JETBRAINS)
    agents = json.loads(path.read_bytes())["agent_servers"]
    assert agents["Other"] == {"command": "keep"}
    assert agents["Claude Code (FCC)"]["custom"] == "keep"
    assert agents["Claude Code (FCC)"]["env"]["EXTRA"] == "keep"
    assert (
        agents["Claude Code (FCC)"]["env"]["ANTHROPIC_AUTH_TOKEN"]
        == ctx.settings.proxy_auth_token
    )


@pytest.mark.parametrize(
    "source",
    [
        'model_providers = {fcc={name="Original", env_key="OLD", experimental_bearer_token="SECRET"}, other={name="Keep"}}\n',
        'model_provider="other"\nmodel_providers = {fcc.name="Original", fcc.auth.command="original-helper", fcc.env_key="OLD", fcc.experimental_bearer_token="SECRET", other.name="Keep"}\n',
        'model_providers.fcc.auth.command="old"\nmodel_providers.fcc.env_key="OLD"\nmodel_providers.fcc.experimental_bearer_token="SECRET"\nmodel_providers.other.name="Keep"\n',
        '[model_providers.fcc.auth]\nargs=["old"]\n[model_providers.fcc]\nname="Original"\nenv_key="OLD"\nexperimental_bearer_token="SECRET"\n[model_providers.other]\nname="Keep"\n',
        '[model_providers]\nother.name="Keep"\nfcc.name="Original"\nfcc.auth={command="old",args=["old"]}\nfcc.env_key="OLD"\nfcc.experimental_bearer_token="SECRET"\n',
    ],
)
def test_codex_apply_and_reapply_preserve_other_provider_across_layouts(setup, source):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(source)
    for current in (ctx, connection(locator.home, port=8182)):
        perform(service, current, Item.CODEX)
        data = tomllib.loads(path.read_text())
        assert data["model_providers"]["other"] == {"name": "Keep"}
        fcc = data["model_providers"]["fcc"]
        assert fcc["base_url"] == current.url + "/v1"
        assert "env_key" not in fcc and "experimental_bearer_token" not in fcc
        assert fcc["auth"]["args"] == ["--print-proxy-auth-token"]


@pytest.mark.parametrize("value", ["false", "true", '"incorrect"'])
def test_codex_only_preserves_explicit_false_openai_auth(setup, value):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(f"[model_providers.fcc]\nrequires_openai_auth={value}\n")
    perform(service, ctx, Item.CODEX)
    fcc = tomllib.loads(path.read_text())["model_providers"]["fcc"]
    if value == "false":
        assert fcc["requires_openai_auth"] is False
    else:
        assert "requires_openai_auth" not in fcc


def test_atomic_write_rejects_relative_target_without_creating_a_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(IntegrationError, match="absolute"):
        storage_module.atomic_write(Path("wrong/config.json"), b"{}")
    assert not (tmp_path / "wrong").exists()


def test_inspection_and_preview_do_not_write_configuration_or_ownership(setup):
    service, locator, ctx = setup
    before = set(locator.home.rglob("*"))
    assert len(service.inspect(ctx)["items"]) == 4
    preview = service.preview(Item.CLAUDE_VSCODE, ctx)
    assert preview["writes_file"]
    assert set(locator.home.rglob("*")) == before
    assert ctx.settings.proxy_auth_token not in json.dumps(preview)


def test_vscode_apply_reapply_preserves_unrelated_edits(setup):
    service, locator, ctx = setup
    original = b'{\n // keep\n "editor.fontSize":15,\n "claudeCode.disableLoginPrompt":false,\n "claudeCode.environmentVariables":[{"name":"KEEP","value":"mine"},{"name":"ANTHROPIC_BASE_URL","value":"https://previous.example"}]\n}'
    locator.vscode_settings.parent.mkdir(parents=True)
    locator.vscode_settings.write_bytes(original)
    perform(service, ctx, Item.CLAUDE_VSCODE)
    document = JsonDocument(locator.vscode_settings.read_bytes(), jsonc=True)
    assert document.get(("claudeCode.disableLoginPrompt",)) is True
    document.set(("editor.fontSize",), 20)
    locator.vscode_settings.write_bytes(document.render())
    changed = connection(locator.home, port=8182, token="new-token")
    assert card(service, changed, Item.CLAUDE_VSCODE)["status"] == "not_configured"
    perform(service, changed, Item.CLAUDE_VSCODE)
    restored = JsonDocument(locator.vscode_settings.read_bytes(), jsonc=True)
    entries = restored.get(("claudeCode.environmentVariables",))
    assert isinstance(entries, list)
    assert {"name": "KEEP", "value": "mine"} in entries
    assert b"new-token" in locator.vscode_settings.read_bytes()
    assert restored.get(("claudeCode.disableLoginPrompt",)) is True
    assert restored.get(("editor.fontSize",)) == 20
    assert b"// keep" in locator.vscode_settings.read_bytes()


def test_codex_apply_preserves_other_provider_and_uses_fcc_catalog(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    original = b'# user\nmodel="old"\nmodel_provider="other"\nmodel_catalog_json="custom.json"\n[model_providers.other]\nname="Keep"\n'
    path.write_bytes(original)
    perform(service, ctx, Item.CODEX)
    data = tomllib.loads(path.read_text())
    assert data["model"] == "test/model"
    assert data["model_providers"]["fcc"]["auth"]["command"] == str(
        locator.home / "bin/fcc-codex"
    )
    assert data["model_providers"]["other"]["name"] == "Keep"
    perform(service, connection(locator.home, port=8182), Item.CODEX)
    data = tomllib.loads(path.read_text())
    assert data["model"] == "test/model" and data["model_provider"] == "fcc"
    assert data["model_catalog_json"] == str(ctx.catalog_path)
    assert b"# user" in path.read_bytes()


def test_codex_reapply_keeps_user_model(setup):
    service, locator, ctx = setup
    perform(service, ctx, Item.CODEX)
    path = locator.home / ".codex/config.toml"
    path.write_bytes(path.read_bytes().replace(b'"test/model"', b'"user/model"'))
    next_ctx = connection(locator.home, port=8182)
    perform(service, next_ctx, Item.CODEX)
    assert tomllib.loads(path.read_text())["model"] == "user/model"


@pytest.mark.parametrize("change", ["file", "connection", "dependency"])
def test_stale_preview_cannot_write(setup, change):
    service, locator, ctx = setup
    preview = service.preview(Item.CLAUDE_VSCODE, ctx)
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
        service.apply(Item.CLAUDE_VSCODE, preview["revision"], ctx)
    assert (
        locator.vscode_settings.read_bytes()
        if locator.vscode_settings.exists()
        else None
    ) == before


def test_duplicate_confirmation_is_stale(setup):
    service, locator, ctx = setup
    preview = service.preview(Item.CLAUDE_VSCODE, ctx)
    service.apply(Item.CLAUDE_VSCODE, preview["revision"], ctx)
    before = locator.vscode_settings.read_bytes()
    with pytest.raises(IntegrationError):
        service.apply(Item.CLAUDE_VSCODE, preview["revision"], ctx)
    assert locator.vscode_settings.read_bytes() == before


def test_jetbrains_reapply_preserves_other_agents_and_added_fields(
    setup,
):
    service, locator, ctx = setup
    path = locator.home / ".jetbrains/acp.json"
    write_json(path, {"agent_servers": {"Other": {"command": "keep"}}})
    perform(service, ctx, Item.CLAUDE_JETBRAINS)
    data = json.loads(path.read_bytes())
    assert data["agent_servers"]["Other"] == {"command": "keep"}
    data["agent_servers"]["Claude Code (FCC)"]["custom"] = True
    write_json(path, data)
    perform(
        service,
        connection(locator.home, port=8182),
        Item.CLAUDE_JETBRAINS,
    )
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
    preview = service.preview(Item.CLAUDE_LOGIN, ctx)
    assert "do-not-return" not in json.dumps(preview)
    service.apply(Item.CLAUDE_LOGIN, preview["revision"], ctx)
    assert json.loads(path.read_bytes()) == {
        **(original or {}),
        "hasCompletedOnboarding": True,
    }
    before = path.read_bytes()
    assert service.preview(Item.CLAUDE_LOGIN, ctx)["writes_file"] is False
    perform(service, ctx, Item.CLAUDE_VSCODE)
    assert path.read_bytes() == before


def test_onboarding_replaces_incorrect_leaf_type(setup):
    service, locator, ctx = setup
    for flag in ("false", 1):
        write_json(
            locator.home / ".claude.json",
            {"hasCompletedOnboarding": flag, "keep": "private"},
        )
        perform(service, ctx, Item.CLAUDE_LOGIN)
        assert json.loads((locator.home / ".claude.json").read_bytes()) == {
            "hasCompletedOnboarding": True,
            "keep": "private",
        }


def test_missing_client_and_catalog_do_not_trigger_installation_or_fallback_model(
    setup,
):
    service, locator, ctx = setup
    (locator.home / "bin/fcc-codex").unlink()
    with pytest.raises(IntegrationError):
        service.preview(Item.CODEX, ctx)
    with pytest.raises(IntegrationError):
        service.preview(
            Item.CODEX,
            replace(ctx, models=()),
        )


def test_codex_auth_cleanup_is_masked_and_does_not_touch_other_providers(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        '[model_providers.fcc]\nname="Old"\nenv_key="PRIVATE_ENV"\nexperimental_bearer_token="SECRET_BEARER"\nrequires_openai_auth=true\n[model_providers.other]\nname="Keep"\n'
    )
    preview = service.preview(Item.CODEX, ctx)
    assert "SECRET_BEARER" not in json.dumps(preview)
    service.apply(Item.CODEX, preview["revision"], ctx)
    assert (
        "experimental_bearer_token"
        not in tomllib.loads(path.read_text())["model_providers"]["fcc"]
    )
    assert tomllib.loads(path.read_text())["model_providers"]["other"] == {
        "name": "Keep"
    }


def test_codex_token_rotation_does_not_offer_update_but_known_auth_mismatch_blocks_setup(
    setup,
):
    service, locator, ctx = setup
    perform(service, ctx, Item.CODEX)
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
        other.preview(Item.CODEX, bad)


def test_known_conflicting_claude_settings_are_reported_without_modification(setup):
    service, locator, ctx = setup
    path = locator.home / ".claude/settings.json"
    write_json(path, {"env": {"ANTHROPIC_BASE_URL": "https://other.example"}})
    before = path.read_bytes()
    with pytest.raises(IntegrationError):
        service.preview(Item.CLAUDE_VSCODE, ctx)
    assert path.read_bytes() == before


def test_read_only_or_malformed_file_is_never_overwritten(setup):
    service, locator, ctx = setup
    path = locator.home / ".claude.json"
    path.write_bytes(b'{"broken":')
    with pytest.raises(IntegrationError):
        service.preview(Item.CLAUDE_LOGIN, ctx)
    path.write_bytes(b"{}")
    path.chmod(0o400)
    try:
        with pytest.raises(IntegrationError):
            perform(service, ctx, Item.CLAUDE_LOGIN)
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
        service.preview(Item.CLAUDE_LOGIN, ctx)
    assert target.read_bytes() == b"{}"


@pytest.mark.parametrize("stage", ["before_replace", "after_replace"])
def test_failed_write_can_be_retried_from_actual_file(setup, monkeypatch, stage):
    service, locator, ctx = setup
    original = {"claudeCode.disableLoginPrompt": False, "editor.fontSize": 16}
    write_json(locator.vscode_settings, original)
    actual_write = service_module.atomic_write

    def fail_write(path, data, **kwargs):
        if stage == "before_replace":
            raise OSError("private error must not escape")
        actual_write(path, data, **kwargs)
        raise OSError("private error must not escape")

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "atomic_write", fail_write)
        with pytest.raises(IntegrationError, match="Could not save"):
            perform(service, ctx, Item.CLAUDE_VSCODE)
    data = json.loads(locator.vscode_settings.read_bytes())
    if stage == "before_replace":
        assert data == original
    else:
        assert data["claudeCode.disableLoginPrompt"] is True
        assert data["editor.fontSize"] == 16
    result = perform(service, ctx, Item.CLAUDE_VSCODE)
    assert result["applied"] is (stage == "before_replace")
    assert not (locator.home / ".fcc/integrations/claude-vscode.json").exists()


def test_interference_after_replace_is_reported_and_preserved(setup, monkeypatch):
    service, locator, ctx = setup
    actual_write = service_module.atomic_write

    def interfere(path, data, **kwargs):
        actual_write(path, data, **kwargs)
        if path == locator.vscode_settings:
            write_json(path, {"user": "later edit"})

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "atomic_write", interfere)
        with pytest.raises(IntegrationError, match="verification"):
            perform(service, ctx, Item.CLAUDE_VSCODE)
    assert card(service, ctx, Item.CLAUDE_VSCODE)["can_apply"] is True
    assert json.loads(locator.vscode_settings.read_bytes()) == {"user": "later edit"}
    assert not (locator.home / ".fcc/integrations/claude-vscode.json").exists()


def test_concurrent_confirmations_across_services_have_one_winner(setup):
    service, locator, ctx = setup
    other = IntegrationService(locator)
    barrier = Barrier(2)

    def confirm(owner):
        preview = owner.preview(Item.CLAUDE_VSCODE, ctx)
        barrier.wait(timeout=5)
        try:
            return owner.apply(Item.CLAUDE_VSCODE, preview["revision"], ctx)["applied"]
        except IntegrationError as error:
            assert error.status_code == 409
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(confirm, [service, other])) == [False, True]
    assert (
        json.loads(locator.vscode_settings.read_bytes())[
            "claudeCode.disableLoginPrompt"
        ]
        is True
    )


def test_missing_codex_default_is_explained_on_the_card(setup):
    service, _locator, ctx = setup
    missing_default = replace(ctx, models=())
    entry = card(service, missing_default, Item.CODEX)
    assert entry["status"] == "needs_attention"
    assert "catalog/default" in entry["message"]
    assert entry["can_apply"] is False


def test_codex_wsl_extension_badge_explains_manual_scope_and_native_app_can_setup(
    setup,
):
    service, locator, ctx = setup
    write_json(
        locator.vscode_settings, {"chatgpt.runCodexInWindowsSubsystemForLinux": True}
    )
    entry = card(service, ctx, Item.CODEX)
    assert any("WSL" in badge for badge in entry["badges"])
    perform(service, ctx, Item.CODEX)
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
        (locator.home / ".fcc/integrations").symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this host")
    with pytest.raises(IntegrationError):
        service.apply(Item.CLAUDE_LOGIN, "a" * 64, ctx)
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
        perform(service, ctx, Item.CLAUDE_VSCODE)
    assert json.loads(locator.vscode_settings.read_bytes()) == later


def test_codex_setup_preserves_unrelated_dotted_provider(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        'model_providers.fcc.name="Old"\nmodel_providers.other.name="Other"\n'
    )
    perform(service, ctx, Item.CODEX)
    actual = tomllib.loads(path.read_text())
    assert actual["model_providers"].get("other") == {"name": "Other"}
    assert actual.get("model_provider") == "fcc"
    assert actual["model_providers"]["fcc"]["base_url"] == "http://127.0.0.1:8082/v1"


def test_inline_fcc_provider_without_auth_keeps_all_cards_available(setup):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        'model_provider="fcc"\n[model_providers]\n'
        'fcc={name="Free Claude Code",base_url="http://localhost:8082/v1"}\n'
        'other={name="Other"}\n'
    )
    result = service.inspect(ctx)
    assert len(result["items"]) == 4
    assert card(service, ctx, Item.CODEX)["status"] == "not_configured"


@pytest.mark.parametrize(
    "changed", ["requirement", "manifest", "version", "entrypoint", "runtime"]
)
def test_acp_dependency_changes_reject_the_old_confirmation(
    setup, monkeypatch, changed
):
    service, locator, ctx = setup
    preview = service.preview(Item.CLAUDE_JETBRAINS, ctx)
    package = locator.home / "lib/node_modules/@agentclientprotocol/claude-agent-acp"
    if changed in {"requirement", "manifest"}:
        manifest = json.loads((package / "package.json").read_bytes())
        if changed == "requirement":
            manifest["engines"]["node"] = ">=24"
        else:
            manifest["version"] = "next-fixture-version"
        write_json(package / "package.json", manifest)
    elif changed == "version":
        monkeypatch.setattr(
            type(locator), "_node_version", lambda _self, _node: (24, 0, 0)
        )
    else:
        path = (
            package / "dist/index.js"
            if changed == "entrypoint"
            else locator.home / "bin/node"
        )
        path.write_text("replaced dependency")
    with pytest.raises(IntegrationError):
        service.apply(Item.CLAUDE_JETBRAINS, preview["revision"], ctx)
    assert not (locator.home / ".jetbrains/acp.json").exists()
    assert not ((locator.home / ".fcc/integrations") / "claude-jetbrains.json").exists()


@pytest.mark.parametrize("failure", ["semantic", "value_error", "toml_error"])
def test_document_failure_is_isolated_before_client_or_history_writes(
    setup, monkeypatch, failure
):
    service, locator, ctx = setup
    path = locator.home / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        'keep=true\nmodel_provider="fcc"\n[model_providers.fcc]\nname="Free Claude Code"\n'
    )
    before = storage_module.FileSnapshot.read(path)
    actual_dumps = tomlkit.dumps

    def fail_mutated_candidate(candidate):
        text = actual_dumps(candidate)
        if "auth" in candidate["model_providers"]["fcc"]:
            if failure == "value_error":
                raise ValueError("private source contents")
            if failure == "toml_error":
                raise TOMLKitError("private source contents")
            return text.replace("keep=true", "keep=1")
        return text

    monkeypatch.setattr(tomlkit, "dumps", fail_mutated_candidate)
    inspected = service.inspect(ctx)
    assert len(inspected["items"]) == 4
    assert card(service, ctx, Item.CODEX)["status"] == "needs_attention"
    assert "private source contents" not in json.dumps(inspected)
    with pytest.raises(IntegrationError):
        service.preview(Item.CODEX, ctx)
    assert storage_module.FileSnapshot.read(path) == before
    assert not ((locator.home / ".fcc/integrations") / "codex.json").exists()
