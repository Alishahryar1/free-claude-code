"""Browser confirmation flows exercise only fixture client files."""

import json
import tomllib

import pytest
from playwright.sync_api import expect

from free_claude_code.application.integrations import IntegrationAction as Action
from free_claude_code.application.integrations import IntegrationId as Item
from free_claude_code.runtime.integrations.service import IntegrationService
from tests.integration_support import connection, write_json


def open_integrations(page, base):
    page.goto(f"{base}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(page.locator(".integration-card")).to_have_count(4)


def item_card(page, item):
    return page.locator(f'[data-integration="{item}"]')


def confirm(page, card, action):
    card.get_by_role("button", name=action, exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name="Confirm", exact=True).click()
    expect(dialog).to_have_count(0)


def test_four_items_preserve_unsaved_admin_fields(page, admin_base_url):
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    field = page.locator("#field-NVIDIA_NIM_API_KEY")
    field.fill("unsaved-draft")
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(page.locator(".integration-card")).to_have_count(4)
    expect(page.locator(".action-bar")).to_be_hidden()
    expect(item_card(page, "codex")).to_contain_text("App installed")
    expect(item_card(page, "codex")).to_contain_text("VS Code extension installed")
    expect(item_card(page, "claude-login")).to_contain_text(
        "Fix Claude Code login prompt"
    )
    page.get_by_role("button", name="Providers", exact=True).click()
    expect(field).to_have_value("unsaved-draft")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")


def test_setup_cancel_confirm_disconnect(
    page, admin_base_url, integration_installations
):
    locator = integration_installations
    original = {"editor.fontSize": 16, "claudeCode.disableLoginPrompt": False}
    write_json(locator.vscode_settings, original)
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    setup = card.get_by_role("button", name="Set up", exact=True)
    setup.click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text(str(locator.vscode_settings))
    expect(dialog).not_to_contain_text("e2e-proxy-token")
    expect(dialog).not_to_contain_text("ANTHROPIC_BASE_URL")
    expect(dialog).to_contain_text("Disconnect")
    expect(dialog.get_by_role("heading")).to_be_in_viewport()
    expect(dialog.get_by_role("button", name="Confirm", exact=True)).to_be_in_viewport()
    expect(dialog.get_by_role("button", name="Cancel", exact=True)).to_be_focused()
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)
    expect(setup).to_be_focused()
    assert json.loads(locator.vscode_settings.read_bytes()) == original
    confirm(page, card, "Set up")
    expect(card).to_contain_text("Configured")
    expect(page.locator("#integrationMessage")).to_contain_text("Reload Window")
    confirm(page, card, "Disconnect")
    assert json.loads(locator.vscode_settings.read_bytes()) == original


def test_explicit_update_of_saved_connection_from_previous_server_settings(
    page, admin_base_url, integration_installations
):
    locator = integration_installations
    service = IntegrationService(locator)
    # Model the client file and undo record left by an earlier FCC server.
    old = connection(locator.home, port=8182, token="previous-fcc-token")
    preview = service.preview(Item.CLAUDE_VSCODE, Action.SETUP, old)
    revision = preview["revision"]
    assert isinstance(revision, str)
    service.apply(Item.CLAUDE_VSCODE, Action.SETUP, revision, old)
    before = locator.vscode_settings.read_bytes()
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    expect(card.get_by_role("button", name="Update", exact=True)).to_be_visible()
    assert locator.vscode_settings.read_bytes() == before
    confirm(page, card, "Update")
    assert "8182" not in locator.vscode_settings.read_text()
    assert "e2e-proxy-token" in locator.vscode_settings.read_text()
    confirm(page, card, "Disconnect")
    assert json.loads(locator.vscode_settings.read_bytes()) == {}


@pytest.mark.parametrize(
    "admin_base_url", [{"MODEL": "open_router/vendor/model-a"}], indirect=True
)
def test_shared_codex_config_and_restart_instructions(
    page, admin_base_url, integration_installations
):
    open_integrations(page, admin_base_url)
    card = item_card(page, "codex")
    confirm(page, card, "Set up")
    expect(page.locator("#integrationMessage")).to_contain_text("normal Codex CLI")
    path = integration_installations.home / ".codex/config.toml"
    assert tomllib.loads(path.read_text())["model_provider"] == "fcc"
    confirm(page, card, "Disconnect")
    assert "model_provider" not in tomllib.loads(path.read_text())


def test_stale_preview_and_later_edits_are_refused(
    page, admin_base_url, integration_installations
):
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    card.get_by_role("button", name="Set up", exact=True).click()
    expect(page.get_by_role("dialog")).to_be_visible()
    write_json(integration_installations.vscode_settings, {"editor.fontSize": 18})
    page.get_by_role("dialog").get_by_role("button", name="Confirm", exact=True).click()
    expect(page.locator("#integrationMessage")).to_contain_text(
        "changed since this preview"
    )
    assert json.loads(integration_installations.vscode_settings.read_bytes()) == {
        "editor.fontSize": 18
    }
    confirm(page, card, "Set up")
    data = json.loads(integration_installations.vscode_settings.read_bytes())
    data["claudeCode.environmentVariables"][0]["value"] = "https://user.example"
    write_json(integration_installations.vscode_settings, data)
    card.get_by_role("button", name="Disconnect", exact=True).click()
    expect(page.locator("#integrationMessage")).to_contain_text("edited after FCC")
    expect(page.get_by_role("dialog")).to_have_count(0)
    assert json.loads(integration_installations.vscode_settings.read_bytes()) == data


@pytest.mark.parametrize(
    "original", [None, {"hasCompletedOnboarding": False, "private": "do-not-show"}]
)
def test_separate_login_repair(
    page, admin_base_url, integration_installations, original
):
    path = integration_installations.home / ".claude.json"
    if original is not None:
        write_json(path, original)
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-login")
    confirm(page, card, "Fix")
    expect(card).to_contain_text("Onboarding already completed")
    expect(card.get_by_role("button", name="Fix", exact=True)).to_have_count(0)
    expect(page.locator("#integrationMessage")).to_contain_text(
        "Onboarding setting saved"
    )
    assert json.loads(path.read_bytes()) == {
        **(original or {}),
        "hasCompletedOnboarding": True,
    }
    assert "do-not-show" not in page.locator("#view-integrations").inner_text()
    confirm(page, item_card(page, "claude-vscode"), "Set up")
    confirm(page, item_card(page, "claude-vscode"), "Disconnect")
    assert json.loads(path.read_bytes())["hasCompletedOnboarding"] is True


@pytest.mark.parametrize(
    "content",
    [b'{"bad":', b'{"hasCompletedOnboarding":false,"hasCompletedOnboarding":true}'],
)
def test_invalid_repair_file_shows_attention_without_write(
    page, admin_base_url, integration_installations, content
):
    path = integration_installations.home / ".claude.json"
    path.write_bytes(content)
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-login")
    expect(card).to_contain_text("Needs attention")
    expect(card.get_by_role("button", name="Fix", exact=True)).to_have_count(0)
    assert path.read_bytes() == content


def test_missing_adapter_shows_manual_prerequisites(
    page, admin_base_url, integration_installations
):
    (
        integration_installations.home
        / "lib/node_modules/@agentclientprotocol/claude-agent-acp/package.json"
    ).unlink()
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-jetbrains")
    expect(card).to_contain_text("Install the Claude ACP adapter")
    expect(card.get_by_role("button", name="Set up", exact=True)).to_have_count(0)
    expect(card.get_by_role("link", name="Manual setup", exact=True)).to_be_visible()


def test_repair_cancel_and_stale_confirmation_preserve_shared_state(
    page, admin_base_url, integration_installations
):
    path = integration_installations.home / ".claude.json"
    write_json(path, {"hasCompletedOnboarding": False, "private": "do-not-show"})
    before = path.read_bytes()
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-login")
    card.get_by_role("button", name="Fix", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Cancel", exact=True).click()
    assert path.read_bytes() == before
    card.get_by_role("button", name="Fix", exact=True).click()
    expect(page.get_by_role("dialog")).to_be_visible()
    write_json(path, {"hasCompletedOnboarding": False, "user": "new state"})
    changed = path.read_bytes()
    page.get_by_role("dialog").get_by_role("button", name="Confirm", exact=True).click()
    expect(page.locator("#integrationMessage")).to_contain_text(
        "changed since this preview"
    )
    assert path.read_bytes() == changed
    confirm(page, card, "Fix")
    assert json.loads(path.read_bytes()) == {
        "hasCompletedOnboarding": True,
        "user": "new state",
    }


def test_manual_disconnect_describes_unknown_history_and_keeps_other_settings(
    page, admin_base_url, integration_installations
):
    path = integration_installations.vscode_settings
    write_json(
        path,
        {
            "editor.fontSize": 16,
            "claudeCode.environmentVariables": [
                {"name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082"},
                {"name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "value": "1"},
                {"name": "ANTHROPIC_AUTH_TOKEN", "value": "old-manual-secret"},
            ],
        },
    )
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    expect(card).to_contain_text("Manual FCC setup detected")
    card.get_by_role("button", name="Disconnect", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("Previous settings are unknown")
    expect(dialog).not_to_contain_text("old-manual-secret")
    dialog.get_by_role("button", name="Confirm", exact=True).click()
    expect(dialog).to_have_count(0)
    assert json.loads(path.read_bytes()) == {
        "editor.fontSize": 16,
        "claudeCode.environmentVariables": [],
    }


@pytest.mark.parametrize(
    "admin_base_url", [{"MODEL": "open_router/vendor/model-a"}], indirect=True
)
def test_preview_shows_only_the_file_and_a_short_summary(
    page, admin_base_url, integration_installations
):
    path = integration_installations.home / ".codex/config.toml"
    path.parent.mkdir()
    text = '<img src=x onerror="window.previewExecuted=true">'
    path.write_text(f"model='{text}'\n")
    open_integrations(page, admin_base_url)
    item_card(page, "codex").get_by_role("button", name="Set up", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog).not_to_contain_text("onerror")
    expect(dialog).not_to_contain_text("model_provider")
    expect(dialog).to_contain_text(str(path))
    expect(dialog).to_contain_text("Disconnect")
    expect(dialog.locator("img")).to_have_count(0)
    assert page.evaluate("window.previewExecuted === undefined")
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert tomllib.loads(path.read_text())["model"] == text
