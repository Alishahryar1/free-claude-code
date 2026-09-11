"""Browser confirmation flows exercise only fixture client files."""

import json
import tomllib

import pytest
from playwright.sync_api import expect

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


def test_apply_cancel_confirm_reapply(page, admin_base_url, integration_installations):
    locator = integration_installations
    original = {"editor.fontSize": 16, "claudeCode.disableLoginPrompt": False}
    write_json(locator.vscode_settings, original)
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    setup = card.get_by_role("button", name="Apply FCC settings", exact=True)
    setup.click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text(str(locator.vscode_settings))
    expect(dialog).not_to_contain_text("e2e-proxy-token")
    expect(dialog).not_to_contain_text("ANTHROPIC_BASE_URL")
    expect(dialog).to_contain_text("Manual disconnect instructions are on the card")
    expect(dialog).to_contain_text("Save and close VS Code")
    expect(dialog.get_by_role("heading")).to_be_in_viewport()
    expect(dialog.get_by_role("button", name="Confirm", exact=True)).to_be_in_viewport()
    expect(dialog.get_by_role("button", name="Cancel", exact=True)).to_be_focused()
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)
    expect(setup).to_be_focused()
    assert json.loads(locator.vscode_settings.read_bytes()) == original
    confirm(page, card, "Apply FCC settings")
    expect(card).to_contain_text("Configured")
    expect(page.locator("#integrationMessage")).to_contain_text("Reload Window")
    before = locator.vscode_settings.read_bytes()
    confirm(page, card, "Apply FCC settings")
    expect(page.locator("#integrationMessage")).to_contain_text(
        "No client settings changed"
    )
    assert locator.vscode_settings.read_bytes() == before


def test_explicit_reapply_of_connection_from_previous_server_settings(
    page, admin_base_url, integration_installations
):
    locator = integration_installations
    service = IntegrationService(locator)
    # Model a client file left by an earlier FCC server.
    old = connection(locator.home, port=8182, token="previous-fcc-token")
    preview = service.preview(Item.CLAUDE_VSCODE, old)
    revision = preview["revision"]
    assert isinstance(revision, str)
    service.apply(Item.CLAUDE_VSCODE, revision, old)
    before = locator.vscode_settings.read_bytes()
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    expect(
        card.get_by_role("button", name="Apply FCC settings", exact=True)
    ).to_be_visible()
    assert locator.vscode_settings.read_bytes() == before
    confirm(page, card, "Apply FCC settings")
    assert "8182" not in locator.vscode_settings.read_text()
    assert "e2e-proxy-token" in locator.vscode_settings.read_text()
    expect(card.get_by_role("button", name="Disconnect", exact=True)).to_have_count(0)


@pytest.mark.parametrize(
    "admin_base_url", [{"MODEL": "open_router/vendor/model-a"}], indirect=True
)
def test_shared_codex_config_and_restart_instructions(
    page, admin_base_url, integration_installations
):
    open_integrations(page, admin_base_url)
    card = item_card(page, "codex")
    confirm(page, card, "Apply FCC settings")
    expect(page.locator("#integrationMessage")).to_contain_text("normal Codex CLI")
    path = integration_installations.home / ".codex/config.toml"
    assert tomllib.loads(path.read_text())["model_provider"] == "fcc"
    expect(card.locator("summary")).to_have_text("How to disconnect")


def test_stale_preview_is_refused_and_fresh_apply_replaces_fcc_edits(
    page, admin_base_url, integration_installations
):
    open_integrations(page, admin_base_url)
    card = item_card(page, "claude-vscode")
    card.get_by_role("button", name="Apply FCC settings", exact=True).click()
    expect(page.get_by_role("dialog")).to_be_visible()
    write_json(integration_installations.vscode_settings, {"editor.fontSize": 18})
    page.get_by_role("dialog").get_by_role("button", name="Confirm", exact=True).click()
    expect(page.locator("#integrationMessage")).to_contain_text(
        "changed since this preview"
    )
    assert json.loads(integration_installations.vscode_settings.read_bytes()) == {
        "editor.fontSize": 18
    }
    confirm(page, card, "Apply FCC settings")
    data = json.loads(integration_installations.vscode_settings.read_bytes())
    data["claudeCode.environmentVariables"][0]["value"] = "https://user.example"
    write_json(integration_installations.vscode_settings, data)
    confirm(page, card, "Apply FCC settings")
    saved = json.loads(integration_installations.vscode_settings.read_bytes())
    assert saved["editor.fontSize"] == 18
    assert (
        "https://user.example"
        not in integration_installations.vscode_settings.read_text()
    )


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
    expect(card.get_by_role("button", name="Fix", exact=True)).to_be_visible()
    expect(card.locator("details")).to_have_count(0)
    expect(page.locator("#integrationMessage")).to_contain_text(
        "Onboarding setting saved"
    )
    assert json.loads(path.read_bytes()) == {
        **(original or {}),
        "hasCompletedOnboarding": True,
    }
    assert "do-not-show" not in page.locator("#view-integrations").inner_text()
    confirm(page, item_card(page, "claude-vscode"), "Apply FCC settings")
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
    expect(
        card.get_by_role("button", name="Apply FCC settings", exact=True)
    ).to_have_count(0)
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
    item_card(page, "codex").get_by_role(
        "button", name="Apply FCC settings", exact=True
    ).click()
    dialog = page.get_by_role("dialog")
    expect(dialog).not_to_contain_text("onerror")
    expect(dialog).not_to_contain_text("model_provider")
    expect(dialog).to_contain_text(str(path))
    expect(dialog).to_contain_text("Manual disconnect instructions are on the card")
    expect(dialog).to_contain_text("Save and close")
    expect(dialog).to_contain_text("Codex App, VS Code, and the CLI")
    expect(dialog.locator("img")).to_have_count(0)
    assert page.evaluate("window.previewExecuted === undefined")
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert tomllib.loads(path.read_text())["model"] == text


@pytest.mark.parametrize(
    "viewport", [{"width": 1280, "height": 900}, {"width": 390, "height": 844}]
)
def test_manual_disconnect_instructions_expand_without_requests_or_writes(
    page, admin_base_url, integration_installations, viewport
):
    locator = integration_installations
    write_json(locator.vscode_settings, {"editor.fontSize": 16})
    page.set_viewport_size(viewport)
    open_integrations(page, admin_base_url)
    paths = [
        locator.vscode_settings,
        locator.home / ".codex/config.toml",
        locator.home / ".jetbrains/acp.json",
        locator.home / ".claude.json",
    ]
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    requests = []
    page.on("request", lambda request: requests.append(request))
    for item, expected in [
        (
            "claude-vscode",
            [
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
                "disableLoginPrompt",
                "global/project",
            ],
        ),
        (
            "codex",
            [
                "model_provider",
                "model_providers",
                "model_catalog_json",
                "inline TOML",
                "normal authentication",
            ],
        ),
        (
            "claude-jetbrains",
            ['agent_servers["Claude Code (FCC)"]', "custom fields", "other agents"],
        ),
    ]:
        card = item_card(page, item)
        help = card.locator("details")
        expect(help).not_to_have_attribute("open", "")
        help.locator("summary").click()
        expect(help).to_have_attribute("open", "")
        for text in expected:
            expect(help).to_contain_text(text)
        expect(card.get_by_role("button", name="Disconnect", exact=True)).to_have_count(
            0
        )
        assert card.evaluate("el => el.scrollWidth <= el.clientWidth")
    assert requests == []
    assert {
        path: path.read_bytes() if path.exists() else None for path in paths
    } == before
    card = item_card(page, "claude-vscode")
    card.get_by_role("button", name="Apply FCC settings", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("heading")).to_be_in_viewport()
    expect(dialog.get_by_role("button", name="Confirm", exact=True)).to_be_in_viewport()
    expect(dialog.locator("details")).to_have_count(0)
    dialog.get_by_role("button", name="Cancel", exact=True).click()
