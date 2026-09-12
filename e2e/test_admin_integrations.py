import json
import tomllib

import pytest
from playwright.sync_api import expect

from free_claude_code.cli import vscode


@pytest.mark.parametrize("width", [1280, 390])
def test_codex_connect_disconnect_and_modal_paths(
    page, admin_base_url, tmp_path, width
):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{admin_base_url}/admin/integrations")
    expect(page.locator("#openClaudeIntegration")).to_be_enabled()
    expect(page.locator("#messageArea")).to_have_text("")
    cards = page.locator("#view-integrations > article")
    expect(cards).to_have_count(2)
    expect(cards.nth(1)).to_contain_text(
        "Use FCC's models in the Codex CLI, VS Code extension, and desktop app."
    )
    opener = page.locator("#openCodexIntegration")
    dialog = page.get_by_role("dialog", name="Codex", exact=True)
    opener.click()
    expect(dialog).to_be_visible()
    expect(page.locator("#claudeIntegrationDialog")).not_to_be_visible()
    expect(dialog.get_by_role("button", name="Close", exact=True)).to_be_focused()
    expect(dialog).to_contain_text(
        "Configure Codex to use FCC. Your selected model stays unchanged."
    )
    assert dialog.evaluate("element => element.scrollWidth <= element.clientWidth")
    path = tmp_path / ".codex" / "config.toml"
    expect(dialog.locator("#codexIntegrationFiles li")).to_have_text(
        [str(path.resolve())]
    )
    page.locator("#codexIntegrationDescription").click()
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name="Close", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_be_focused()
    opener.click()
    page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    opener.click()
    page.mouse.click(1, 1)
    expect(dialog).not_to_be_visible()
    assert not path.exists()
    path.parent.mkdir()
    path.write_text('model = "my-choice" # Keep this\n')
    opener.click()
    page.locator("#confirmCodexIntegration").click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")
    expect(page.locator("#codexIntegrationStatus")).to_have_text("Connected")
    expect(page.locator("#codexIntegrationMessage")).to_have_text(
        "Settings saved. Restart Codex and select an FCC model."
    )
    assert tomllib.loads(path.read_text())["model"] == "my-choice"
    assert "# Keep this" in path.read_text()
    page.reload()
    expect(opener).to_have_text("Disconnect")
    opener.click()
    expect(page.locator("#confirmCodexIntegration")).to_have_text("Disconnect")
    expect(dialog.locator("#codexIntegrationFiles li")).to_have_text(
        [str(path.resolve())]
    )
    page.locator("#confirmCodexIntegration").click()
    expect(opener).to_have_text("Connect")
    assert tomllib.loads(path.read_text()) == {"model": "my-choice"}
    assert not (tmp_path / "vscode" / "settings.json").exists()
    assert not (tmp_path / ".claude.json").exists()


@pytest.mark.parametrize("width", [1280, 390])
def test_modal_shows_files_for_the_selected_action(
    page, admin_base_url, tmp_path, width
):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{admin_base_url}/admin/integrations")
    page.locator("#openClaudeIntegration").click()
    dialog = page.locator("#claudeIntegrationDialog")
    paths = dialog.locator("#claudeIntegrationFiles li")
    expect(paths).to_have_text(
        [
            str((tmp_path / "vscode" / "settings.json").resolve()),
            str((tmp_path / ".claude.json").resolve()),
        ]
    )
    assert dialog.evaluate("element => element.scrollWidth <= element.clientWidth")
    page.locator("#confirmClaudeIntegration").click()
    expect(dialog).not_to_be_visible()
    page.locator("#openClaudeIntegration").click()
    expect(paths).to_have_text([str((tmp_path / "vscode" / "settings.json").resolve())])
    page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()


def test_connect_disconnect_and_modal_dismissal(page, admin_base_url, tmp_path):
    path = tmp_path / "vscode" / "settings.json"
    page.goto(f"{admin_base_url}/admin/integrations")
    card_button = page.locator("#openClaudeIntegration")
    dialog = page.locator("#claudeIntegrationDialog")
    action = page.locator("#confirmClaudeIntegration")
    expect(card_button).to_be_enabled()
    expect(card_button).to_have_text("Connect")
    for dismiss in ("close", "escape", "outside"):
        card_button.click()
        expect(dialog).to_be_visible()
        if dismiss == "close":
            dialog.get_by_role("button", name="Close", exact=True).click()
        elif dismiss == "escape":
            page.keyboard.press("Escape")
        else:
            page.mouse.click(1, 1)
        expect(dialog).not_to_be_visible()
        assert not path.exists()
    card_button.click()
    action.click()
    expect(dialog).not_to_be_visible()
    expect(card_button).to_have_text("Disconnect")
    expect(page.locator("#claudeIntegrationStatus")).to_have_text("Connected")
    expect(page.locator("#claudeIntegrationMessage")).to_contain_text("Reload VS Code")
    assert json.loads(path.read_text())["claudeCode.disableLoginPrompt"] is True
    state_path = tmp_path / ".claude.json"
    assert json.loads(state_path.read_text())["hasCompletedOnboarding"] is True
    page.reload()
    expect(card_button).to_have_text("Disconnect")
    card_button.click()
    expect(action).to_have_text("Disconnect")
    expect(page.locator("#claudeIntegrationDescription")).to_contain_text("Remove")
    page.keyboard.press("Escape")
    assert json.loads(path.read_text())["claudeCode.disableLoginPrompt"] is True
    card_button.click()
    action.click()
    expect(card_button).to_have_text("Connect")
    assert json.loads(path.read_text()) == {}
    assert json.loads(state_path.read_text())["hasCompletedOnboarding"] is True


def test_manual_setup_and_revisit_read_the_file(page, admin_base_url, tmp_path):
    path = tmp_path / "vscode" / "settings.json"
    status = page.request.get(f"{admin_base_url}/admin/api/status").json()
    vscode.configure(
        path,
        tmp_path / ".claude.json",
        f"http://localhost:{status['port']}/",
        "e2e-proxy-token",
        True,
    )
    page.goto(f"{admin_base_url}/admin/integrations")
    expect(page.locator("#openClaudeIntegration")).to_have_text("Disconnect")
    (tmp_path / ".claude.json").unlink()
    page.reload()
    expect(page.locator("#openClaudeIntegration")).to_have_text("Connect")
    page.locator("#openClaudeIntegration").click()
    expect(page.locator("#claudeIntegrationDescription")).to_contain_text("onboarding")
    page.locator("#confirmClaudeIntegration").click()
    expect(page.locator("#openClaudeIntegration")).to_have_text("Disconnect")
    page.get_by_role("button", name="Providers", exact=True).click()
    path.write_text("{}")
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(page.locator("#openClaudeIntegration")).to_have_text("Connect")


def test_invalid_settings_error_can_be_retried_and_does_not_break_admin(
    page, admin_base_url, tmp_path
):
    path = tmp_path / "vscode" / "settings.json"
    path.parent.mkdir()
    path.write_text("{invalid}")
    page.goto(f"{admin_base_url}/admin/integrations")
    expect(page.locator("#claudeIntegrationMessage")).to_contain_text("Check the JSON")
    expect(page.locator("#claudeIntegrationStatus")).to_have_text(
        "Could not check settings"
    )
    page.get_by_role("button", name="Providers", exact=True).click()
    expect(page.locator('[data-provider="nvidia_nim"]')).to_be_visible()
    page.get_by_role("button", name="Integrations", exact=True).click()
    path.write_text("{}")
    page.locator("#openClaudeIntegration").click()
    expect(page.locator("#openClaudeIntegration")).to_have_text("Connect")


def test_save_pending_and_failure_stay_in_modal(page, admin_base_url):
    page.goto(f"{admin_base_url}/admin/integrations")
    page.locator("#openClaudeIntegration").click()
    requests = []
    page.route(
        "**/admin/api/integrations/claude-vscode/connect",
        lambda route: requests.append(route),
    )
    action = page.locator("#confirmClaudeIntegration")
    action.click()
    expect(action).to_be_disabled()
    expect(action).to_have_text("Saving…")
    requests[0].fulfill(status=503, json={"detail": "Could not save settings."})
    expect(action).to_be_enabled()
    expect(page.locator("#claudeIntegrationDialog")).to_be_visible()
    expect(page.locator("#claudeIntegrationDialogMessage")).to_have_text(
        "Could not save settings."
    )


def test_codex_existing_setup_revisit_and_invalid_config_retry(
    page, admin_base_url, tmp_path
):
    path = tmp_path / ".codex" / "config.toml"
    assert page.request.post(
        f"{admin_base_url}/admin/api/integrations/codex/connect"
    ).ok
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openCodexIntegration")
    expect(opener).to_have_text("Disconnect")
    page.get_by_role("button", name="Providers", exact=True).click()
    path.write_text("[invalid")
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(page.locator("#codexIntegrationMessage")).to_contain_text("Check the TOML")
    expect(opener).to_have_text("Retry")
    path.write_text("")
    opener.click()
    expect(opener).to_have_text("Connect")


def test_codex_save_pending_and_failure_stay_in_modal(page, admin_base_url, tmp_path):
    page.goto(f"{admin_base_url}/admin/integrations")
    page.locator("#openCodexIntegration").click()
    requests = []
    page.route(
        "**/admin/api/integrations/codex/connect", lambda route: requests.append(route)
    )
    action = page.locator("#confirmCodexIntegration")
    action.click()
    expect(action).to_be_disabled()
    expect(action).to_have_text("Saving…")
    expect(page.locator("#openCodexIntegration")).to_be_disabled()
    requests[0].fulfill(status=503, json={"detail": "Could not save settings."})
    expect(action).to_be_enabled()
    expect(page.locator("#codexIntegrationDialog")).to_be_visible()
    expect(page.locator("#codexIntegrationDialogMessage")).to_have_text(
        "Could not save settings."
    )
    assert not (tmp_path / ".codex" / "config.toml").exists()
