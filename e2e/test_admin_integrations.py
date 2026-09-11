import json

from playwright.sync_api import expect

from free_claude_code.cli import vscode


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


def test_manual_setup_and_revisit_read_the_file(page, admin_base_url, tmp_path):
    path = tmp_path / "vscode" / "settings.json"
    status = page.request.get(f"{admin_base_url}/admin/api/status").json()
    vscode.configure(
        path, f"http://localhost:{status['port']}/", "e2e-proxy-token", True
    )
    page.goto(f"{admin_base_url}/admin/integrations")
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
