import json

from playwright.sync_api import expect

from free_claude_code.harnesses import devin_acp_integration as devin


def test_connect_retry_disconnect_and_modal_dismissal(
    page, admin_base_url, monkeypatch
):
    def missing():
        raise devin.SetupError("Install OpenCode 2 and retry.")

    monkeypatch.setattr(devin, "_launch", missing)
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openDevinIntegration")
    dialog = page.locator("#devinIntegrationDialog")
    action = page.locator("#confirmDevinIntegration")
    expect(opener).to_have_text("Connect")
    for dismiss in ("close", "escape", "outside"):
        opener.click()
        expect(dialog).to_be_visible()
        expect(page.locator("#devinIntegrationFiles li")).to_have_text(
            [str(devin.config_path().resolve())]
        )
        page.locator("#devinIntegrationDescription").click()
        expect(dialog).to_be_visible()
        if dismiss == "close":
            page.locator("#closeDevinIntegration").click()
        elif dismiss == "escape":
            page.keyboard.press("Escape")
        else:
            page.mouse.click(1, 1)
        expect(dialog).not_to_be_visible()
    opener.click()
    action.click()
    expect(page.locator("#devinIntegrationDialogMessage")).to_be_visible()
    assert not devin.config_path().exists()
    monkeypatch.setattr(devin, "_launch", lambda: ("fcc-opencode", "test-path"))
    action.click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")
    entry = json.loads(devin.config_path().read_text())["agents"][0]
    assert entry["id"] == "fcc-opencode"
    page.reload()
    expect(opener).to_have_text("Disconnect")
    expect(opener).to_have_css("color", "rgb(239, 68, 68)")
    monkeypatch.setattr(devin, "_launch", missing)
    assert page.request.post(
        f"{admin_base_url}/admin/api/integrations/devin-acp/refresh"
    ).ok
    page.reload()
    expect(page.locator("#devinIntegrationMessage")).to_be_visible()
    expect(opener).to_be_enabled()
    opener.click()
    action.click()
    expect(opener).to_have_text("Connect")
    assert json.loads(devin.config_path().read_text())["agents"] == []


def test_unreadable_registry_can_be_retried(page, admin_base_url):
    path = devin.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{")
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openDevinIntegration")
    expect(opener).to_have_text("Retry")
    path.write_text("{}")
    opener.click()
    expect(opener).to_have_text("Connect")
