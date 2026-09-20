import json

from playwright.sync_api import expect

from free_claude_code.harnesses import claude_desktop_integration as desktop


def test_desktop_connect_disconnect_and_retry(page, admin_base_url, tmp_path):
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openClaudeDesktopIntegration")
    expect(opener).to_have_text("Connect")
    expect(opener).to_be_enabled()
    dialog = page.get_by_role("dialog", name="Claude Desktop", exact=True)
    profile = tmp_path / "Claude-3p/configLibrary" / f"{desktop.FCC_ID}.json"
    for dismissal in ("close", "escape", "outside"):
        opener.click()
        expect(dialog).to_be_visible()
        expect(dialog.locator("#claudeDesktopIntegrationFiles")).to_contain_text(
            str(profile.resolve())
        )
        dialog.locator("p").first.click()
        expect(dialog).to_be_visible()
        if dismissal == "close":
            dialog.get_by_role("button", name="Close", exact=True).click()
        elif dismissal == "escape":
            page.keyboard.press("Escape")
        else:
            page.mouse.click(1, 1)
        expect(dialog).not_to_be_visible()
        expect(opener).to_be_focused()
    assert not profile.exists()
    opener.click()
    page.screenshot(path=str(tmp_path / "claude-desktop-connect.png"))
    action = dialog.get_by_role("button", name="Connect", exact=True)
    action.click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")
    expect(opener).to_have_css("color", "rgb(239, 68, 68)")
    assert json.loads(profile.read_text())["inferenceProvider"] == "gateway"
    page.reload()
    expect(opener).to_have_text("Disconnect")
    page.screenshot(path=str(tmp_path / "claude-desktop-connected.png"))
    opener.click()
    endpoint = "**/admin/api/integrations/claude-desktop/disconnect"
    page.route(
        endpoint,
        lambda route: route.fulfill(
            status=503, json={"detail": "File is busy; retry."}
        ),
    )
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("alert")).to_contain_text("retry")
    expect(dialog.get_by_role("button", name="Disconnect", exact=True)).to_be_enabled()
    page.unroute(endpoint)
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Connect")
    assert not profile.exists()
    assert (
        json.loads((tmp_path / "Claude-3p/claude_desktop_config.json").read_text())[
            "deploymentMode"
        ]
        == "1p"
    )
    assert not (tmp_path / "vscode/settings.json").exists()
    expect(page.locator("#dirtyState")).to_have_text("No changes")
