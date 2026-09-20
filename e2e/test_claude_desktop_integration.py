import json
from pathlib import Path

import pytest
from playwright.sync_api import expect

from free_claude_code.harnesses import claude_desktop_integration as desktop


@pytest.mark.parametrize("status_failure", ["none", "read", "startup"])
def test_real_disconnect_failure_keeps_retry_after_reload(
    page, admin_base_url, tmp_path, monkeypatch, status_failure
):
    profile = tmp_path / "Claude-3p/configLibrary" / f"{desktop.FCC_ID}.json"
    page.goto(admin_base_url + "/admin/integrations")
    opener = page.locator("#openClaudeDesktopIntegration")
    expect(opener).to_have_text("Connect")
    opener.click()
    dialog = page.get_by_role("dialog", name="Claude Desktop", exact=True)
    dialog.get_by_role("button", name="Connect", exact=True).click()
    expect(opener).to_have_text("Disconnect")
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == profile:
            raise PermissionError("test")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        opener.click()
        endpoint = "**/admin/api/integrations/claude-desktop"
        if status_failure == "read":
            page.route(
                endpoint,
                lambda route: route.fulfill(
                    status=503, json={"detail": "Status unavailable"}
                ),
            )
        elif status_failure == "startup":

            def failed_update(route):
                result = route.fetch()
                body = result.json()
                if body["disconnect_pending"]:
                    body["connected"] = None
                    body["update"] = {
                        "state": "failed",
                        "message": "Startup check failed",
                    }
                route.fulfill(response=result, json=body)

            page.route(endpoint, failed_update)
        dialog.get_by_role("button", name="Disconnect", exact=True).click()
        expect(
            dialog.get_by_role(
                "button",
                name="Disconnect" if status_failure == "read" else "Retry disconnect",
                exact=True,
            )
        ).to_be_enabled()
        expect(dialog.get_by_role("alert")).to_be_visible()
        assert profile.exists()
        if status_failure == "read":
            page.unroute(endpoint)
        page.reload()
        expect(opener).to_have_text("Retry disconnect")
        expect(opener).to_be_enabled()
    opener.click()
    dialog.get_by_role("button", name="Retry disconnect", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Connect")
    assert not profile.exists()
    assert not (tmp_path / ".fcc/claude-desktop-disconnect.json").exists()


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
