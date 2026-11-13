import json

import pytest
from playwright.sync_api import expect

from free_claude_code.harnesses import antigravity_integration as native


@pytest.mark.parametrize("interrupted", [False, True])
def test_cleanup_remains_reachable_without_reconnecting(
    page, admin_base_url, monkeypatch, interrupted
):
    endpoint = f"{admin_base_url}/admin/api/integrations/antigravity"
    path, state = native.config_path(), native.state_path()
    original = {"modelProvider": "personal", "unrelated": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(original))
    if interrupted:
        page.goto(f"{admin_base_url}/admin/integrations")
        page.locator("#openAntigravityIntegration").click()

        def fail(*args):
            raise PermissionError("write interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(native, "_write_native", fail)
            page.locator("#confirmAntigravityIntegration").click()
            expect(
                page.locator("#antigravityIntegrationDialogMessage")
            ).to_contain_text("Could not access")
        page.keyboard.press("Escape")
    else:
        assert page.request.post(endpoint + "/connect").status == 200
        document = json.loads(path.read_text())
        document["modelProvider"] = "personal"
        path.write_text(json.dumps(document))
        page.goto(f"{admin_base_url}/admin/integrations")
    card = page.locator("article").filter(
        has=page.get_by_role("heading", name="Antigravity CLI", exact=True)
    )
    expect(card.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    cleanup = card.get_by_role("button", name="Disconnect", exact=True)
    expect(cleanup).to_be_enabled()
    assert state.exists()
    cleanup.click()
    confirm = page.locator("#confirmAntigravityIntegration")
    expect(confirm).to_have_text("Disconnect")
    confirm.click()
    expect(page.locator("#antigravityIntegrationDialog")).not_to_be_visible()
    expect(cleanup).to_have_count(0)
    assert not state.exists()
    assert json.loads(path.read_text()) == original


def test_connect_disconnect_and_existing_connection(page, admin_base_url):
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openAntigravityIntegration")
    expect(opener).to_have_text("Connect")
    path, state = native.config_path(), native.state_path()
    assert not path.exists() and not state.exists()
    original = {
        "modelProvider": "personal",
        "model": "chosen",
        "unrelated": {"keep": True},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(original))
    opener.click()
    dialog = page.get_by_role("dialog", name="Antigravity CLI", exact=True)
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("button", name="Close", exact=True)).to_be_focused()
    expect(dialog).to_contain_text("Gemini API provider")
    expect(dialog.locator("li")).to_have_text(
        [str(path.resolve()), str(state.resolve())]
    )
    page.keyboard.press("Escape")
    assert json.loads(path.read_text()) == original and not state.exists()
    opener.click()
    page.locator("#confirmAntigravityIntegration").click()
    expect(dialog).not_to_be_visible()
    expect(opener).to_have_text("Disconnect")
    expect(opener).to_have_css("color", "rgb(239, 68, 68)")
    assert json.loads(path.read_text())["modelProvider"] == "gemini"
    page.reload()
    expect(opener).to_have_text("Disconnect")
    opener.click()
    expect(dialog).to_contain_text("Running sessions can continue")
    page.locator("#confirmAntigravityIntegration").click()
    expect(opener).to_have_text("Connect")
    expect(opener).to_have_css("color", "rgb(6, 16, 11)")
    assert json.loads(path.read_text()) == original
    assert not state.exists()


def test_failed_connect_keeps_modal_retryable(page, admin_base_url):
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openAntigravityIntegration")
    expect(opener).to_have_text("Connect")
    path = native.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("invalid JSON")
    opener.click()
    confirm = page.locator("#confirmAntigravityIntegration")
    confirm.click()
    expect(page.locator("#antigravityIntegrationDialogMessage")).to_contain_text(
        "Could not configure"
    )
    expect(confirm).to_be_enabled()
    assert path.read_text() == "invalid JSON" and not native.state_path().exists()
    path.write_text("{}")
    confirm.click()
    expect(opener).to_have_text("Disconnect")


def test_failed_status_can_be_retried(page, admin_base_url):
    endpoint = "**/admin/api/integrations/antigravity"
    page.route(
        endpoint, lambda route: route.fulfill(status=503, json={"detail": "Try again"})
    )
    page.goto(f"{admin_base_url}/admin/integrations")
    opener = page.locator("#openAntigravityIntegration")
    expect(opener).to_have_text("Retry")
    expect(page.locator("#antigravityIntegrationMessage")).to_have_text("Try again")
    page.unroute(endpoint)
    opener.click()
    expect(opener).to_have_text("Connect")
    assert not native.state_path().exists()
