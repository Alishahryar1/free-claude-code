import json

import pytest
from playwright.sync_api import expect

from free_claude_code.harnesses import vscode_chat_integration as vscode


def test_connect_and_disconnect_native_chat(page, admin_base_url):
    path = vscode.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    other = {"vendor": "customendpoint", "name": "Other", "models": []}
    path.write_text(json.dumps([other]))
    page.goto(f"{admin_base_url}/admin/integrations")
    button = page.locator("#openVSCodeChatIntegration")
    dialog = page.locator("#vscodeChatIntegrationDialog")
    expect(button).to_have_text("Connect")
    button.click()
    expect(dialog).to_be_visible()
    expect(dialog.locator("code")).to_have_text(str(path.resolve()))
    page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    button.click()
    dialog.get_by_role("button", name="Close", exact=True).click()
    expect(dialog).not_to_be_visible()
    button.click()
    dialog.get_by_role("button", name="Connect", exact=True).click()
    expect(button).to_have_text("Disconnect")
    expect(button).to_have_class("danger-button")
    group = json.loads(path.read_text())[1]
    assert group["apiType"] == "messages"
    assert group["models"]
    assert all(m["url"].endswith("/v1/messages") for m in group["models"])
    page.reload()
    expect(button).to_have_text("Disconnect")
    expect(page.locator("#vscodeChatIntegrationMessage")).not_to_be_visible()
    button.click()
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(button).to_have_text("Connect")
    assert json.loads(path.read_text()) == [other]


@pytest.mark.parametrize("status_read_failed", [False, True])
def test_failed_update_has_actionable_retry(page, admin_base_url, status_read_failed):
    progress: dict[str, str | bool | None] = {
        "state": "failed",
        "changed": False,
        "message": "Could not update settings.",
    }
    retries = []

    def startup(route):
        response = route.fetch()
        payload = response.json()
        payload["startup"]["integrations"]["vscode-chat"] = dict(progress)
        route.fulfill(response=response, json=payload)

    def status(route):
        if status_read_failed and progress["state"] == "failed":
            route.fulfill(status=503, json={"detail": "Could not read settings."})
        else:
            route.fulfill(
                json={"connected": True, "paths": None, "update": dict(progress)}
            )

    def retry(route):
        retries.append(route.request.method)
        progress.update(state="ready", message=None)
        route.fulfill(json={"update": dict(progress)})

    try:
        page.route("**/admin/api/status", startup)
        page.route("**/admin/api/integrations/vscode-chat", status)
        page.route("**/admin/api/integrations/vscode-chat/refresh", retry)
        page.goto(f"{admin_base_url}/admin/integrations")
        # Startup reconciliation can temporarily disable Retry during the click.
        # Exercise the action after initial polling has settled.
        page.wait_for_function(
            "state.startup && !state.startupRequest && state.startupTimer === null"
        )
        main = page.locator("#openVSCodeChatIntegration")
        secondary = page.locator("#retryVSCodeChatIntegration")
        if status_read_failed:
            expect(main).to_have_text("Retry")
            action = main
        else:
            expect(main).to_have_text("Disconnect")
            expect(secondary).to_be_visible()
            action = secondary
        with page.expect_response("**/admin/api/integrations/vscode-chat/refresh"):
            action.click()
        assert retries == ["POST"]
        expect(main).to_have_text("Disconnect")
        expect(secondary).to_be_hidden()
        expect(page.locator("#vscodeChatIntegrationMessage")).to_be_hidden()
    finally:
        page.unroute_all(behavior="wait")
