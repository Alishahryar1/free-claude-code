import json

import pytest
from playwright.sync_api import Page, Route, expect

from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.harnesses import vscode_chat_integration as vscode
from tests.wire_payload import (
    array_payload,
    object_array,
    object_payload,
    response_object,
    string_payload,
)


def test_connect_and_disconnect_native_chat(page: Page, admin_base_url: str):
    path = vscode.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    other: JsonObject = {"vendor": "customendpoint", "name": "Other", "models": []}
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
    group = array_payload(load_json_value(path.read_text()))[1]
    assert object_payload(group)["apiType"] == "messages"
    assert object_payload(group)["models"]
    assert all(
        string_payload(m["url"]).endswith("/v1/messages")
        for m in object_array(object_payload(group)["models"])
    )
    page.reload()
    expect(button).to_have_text("Disconnect")
    expect(page.locator("#vscodeChatIntegrationMessage")).not_to_be_visible()
    button.click()
    dialog.get_by_role("button", name="Disconnect", exact=True).click()
    expect(button).to_have_text("Connect")
    assert load_json_value(path.read_text()) == [other]


@pytest.mark.parametrize("status_read_failed", [False, True])
def test_failed_update_has_actionable_retry(
    page: Page, admin_base_url: str, status_read_failed: bool
):
    progress: dict[str, str | bool | None] = {
        "state": "failed",
        "changed": False,
        "message": "Could not update settings.",
    }
    retries = []

    def startup(route: Route):
        response = route.fetch()
        payload = response_object(response)
        object_payload(object_payload(payload["startup"])["integrations"])[
            "vscode-chat"
        ] = dict(progress)
        object_payload(object_payload(payload["startup"])["messaging"])["state"] = (
            "starting"
        )
        route.fulfill(response=response, json=payload)

    def status(route: Route):
        if status_read_failed and progress["state"] == "failed":
            route.fulfill(status=503, json={"detail": "Could not read settings."})
        else:
            route.fulfill(
                json={"connected": True, "paths": None, "update": dict(progress)}
            )

    def retry(route: Route):
        retries.append(route.request.method)
        progress.update(state="ready", message=None)
        route.fulfill(json={"update": dict(progress)})

    try:
        page.route("**/admin/api/status", startup)
        page.route("**/admin/api/integrations/vscode-chat", status)
        page.route("**/admin/api/integrations/vscode-chat/refresh", retry)
        page.goto(f"{admin_base_url}/admin/integrations")
        # Startup reconciliation can temporarily disable Retry during the click.
        # Wait for this integration without waiting for unrelated startup work.
        page.wait_for_function(
            "state.startup?.startup?.integrations?.['vscode-chat']?.state === 'failed'"
            " && !vscodeChatIntegration.busy"
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
