import json

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
