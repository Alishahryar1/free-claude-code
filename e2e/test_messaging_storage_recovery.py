import pytest
from playwright.sync_api import expect


@pytest.fixture
def admin_client_files(tmp_path):
    source = tmp_path / ".fcc/agent_workspace/sessions.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"conversation":')
    return source


def test_damaged_legacy_history_warns_without_blocking_settings(
    page, admin_base_url, admin_client_files
):
    page.goto(f"{admin_base_url}/admin")
    page.get_by_role("button", name="Messaging", exact=True).click()
    notice = page.locator("#startupMessage")
    expect(notice).to_be_visible()
    expect(notice).to_contain_text("Messaging can still be used")
    expect(page.locator("#field-TELEGRAM_BOT_TOKEN")).to_be_enabled()
    assert admin_client_files.read_text() == '{"conversation":'
    # A preserved source is not a second writer, and the server stays usable.
    response = page.request.get(f"{admin_base_url}/health")
    assert response.ok
    page.get_by_role("button", name="Providers", exact=True).click()
    expect(notice).to_be_hidden()
