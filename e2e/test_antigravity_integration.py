"""The remaining integrations work without Antigravity's former card/scripts."""

from playwright.sync_api import Page, expect


def test_antigravity_card_is_removed_without_breaking_other_integrations(
    page: Page, admin_base_url: str
):
    errors = []
    page.on("pageerror", lambda error: errors.append(error))
    page.goto(f"{admin_base_url}/admin/integrations")
    expect(page.locator("#openCodexIntegration")).to_be_enabled()
    expect(
        page.get_by_role("heading", name="Antigravity CLI", exact=True)
    ).to_have_count(0)
    expect(page.locator("#antigravityIntegrationDialog")).to_have_count(0)
    page.locator("#openCodexIntegration").click()
    expect(page.locator("#codexIntegrationDialog")).to_be_visible()
    assert not errors
