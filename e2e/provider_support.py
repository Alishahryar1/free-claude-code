"""Interactions with provider settings through the rendered Admin UI."""

from playwright.sync_api import Locator, Page, expect


def open_provider(page: Page, provider_id: str) -> Locator:
    page.locator(f'[data-provider="{provider_id}"] [data-provider-settings]').click()
    dialog = page.locator("#providerDialog")
    expect(dialog).to_be_visible()
    return dialog


def close_provider(page: Page) -> None:
    page.locator("#closeProviderDialog").click()
    expect(page.locator("#providerDialog")).not_to_be_visible()


def api_key_input(page: Page, key: str, index: int = 0) -> Locator:
    editor = page.locator(f".api-key-editor:has(#field-{key})")
    inputs = editor.locator(".api-key-row input")
    while inputs.count() <= index:
        editor.get_by_role("button", name="Add key", exact=True).click()
    return inputs.nth(index)
