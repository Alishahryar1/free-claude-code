"""Custom endpoint configuration through the real Admin Apply flow."""

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(autouse=True)
def no_page_errors(page: Page):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    yield
    assert errors == []


def add_provider(page: Page, name: str) -> str:
    page.get_by_role("button", name="Add provider", exact=True).click()
    dialog = page.locator("#providerDialog")
    dialog.get_by_label("Name", exact=True).fill(name)
    dialog.get_by_label("Base URL", exact=True).fill("https://gateway.example/team/v1")
    dialog.get_by_label("API key", exact=True).fill("example-custom-key")
    dialog.get_by_label("Model IDs", exact=True).fill("org/model-one\nmodel-two")
    page.locator("#saveProvider").click()
    expect(dialog).not_to_be_visible()
    card = page.locator("#providers-custom .provider-card").filter(
        has=page.get_by_text(name, exact=True)
    )
    expect(card).to_be_visible()
    expect(page.locator("#addCustomProvider")).to_be_focused()
    provider_id = card.get_attribute("data-provider")
    assert provider_id is not None
    return provider_id


def test_custom_provider_save_rename_and_remove_preserve_unrelated_draft(
    page: Page, admin_base_url: str
):
    page.goto(admin_base_url + "/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.locator("#field-HTTP_READ_TIMEOUT").fill("181")
    provider_id = add_provider(page, "Work gateway")
    expect(page.locator("#field-HTTP_READ_TIMEOUT")).to_have_value("181")
    card = page.locator(f'[data-provider="{provider_id}"]')
    expect(card.locator(".provider-check-result")).to_have_text("2 models available")
    card.get_by_role("button", name="Edit", exact=True).click()
    dialog = page.locator("#providerDialog")
    expect(dialog.get_by_label("API key", exact=True)).to_have_value("")
    dialog.get_by_label("Name", exact=True).fill("Renamed gateway")
    page.locator("#saveProvider").click()
    expect(dialog).not_to_be_visible()
    expect(card.locator(".provider-title")).to_have_text("Renamed gateway")
    expect(card.get_by_role("button", name="Edit", exact=True)).to_be_focused()
    card.get_by_role("button", name="Edit", exact=True).click()
    dialog.get_by_role("button", name="Remove provider", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(card).to_have_count(0)
    expect(page.locator("#addCustomProvider")).to_be_focused()
    expect(page.locator("#field-HTTP_READ_TIMEOUT")).to_have_value("181")


def test_custom_provider_sorting_selection_and_blocked_removal(
    page: Page, admin_base_url: str
):
    page.goto(admin_base_url + "/admin")
    add_provider(page, "Zulu gateway")
    selected = add_provider(page, "Alpha gateway")
    expect(page.locator("#providers-custom .provider-title")).to_have_text(
        ["Alpha gateway", "Zulu gateway"]
    )
    page.get_by_role("button", name="Model Config", exact=True).click()
    model = page.locator("#field-MODEL_HAIKU")
    model.fill("Alpha gateway")
    page.get_by_role("option", name="Alpha gateway/org/model-one", exact=True).click()
    expect(model).to_have_value(f"{selected}/org/model-one")
    page.locator("#applyButton").click()
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    page.get_by_role("button", name="Providers", exact=True).click()
    page.locator(f'[data-provider="{selected}"]').get_by_role(
        "button", name="Edit", exact=True
    ).click()
    page.get_by_role("button", name="Remove provider", exact=True).click()
    expect(page.locator("#providerMessage")).to_contain_text("MODEL")
    expect(page.locator("#providerDialog")).to_be_visible()


def test_custom_dialog_cancel_and_format_changes(page: Page, admin_base_url: str):
    page.goto(admin_base_url + "/admin")
    page.get_by_role("button", name="Add provider", exact=True).click()
    dialog = page.locator("#providerDialog")
    dialog.get_by_label("Reasoning format", exact=True).select_option("thinking")
    dialog.get_by_label("Reasoning history format", exact=True).select_option(
        "reasoning_content"
    )
    dialog.get_by_label("API format", exact=True).select_option("anthropic_messages")
    expect(dialog.get_by_label("Reasoning format", exact=True)).to_have_value(
        "provider_default"
    )
    expect(
        dialog.get_by_label("Reasoning history format", exact=True)
    ).not_to_be_visible()
    page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    expect(page.locator("#providers-custom .provider-card")).to_have_count(0)
    expect(page.locator("#addCustomProvider")).to_be_focused()


def test_duplicate_custom_provider_keeps_failed_save_draft(
    page: Page, admin_base_url: str
):
    page.goto(admin_base_url + "/admin")
    add_provider(page, "Gateway")
    page.get_by_role("button", name="Add provider", exact=True).click()
    dialog = page.locator("#providerDialog")
    dialog.get_by_label("Name", exact=True).fill("Gateway")
    dialog.get_by_label("Base URL", exact=True).fill("https://another.example/v1")
    page.locator("#saveProvider").click()
    expect(page.locator("#providerMessage")).to_contain_text("already exists")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_label("Base URL", exact=True)).to_have_value(
        "https://another.example/v1"
    )
    expect(page.locator("#saveProvider")).to_be_enabled()
    expect(page.locator("#providers-custom .provider-card")).to_have_count(1)
