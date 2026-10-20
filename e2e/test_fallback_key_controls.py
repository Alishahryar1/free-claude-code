"""Fallback rows retain numbered credential choices through actual Admin edits."""

import json

import pytest
from playwright.sync_api import Page, expect

from e2e.provider_support import open_provider
from e2e.test_admin_models import _fallbacks, _open_models
from e2e.test_api_key_lists import add_key

CUSTOM_ID = "custom_" + "a" * 32


@pytest.mark.parametrize(
    "admin_base_url",
    [{"GROQ_API_KEY": '["first","second","third"]'}],
    indirect=True,
)
def test_numbered_rows_edit_reorder_save_and_reopen(page: Page, admin_base_url: str):
    page.set_viewport_size({"width": 1280, "height": 1200})
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    model = editor.locator("#field-MODEL_FALLBACKS-add")
    key = editor.get_by_label("Fallback API key", exact=True)
    effort = editor.get_by_label("Fallback reasoning effort", exact=True)
    for number, reasoning in ((2, "low"), (3, "high")):
        model.fill("groq/model-a")
        expect(key.locator("option")).to_have_text(["Key 1", "Key 2", "Key 3"])
        key.select_option(str(number))
        effort.select_option(reasoning)
        editor.get_by_role("button", name="Add fallback", exact=True).click()
        expect(key).to_have_value("1")
    model.fill("groq/model-a")
    key.select_option("2")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    expect(editor.locator(".model-list-row")).to_have_count(2)
    expect(page.locator("#messageArea")).not_to_have_text("")
    first = editor.get_by_role("combobox", name="Fallback model 1", exact=True)
    first.fill("groq/model-b")
    expect(first).to_be_focused()
    expect(editor.get_by_label("API key for groq/model-b", exact=True)).to_have_value(
        "2"
    )
    handle = editor.get_by_role("button", name="Move groq/model-a", exact=True)
    handle.focus()
    handle.press("ArrowUp")
    expected = [
        {"model": "groq/model-a", "reasoning": "high", "api_key_number": 3},
        {"model": "groq/model-b", "reasoning": "low", "api_key_number": 2},
    ]
    assert _fallbacks(editor) == expected
    editor.locator(".model-list-row").first.get_by_role(
        "button", name="Move groq/model-a", exact=True
    ).drag_to(
        editor.locator(".model-list-row").last, target_position={"x": 10, "y": 38}
    )
    expected.reverse()
    assert _fallbacks(editor) == expected
    with page.expect_response("**/admin/api/config/apply") as saved:
        page.locator("#applyButton").click()
    assert saved.value.json()["applied"] is True
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    page.reload()
    page.get_by_role("button", name="Model Config", exact=True).click()
    assert _fallbacks(editor) == expected
    expect(editor.get_by_label("API key for groq/model-a", exact=True)).to_have_value(
        "3"
    )
    expect(editor.get_by_label("API key for groq/model-b", exact=True)).to_have_value(
        "2"
    )
    page.screenshot(path="untracked/fallback-api-key-controls.png", full_page=True)


@pytest.mark.parametrize(
    "admin_base_url",
    [
        {
            "OPENAI_API_KEY": '["oa-first","oa-second"]',
            "ZAI_API_KEY": '["zai-first","zai-second","zai-third"]',
            "OPENCODE_API_KEY": '["oc-first","oc-second"]',
            "FCC_CUSTOM_PROVIDERS": json.dumps(
                [
                    {
                        "provider_id": CUSTOM_ID,
                        "display_name": "Gateway",
                        "base_url": "https://gateway.example/v1",
                        "api_keys": ["custom-first", "custom-second"],
                    }
                ]
            ),
        }
    ],
    indirect=True,
)
def test_options_follow_shared_custom_and_default_authentication(
    page: Page, admin_base_url: str
):
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    model = editor.locator("#field-MODEL_FALLBACKS-add")
    key = editor.get_by_label("Fallback API key", exact=True)
    for provider, count in (
        ("openai_api", 2),
        ("zai", 3),
        ("zai_api", 3),
        ("opencode_zen", 2),
        ("opencode_go", 2),
        (CUSTOM_ID, 2),
    ):
        model.fill(provider + "/model")
        expect(key.locator("option")).to_have_text(
            [f"Key {index}" for index in range(1, count + 1)]
        )
        expect(key).to_be_enabled()
    for provider in ("openai", "github_copilot", "vertex", "lmstudio"):
        model.fill(provider + "/model")
        expect(key.locator("option")).to_have_text(["Default authentication"])
        expect(key).to_be_disabled()
        expect(key).to_have_value("1")


@pytest.mark.parametrize(
    "admin_base_url",
    [{"GROQ_API_KEY": '["first","second"]'}],
    indirect=True,
)
def test_changed_provider_resets_key_without_resetting_reasoning(
    page: Page, admin_base_url: str
):
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    add = editor.locator("#field-MODEL_FALLBACKS-add")
    add.fill("groq/model")
    editor.get_by_label("Fallback API key", exact=True).select_option("2")
    editor.get_by_label("Fallback reasoning effort", exact=True).select_option("max")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    model = editor.get_by_role("combobox", name="Fallback model 1", exact=True)
    model.fill("open_router/other")
    expect(model).to_be_focused()
    assert _fallbacks(editor) == [
        {"model": "open_router/other", "reasoning": "max", "api_key_number": 1}
    ]
    expect(
        editor.get_by_label("API key for open_router/other", exact=True)
    ).to_have_value("1")
    add.fill("groq/model")
    editor.get_by_label("Fallback API key", exact=True).select_option("2")
    add.fill("lmstudio/local")
    expect(editor.get_by_label("Fallback API key", exact=True)).to_have_value("1")


@pytest.mark.parametrize(
    "admin_base_url",
    [
        {
            "MODEL_FALLBACKS": json.dumps(
                [{"model": "groq/model", "api_key_number": 2}]
            ),
            "GROQ_API_KEY": '["first","second"]',
        }
    ],
    indirect=True,
)
def test_process_owned_key_selector_is_locked(page: Page, admin_base_url: str):
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    expect(editor.get_by_label("Fallback API key", exact=True)).to_be_disabled()
    key = editor.get_by_label("API key for groq/model", exact=True)
    expect(key).to_be_disabled()
    expect(key).to_have_value("2")
    expect(page.locator("#dirtyState")).to_have_text("No changes")


def test_provider_reload_keeps_draft_and_marks_removed_number_unavailable(
    page: Page, admin_base_url: str
):
    _open_models(page, admin_base_url)
    page.get_by_role("button", name="Providers", exact=True).click()
    dialog = open_provider(page, "nvidia_nim")
    for value in ("first", "second", "third"):
        add_key(dialog, value)
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(page.locator("#applyButton")).to_have_text("Apply")
    page.get_by_role("button", name="Model Config", exact=True).click()
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    editor.locator("#field-MODEL_FALLBACKS-add").fill("nvidia_nim/draft")
    editor.get_by_label("Fallback API key", exact=True).select_option("3")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    page.locator("#field-MODEL_SONNET").fill("groq/unsaved")
    page.get_by_role("button", name="Providers", exact=True).click()
    dialog = open_provider(page, "nvidia_nim")
    dialog.get_by_role("button", name="Remove Key 3", exact=True).click()
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    expect(page.locator("#applyButton")).to_have_text("Apply")
    page.get_by_role("button", name="Model Config", exact=True).click()
    expect(page.locator("#field-MODEL_SONNET")).to_have_value("groq/unsaved")
    key = editor.get_by_label("API key for nvidia_nim/draft", exact=True)
    expect(key).to_have_value("3")
    expect(key.locator("option")).to_have_text(
        ["Key 1", "Key 2", "Key 3 (unavailable)"]
    )
    assert _fallbacks(editor)[0]["api_key_number"] == 3
    key.select_option("2")
    with page.expect_response("**/admin/api/config/apply") as saved:
        page.locator("#applyButton").click()
    assert saved.value.json()["applied"] is True
    expect(page.locator("#dirtyState")).to_have_text("No changes")
