"""Rendered model-fallback controls for the local Admin UI."""

import pytest
from playwright.sync_api import Page, expect


def _open_models(page: Page, admin_base_url: str) -> None:
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.get_by_role("button", name="Model Config", exact=True).click()


def _refresh_openrouter_models(page: Page) -> None:
    page.get_by_role("button", name="Providers", exact=True).click()
    card = page.locator('[data-provider="open_router"]')
    card.get_by_role("button", name="Refresh models", exact=True).click()
    expect(card.locator(".provider-check-result")).to_have_text("3 models available")
    page.get_by_role("button", name="Model Config", exact=True).click()


def test_fallback_editor_adds_filters_reorders_and_removes_models(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 1280, "height": 720})
    _open_models(page, admin_base_url)
    _refresh_openrouter_models(page)

    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    add_input = editor.locator("#field-MODEL_FALLBACKS-add")
    stored_value = editor.locator("#field-MODEL_FALLBACKS")
    expect(editor.locator(".model-list-empty")).to_have_text(
        "No fallback models configured."
    )

    add_input.fill("model-b")
    expect(
        editor.get_by_role("option", name="open_router/vendor/model-b", exact=True)
    ).to_be_visible()
    add_input.press("ArrowDown")
    add_input.press("Enter")
    expect(add_input).to_have_value("open_router/vendor/model-b")
    editor.get_by_role("button", name="Add", exact=True).click()

    add_input.fill("groq/custom-model")
    editor.get_by_role("button", name="Add", exact=True).click()
    expect(stored_value).to_have_value("open_router/vendor/model-b,groq/custom-model")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    expect(page.locator("#applyButton")).to_be_enabled()

    editor.get_by_role("button", name="Move groq/custom-model up", exact=True).click()
    expect(stored_value).to_have_value("groq/custom-model,open_router/vendor/model-b")
    expect(
        editor.get_by_role("button", name="Move groq/custom-model up", exact=True)
    ).to_be_disabled()
    expect(
        editor.get_by_role(
            "button", name="Move open_router/vendor/model-b down", exact=True
        )
    ).to_be_disabled()

    editor.get_by_role("button", name="Remove groq/custom-model", exact=True).click()
    expect(stored_value).to_have_value("open_router/vendor/model-b")
    editor.get_by_role(
        "button", name="Remove open_router/vendor/model-b", exact=True
    ).click()
    expect(stored_value).to_have_value("")
    expect(editor.locator(".model-list-empty")).to_be_visible()
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    expect(page.locator("#applyButton")).to_be_disabled()


def test_fallback_editor_rejects_blank_and_duplicate_rows(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 1280, "height": 720})
    _open_models(page, admin_base_url)

    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    add_input = editor.locator("#field-MODEL_FALLBACKS-add")
    add_button = editor.get_by_role("button", name="Add", exact=True)

    add_button.click()
    expect(page.locator("#messageArea")).to_have_text(
        "Enter a full provider/model fallback."
    )
    add_input.fill("groq/custom-model")
    add_button.click()
    add_input.fill("groq/custom-model")
    add_button.click()
    expect(page.locator("#messageArea")).to_have_text(
        "That fallback model is already in the list."
    )
    expect(editor.locator(".model-list-row")).to_have_count(1)


@pytest.mark.parametrize(
    "admin_base_url",
    [{"MODEL_FALLBACKS": "groq/model-a,open_router/model-b"}],
    indirect=True,
)
def test_process_owned_fallback_list_locks_every_mutation(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 1280, "height": 720})
    _open_models(page, admin_base_url)

    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    expect(editor.locator("#field-MODEL_FALLBACKS-add")).to_be_disabled()
    expect(editor.get_by_role("button", name="Add", exact=True)).to_be_disabled()
    expect(editor.locator(".model-list-action")).to_have_count(6)
    for button in editor.locator(".model-list-action").all():
        expect(button).to_be_disabled()


def test_fallback_editor_remains_usable_at_narrow_viewport(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    _open_models(page, admin_base_url)

    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    editor.scroll_into_view_if_needed()
    expect(editor.locator("#field-MODEL_FALLBACKS-add")).to_be_visible()
    expect(editor.get_by_role("button", name="Add", exact=True)).to_be_visible()
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )


def test_model_suggestions_keep_server_order_after_provider_check_and_late_response(
    page: Page,
    admin_base_url: str,
) -> None:
    models = [
        "alpha/Apple",
        "alpha/apple",
        "alpha/ss",
        "alpha/ß",
        "alpha/z",
        "alpha-2/model",
    ]
    page.route(
        "**/admin/api/models",
        lambda route: route.fulfill(json={"models": models, "failed_providers": []}),
    )
    _open_models(page, admin_base_url)
    page.wait_for_function("!state.startupRequest && !state.startupTimer")
    field = page.locator('.field[data-key="MODEL_SONNET"]')
    field.locator("input").fill("")
    expect(field.get_by_role("option")).to_have_text(["None", *models])
    field.locator("input").fill("open_router/custom-unsaved")
    field.locator("input").press("Tab")

    models.append("open_router/new-model")
    page.route(
        "**/admin/api/providers/open_router/test",
        lambda route: route.fulfill(json={"ok": True, "models": ["new-model"]}),
    )
    page.get_by_role("button", name="Providers", exact=True).click()
    with page.expect_request("**/admin/api/models"):
        page.locator('[data-provider="open_router"]').get_by_role(
            "button", name="Refresh models", exact=True
        ).click()
    page.get_by_role("button", name="Model Config", exact=True).click()
    expect(field.locator("input")).to_have_value("open_router/custom-unsaved")
    field.locator("input").fill("")
    expect(field.get_by_role("option")).to_have_text(["None", *models])
    field.locator("input").press("Escape")
    optional = page.locator('.field[data-key="MODEL_OPUS"]')
    optional.locator("input").fill("")
    expect(optional.get_by_role("option")).to_have_text(["None", *models])

    page.wait_for_function("!state.startupRequest && !state.startupTimer")
    page.unroute("**/admin/api/models")
    pending = []

    def hold_catalog(route):
        pending.append(route)
        page.evaluate("count => { window.heldCatalogLoads = count; }", len(pending))

    page.route("**/admin/api/models", hold_catalog)
    for _ in range(2):
        with page.expect_request("**/admin/api/models"):
            page.evaluate("void hydrateModelOptions()")
    page.wait_for_function("window.heldCatalogLoads === 2")
    with page.expect_response("**/admin/api/models"):
        pending[1].fulfill(json={"models": models, "failed_providers": []})
    with page.expect_response("**/admin/api/models") as stale:
        pending[0].fulfill(json={"models": ["old/stale"], "failed_providers": []})
    stale.value.finished()
    expect(optional.get_by_role("option")).to_have_text(["None", *models])
