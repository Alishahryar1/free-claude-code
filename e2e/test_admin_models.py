"""Rendered model-fallback controls for the local Admin UI."""

import json
from typing import cast

import pytest
from playwright.sync_api import Locator, Page, expect

from e2e.provider_support import close_provider, open_provider


def _open_models(page: Page, admin_base_url: str) -> None:
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.get_by_role("button", name="Model Config", exact=True).click()


def _refresh_openrouter_models(page: Page) -> None:
    page.get_by_role("button", name="Providers", exact=True).click()
    card = page.locator('[data-provider="open_router"]')
    dialog = open_provider(page, "open_router")
    dialog.get_by_role("button", name="Refresh models", exact=True).click()
    expect(card.locator(".provider-check-result")).to_have_text("3 models available")
    close_provider(page)
    page.get_by_role("button", name="Model Config", exact=True).click()


def _fallbacks(editor: Locator) -> list[dict[str, str | int]]:
    value = editor.locator("#field-MODEL_FALLBACKS").input_value()
    return json.loads(value) if value else []


def test_model_routing_pairs_five_models_with_effort_and_defaults(
    page: Page, admin_base_url: str
) -> None:
    page.set_viewport_size({"width": 1280, "height": 900})
    _open_models(page, admin_base_url)
    expect(page.locator("#section-reasoning")).to_have_count(0)
    routes = ("Default", "Fable", "Opus", "Sonnet", "Haiku")
    previous_bottom = 0.0
    for route in routes:
        group = page.get_by_role("group", name=f"{route} model and effort", exact=True)
        model_key = "MODEL" if route == "Default" else f"MODEL_{route.upper()}"
        effort_key = (
            "REASONING_POLICY" if route == "Default" else f"REASONING_{route.upper()}"
        )
        model = group.locator(f"#field-{model_key}")
        effort = group.locator(f"#field-{effort_key}")
        expect(model).to_be_visible()
        expect(effort).to_have_value("client" if route == "Default" else "inherit")
        if route != "Default":
            expect(model).to_have_value("None")
            expect(effort).to_be_enabled()
        model_box, effort_box = model.bounding_box(), effort.bounding_box()
        assert model_box is not None and effort_box is not None
        assert abs(model_box["width"] - effort_box["width"]) <= 1
        assert abs(model_box["height"] - effort_box["height"]) <= 1
        assert abs(model_box["y"] - effort_box["y"]) <= 1
        assert model_box["x"] < effort_box["x"]
        assert model_box["y"] > previous_bottom
        previous_bottom = model_box["y"] + model_box["height"]
    expect(page.locator("#field-REASONING_POLICY option[value=inherit]")).to_have_count(
        0
    )
    fallback_box = page.locator('.field[data-key="MODEL_FALLBACKS"]').bounding_box()
    assert fallback_box is not None and fallback_box["y"] > previous_bottom


def test_model_routing_saves_independent_model_and_effort_choices(
    page: Page, admin_base_url: str
) -> None:
    _open_models(page, admin_base_url)
    page.locator("#field-REASONING_POLICY").select_option("medium")
    opus = page.get_by_role("group", name="Opus model and effort", exact=True)
    expect(opus.locator("#field-MODEL_OPUS")).to_have_value("None")
    opus.locator("#field-REASONING_OPUS").select_option("high")
    sonnet = page.get_by_role("group", name="Sonnet model and effort", exact=True)
    sonnet.locator("#field-MODEL_SONNET").fill("open_router/vendor/model-b")
    sonnet.locator("#field-REASONING_SONNET").select_option("low")
    fallback = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    fallback.locator("#field-MODEL_FALLBACKS-add").fill("open_router/vendor/model-a")
    fallback.get_by_label("Fallback reasoning effort", exact=True).select_option("off")
    fallback.get_by_role("button", name="Add fallback", exact=True).click()
    with page.expect_response("**/admin/api/config/apply") as saved:
        page.locator("#applyButton").click()
    assert saved.value.json()["applied"] is True
    payload = saved.value.request.post_data_json
    assert isinstance(payload, dict)
    values = payload["values"]
    assert values["REASONING_POLICY"] == "medium"
    assert values["REASONING_OPUS"] == "high"
    assert "MODEL_OPUS" not in values
    assert values["MODEL_SONNET"] == "open_router/vendor/model-b"
    assert values["REASONING_SONNET"] == "low"
    assert json.loads(values["MODEL_FALLBACKS"]) == [
        {"model": "open_router/vendor/model-a", "reasoning": "off", "api_key_number": 1}
    ]
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    page.reload()
    expect(page.locator("#messageArea")).to_have_text("")
    expect(page.locator("#field-REASONING_POLICY")).to_have_value("medium")
    expect(opus.locator("#field-MODEL_OPUS")).to_have_value("None")
    expect(opus.locator("#field-REASONING_OPUS")).to_have_value("high")
    expect(sonnet.locator("#field-MODEL_SONNET")).to_have_value(
        "open_router/vendor/model-b"
    )
    expect(sonnet.locator("#field-REASONING_SONNET")).to_have_value("low")
    assert _fallbacks(fallback) == [
        {"model": "open_router/vendor/model-a", "reasoning": "off", "api_key_number": 1}
    ]
    expect(page.locator("#dirtyState")).to_have_text("No changes")


@pytest.mark.parametrize(
    "admin_base_url",
    [
        {
            "MODEL_OPUS": "groq/locked-model",
            "REASONING_SONNET": "off",
            "REASONING_POLICY": "client",
        }
    ],
    indirect=True,
)
def test_process_owned_route_controls_lock_independently(
    page: Page, admin_base_url: str
) -> None:
    _open_models(page, admin_base_url)
    expect(page.locator("#field-REASONING_POLICY")).to_have_value("client")
    expect(page.locator("#field-REASONING_POLICY")).to_be_disabled()
    opus = page.get_by_role("group", name="Opus model and effort", exact=True)
    expect(opus.locator("#field-MODEL_OPUS")).to_be_disabled()
    expect(opus.locator("#field-REASONING_OPUS")).to_be_enabled()
    sonnet = page.get_by_role("group", name="Sonnet model and effort", exact=True)
    expect(sonnet.locator("#field-MODEL_SONNET")).to_be_enabled()
    expect(sonnet.locator("#field-REASONING_SONNET")).to_be_disabled()


def test_fallback_editor_adds_filters_reorders_and_removes_models(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 1280, "height": 720})
    _open_models(page, admin_base_url)
    _refresh_openrouter_models(page)

    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    add_input = editor.locator("#field-MODEL_FALLBACKS-add")
    add_effort = editor.get_by_label("Fallback reasoning effort", exact=True)
    stored_value = editor.locator("#field-MODEL_FALLBACKS")
    expect(editor.locator(".model-list-row")).to_have_count(0)

    add_input.fill("model-b")
    expect(
        editor.get_by_role("option", name="open_router/vendor/model-b", exact=True)
    ).to_be_visible()
    add_input.press("ArrowDown")
    add_input.press("Enter")
    expect(add_input).to_have_value("open_router/vendor/model-b")
    add_effort.select_option("low")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    expect(add_effort).to_have_value("inherit")
    expect(add_input).to_have_value("")

    add_input.fill("groq/custom-model")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    assert _fallbacks(editor) == [
        {
            "model": "open_router/vendor/model-b",
            "reasoning": "low",
            "api_key_number": 1,
        },
        {"model": "groq/custom-model", "reasoning": "inherit", "api_key_number": 1},
    ]
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    expect(page.locator("#applyButton")).to_be_enabled()

    rows = editor.locator(".model-list-row")
    expect(rows.locator('input[role="combobox"]')).to_have_count(2)
    last_model_box = editor.get_by_role(
        "combobox", name="Fallback model 2", exact=True
    ).bounding_box()
    add_model_box = add_input.bounding_box()
    assert last_model_box is not None and add_model_box is not None
    assert add_model_box["y"] > last_model_box["y"]
    editor.get_by_role("button", name="Move groq/custom-model", exact=True).drag_to(
        rows.nth(0), target_position={"x": 10, "y": 5}
    )
    assert _fallbacks(editor) == [
        {"model": "groq/custom-model", "reasoning": "inherit", "api_key_number": 1},
        {
            "model": "open_router/vendor/model-b",
            "reasoning": "low",
            "api_key_number": 1,
        },
    ]
    editor.get_by_role("button", name="Remove groq/custom-model", exact=True).click()
    assert _fallbacks(editor) == [
        {"model": "open_router/vendor/model-b", "reasoning": "low", "api_key_number": 1}
    ]
    editor.get_by_role(
        "button", name="Remove open_router/vendor/model-b", exact=True
    ).click()
    expect(stored_value).to_have_value("")
    expect(editor.locator(".model-list-row")).to_have_count(0)
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
    add_button = editor.get_by_role("button", name="Add fallback", exact=True)

    add_button.click()
    expect(page.locator("#messageArea")).to_have_text(
        "Enter a full provider/model fallback."
    )
    add_input.fill("groq/custom-model")
    editor.get_by_label("Fallback reasoning effort", exact=True).select_option("high")
    add_button.click()
    add_input.fill("groq/custom-model")
    add_button.click()
    expect(page.locator("#messageArea")).to_have_text(
        "That model and API key are already in the fallback list."
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
    expect(
        editor.get_by_label("Fallback reasoning effort", exact=True)
    ).to_be_disabled()
    expect(
        editor.get_by_role("button", name="Add fallback", exact=True)
    ).to_be_disabled()
    expect(editor.locator(".model-list-row .model-list-action")).to_have_count(4)
    for button in editor.locator(".model-list-action").all():
        expect(button).to_be_disabled()
    for effort in editor.locator(".model-list-row select").all():
        expect(effort).to_be_disabled()
    for model in editor.locator(".model-list-row input").all():
        expect(model).to_be_disabled()
    for handle in editor.locator(".model-list-drag").all():
        expect(handle).to_have_attribute("draggable", "false")


def test_fallback_editor_pairs_match_existing_controls_and_save_effort(
    page: Page, admin_base_url: str
) -> None:
    page.set_viewport_size({"width": 1280, "height": 900})
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    model = editor.locator("#field-MODEL_FALLBACKS-add")
    effort = editor.get_by_label("Fallback reasoning effort", exact=True)
    expect(effort.locator("option")).to_have_text(
        ["Inherit", "Off", "From client", "Low", "Medium", "High", "X-High", "Max"]
    )
    bounds = [
        control.bounding_box()
        for control in (
            model,
            effort,
            editor.get_by_role("button", name="Add fallback", exact=True),
        )
    ]
    assert all(box is not None for box in bounds)
    model_box, effort_box, add_box = bounds
    assert model_box is not None and effort_box is not None and add_box is not None
    assert abs(model_box["width"] - effort_box["width"]) <= 1
    assert abs(model_box["height"] - effort_box["height"]) <= 1
    assert model_box["x"] < effort_box["x"] < add_box["x"]
    assert abs(model_box["y"] - effort_box["y"]) <= 1
    style = "el => { const s = getComputedStyle(el); return [s.fontSize, s.borderRadius, s.borderWidth, s.backgroundSize]; }"
    assert effort.evaluate(style) == page.locator("#field-REASONING_POLICY").evaluate(
        style
    )

    # A fallback can match Default today, even with a different effort.
    model.fill("open_router/e2e-default")
    effort.select_option("off")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    row_model = editor.get_by_role("combobox", name="Fallback model 1", exact=True)
    row_model.fill("model-b")
    editor.get_by_role("option", name="open_router/vendor/model-b", exact=True).click()
    expect(row_model).to_have_value("open_router/vendor/model-b")
    row_effort = editor.get_by_label(
        "Reasoning effort for open_router/vendor/model-b", exact=True
    )
    expect(row_effort).to_have_value("off")
    row_model_box, row_effort_box = row_model.bounding_box(), row_effort.bounding_box()
    assert row_model_box is not None and row_effort_box is not None
    assert abs(row_model_box["width"] - model_box["width"]) <= 1
    assert abs(row_effort_box["width"] - effort_box["width"]) <= 1
    assert row_model.evaluate(style) == model.evaluate(style)
    row_effort.select_option("max")
    with page.expect_response("**/admin/api/config/apply") as saved:
        page.locator("#applyButton").click()
    assert saved.value.json()["applied"] is True
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    page.reload()
    page.get_by_role("button", name="Model Config", exact=True).click()
    expect(
        editor.get_by_label(
            "Reasoning effort for open_router/vendor/model-b", exact=True
        )
    ).to_have_value("max")
    assert _fallbacks(editor) == [
        {"model": "open_router/vendor/model-b", "reasoning": "max", "api_key_number": 1}
    ]
    expect(page.locator("#dirtyState")).to_have_text("No changes")


def test_fallback_drag_and_keyboard_reordering_save_model_effort_pairs(
    page: Page, admin_base_url: str
) -> None:
    page.set_viewport_size({"width": 1280, "height": 1100})
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    entries = [
        {"model": "groq/model-a", "reasoning": "low", "api_key_number": 1},
        {"model": "groq/model-b", "reasoning": "client", "api_key_number": 1},
        {"model": "groq/model-c", "reasoning": "max", "api_key_number": 1},
        {"model": "groq/model-d", "reasoning": "high", "api_key_number": 1},
    ]
    for entry in entries:
        editor.locator("#field-MODEL_FALLBACKS-add").fill(cast(str, entry["model"]))
        editor.get_by_label("Fallback reasoning effort", exact=True).select_option(
            cast(str, entry["reasoning"])
        )
        editor.get_by_role("button", name="Add fallback", exact=True).click()
    rows = editor.locator(".model-list-row")
    target_box = rows.nth(3).bounding_box()
    assert target_box is not None
    editor.get_by_role("button", name="Move groq/model-a", exact=True).drag_to(
        rows.nth(3), target_position={"x": 10, "y": target_box["height"] - 5}
    )
    assert _fallbacks(editor) == [entries[1], entries[2], entries[3], entries[0]]
    handle = editor.get_by_role("button", name="Move groq/model-a", exact=True)
    handle.focus()
    handle.press("ArrowUp")
    expect(handle).to_be_focused()
    assert _fallbacks(editor) == [entries[1], entries[2], entries[0], entries[3]]
    handle.press("ArrowDown")
    handle.press("ArrowDown")  # The last row stays in place.
    assert _fallbacks(editor) == [entries[1], entries[2], entries[3], entries[0]]
    expect(page.locator("#field-MODEL_SONNET")).to_have_value("None")
    with page.expect_response("**/admin/api/config/apply") as saved:
        page.locator("#applyButton").click()
    assert saved.value.json()["applied"] is True
    page.reload()
    expect(page.locator("#messageArea")).to_have_text("")
    assert _fallbacks(editor) == [entries[1], entries[2], entries[3], entries[0]]
    expect(
        editor.get_by_role("combobox", name="Fallback model 1", exact=True)
    ).to_have_value("groq/model-b")


def test_fallback_plus_and_trash_have_matching_shape_and_centered_icons(
    page: Page, admin_base_url: str
) -> None:
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    editor.locator("#field-MODEL_FALLBACKS-add").fill("groq/model-a")
    plus = editor.get_by_role("button", name="Add fallback", exact=True)
    plus.click()
    trash = editor.get_by_role("button", name="Remove groq/model-a", exact=True)
    shape = "el => { const s = getComputedStyle(el); return [s.width, s.height, s.borderRadius]; }"
    assert plus.evaluate(shape) == trash.evaluate(shape)
    for button in (plus, trash):
        icon = button.locator("svg")
        expect(icon).to_be_visible()
        button_box, icon_box = button.bounding_box(), icon.bounding_box()
        assert button_box is not None and icon_box is not None
        for axis, size in (("x", "width"), ("y", "height")):
            button_center = button_box[axis] + button_box[size] / 2
            icon_center = icon_box[axis] + icon_box[size] / 2
            assert abs(button_center - icon_center) <= 0.5


def test_fallback_drag_indicators_only_offer_unique_positions_that_change_order(
    page: Page, admin_base_url: str
) -> None:
    page.set_viewport_size({"width": 1280, "height": 1100})
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    for suffix in ("a", "b", "c", "d"):
        editor.locator("#field-MODEL_FALLBACKS-add").fill(f"groq/model-{suffix}")
        editor.get_by_role("button", name="Add fallback", exact=True).click()
    editor.scroll_into_view_if_needed()
    original = _fallbacks(editor)
    rows = editor.locator(".model-list-row")
    handle = editor.get_by_role("button", name="Move groq/model-b", exact=True)
    handle_box = handle.bounding_box()
    assert handle_box is not None
    x = handle_box["x"] + handle_box["width"] / 2
    y = handle_box["y"] + handle_box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 8, y + 8, steps=5)
    expect(editor.locator(".model-list-row.dragging")).to_have_count(1)
    indicators = editor.locator(".model-list-row[data-drop-position]")
    try:
        # Both halves of the source and the two neighboring insertion points
        # retain the current order, so they should show no placement line.
        for index, after in ((1, False), (1, True), (0, True), (2, False)):
            bounds = rows.nth(index).bounding_box()
            assert bounds is not None
            page.mouse.move(
                bounds["x"] + 10,
                bounds["y"] + (bounds["height"] - 2 if after else 2),
                steps=5,
            )
            expect(indicators).to_have_count(0)
            assert _fallbacks(editor) == original
        for index, position in ((3, "after"), (0, "before")):
            bounds = rows.nth(index).bounding_box()
            assert bounds is not None
            page.mouse.move(
                bounds["x"] + 10,
                bounds["y"] + (bounds["height"] - 2 if position == "after" else 2),
                steps=5,
            )
            expect(indicators).to_have_count(1)
            expect(rows.nth(index)).to_have_attribute("data-drop-position", position)
        line_y = None
        for index, after in ((2, True), (3, False)):
            bounds = rows.nth(index).bounding_box()
            assert bounds is not None
            page.mouse.move(
                bounds["x"] + 10,
                bounds["y"] + (bounds["height"] - 2 if after else 2),
                steps=5,
            )
            expect(indicators).to_have_count(1)
            current_y = indicators.evaluate(
                "el => { const r = el.getBoundingClientRect(); return el.dataset.dropPosition === 'before' ? r.top : r.bottom; }"
            )
            if line_y is None:
                line_y = current_y
            else:
                assert current_y == line_y
        upper, lower = rows.nth(2).bounding_box(), rows.nth(3).bounding_box()
        assert upper is not None and lower is not None
        page.mouse.move(
            upper["x"] + 10,
            (upper["y"] + upper["height"] + lower["y"]) / 2,
            steps=5,
        )
        expect(indicators).to_have_count(1)
        assert (
            indicators.evaluate(
                "el => { const r = el.getBoundingClientRect(); return el.dataset.dropPosition === 'before' ? r.top : r.bottom; }"
            )
            == line_y
        )
    finally:
        page.mouse.up()
    assert _fallbacks(editor) == [original[0], original[2], original[1], original[3]]
    expect(indicators).to_have_count(0)


@pytest.mark.parametrize("invalid_model", ["", "groq/model-b"])
def test_fallback_model_edits_validate_without_losing_saved_configuration(
    page: Page, admin_base_url: str, invalid_model: str
) -> None:
    _open_models(page, admin_base_url)
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    for model in ("groq/model-a", "groq/model-b"):
        editor.locator("#field-MODEL_FALLBACKS-add").fill(model)
        editor.get_by_role("button", name="Add fallback", exact=True).click()
    with page.expect_response("**/admin/api/config/apply") as saved:
        page.locator("#applyButton").click()
    assert saved.value.json()["applied"] is True
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    original = _fallbacks(editor)
    model = editor.get_by_role("combobox", name="Fallback model 1", exact=True)
    model.fill(invalid_model)
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    model.press("Tab")
    with page.expect_response("**/admin/api/config/apply") as rejected:
        page.locator("#applyButton").click()
    assert rejected.value.json()["applied"] is False
    expect(model).to_have_value(invalid_model)
    fields = page.request.get(f"{admin_base_url}/admin/api/config").json()["fields"]
    persisted = next(field for field in fields if field["key"] == "MODEL_FALLBACKS")
    assert json.loads(persisted["value"]) == original


def test_editable_fallbacks_keep_focus_and_live_model_suggestions(
    page: Page, admin_base_url: str
) -> None:
    _open_models(page, admin_base_url)
    page.wait_for_function("!state.startupRequest && !state.startupTimer")
    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    editor.locator("#field-MODEL_FALLBACKS-add").fill("groq/custom")
    editor.get_by_role("button", name="Add fallback", exact=True).click()
    model = editor.get_by_role("combobox", name="Fallback model 1", exact=True)
    model.fill("open_router/new-model")
    expect(model).to_be_focused()
    assert _fallbacks(editor) == [
        {"model": "open_router/new-model", "reasoning": "inherit", "api_key_number": 1}
    ]
    for effort in ("low", "high", "client"):
        editor.get_by_label(
            "Reasoning effort for open_router/new-model", exact=True
        ).select_option(effort)
    model.fill("new-model")
    page.route(
        "**/admin/api/models",
        lambda route: route.fulfill(
            json={"models": ["open_router/new-model"], "failed_providers": []}
        ),
    )
    # Refresh the catalog without closing the currently edited model dropdown.
    with page.expect_response("**/admin/api/models"):
        page.evaluate("void hydrateModelOptions()")
    expect(
        editor.get_by_role("option", name="open_router/new-model", exact=True)
    ).to_be_visible()
    expect(model).to_have_value("new-model")
    editor.get_by_role("option", name="open_router/new-model", exact=True).click()
    assert _fallbacks(editor) == [
        {"model": "open_router/new-model", "reasoning": "client", "api_key_number": 1}
    ]
    registered = page.evaluate("state.modelComboboxes.size")
    for _ in range(3):
        editor.locator("#field-MODEL_FALLBACKS-add").fill("groq/discarded")
        editor.get_by_role("button", name="Add fallback", exact=True).click()
        editor.get_by_role("button", name="Remove groq/discarded", exact=True).click()
    assert page.evaluate("state.modelComboboxes.size") == registered


def test_fallback_editor_remains_usable_at_narrow_viewport(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    _open_models(page, admin_base_url)

    editor = page.locator('.field[data-key="MODEL_FALLBACKS"]')
    editor.scroll_into_view_if_needed()
    expect(editor.locator("#field-MODEL_FALLBACKS-add")).to_be_visible()
    expect(
        editor.get_by_role("button", name="Add fallback", exact=True)
    ).to_be_visible()
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
    dialog = open_provider(page, "open_router")
    with page.expect_request("**/admin/api/models"):
        dialog.get_by_role("button", name="Refresh models", exact=True).click()
    close_provider(page)
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
