"""Numbered secret rows keep their saved identity through edits and checks."""

import json

import pytest
from playwright.sync_api import Page, expect

from e2e.provider_support import close_provider, open_provider
from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.providers.credential_validation import (
    CredentialCheck,
    CredentialStatus,
)


def add_key(dialog, value):
    dialog.get_by_role("button", name="Add key", exact=True).click()
    inputs = dialog.locator(".api-key-row input")
    inputs.last.fill(value)


def test_keyboard_reorder_replace_remove_and_reopen_preserve_saved_keys(
    page: Page, admin_base_url: str, tmp_path
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    for value in ("first-secret", "second-secret", "third-secret"):
        add_key(dialog, value)
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator(".api-key-row input")).to_have_count(3)
    for input in dialog.locator(".api-key-row input").all():
        expect(input).to_have_value("")
        assert input.get_attribute("type") == "password"
    handle = dialog.get_by_role("button", name="Move Key 3", exact=True)
    handle.focus()
    handle.press("Space")
    page.keyboard.press("ArrowUp")
    page.keyboard.press("ArrowUp")
    page.keyboard.press("Space")
    expect(dialog.locator(".api-key-number")).to_have_text(["1", "2", "3"])
    dialog.locator(".api-key-row input").nth(1).fill("replacement-secret")
    dialog.get_by_role("button", name="Remove Key 3", exact=True).click()
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    keys = json.loads(
        dotenv_values_from_file(tmp_path / ".fcc/.env")["NVIDIA_NIM_API_KEY"]
    )
    assert keys == ["third-secret", "replacement-secret"]
    page.reload()
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator(".api-key-row input")).to_have_count(2)
    dialog.get_by_role("button", name="Remove Key 1", exact=True).click()
    dialog.get_by_role("button", name="Remove Key 1", exact=True).click()
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    assert "NVIDIA_NIM_API_KEY" not in dotenv_values_from_file(tmp_path / ".fcc/.env")


def test_adding_row_finishes_move_before_next_keyboard_cancel(
    page: Page, admin_base_url: str, tmp_path
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    add_key(dialog, "draft-first")
    add_key(dialog, "draft-second")
    handle = dialog.get_by_role("button", name="Move Key 1", exact=True)
    handle.press("Space")
    handle.press("ArrowDown")
    add_key(dialog, "draft-third")

    handle = dialog.get_by_role("button", name="Move Key 2", exact=True)
    handle.press("Space")
    handle.press("ArrowUp")
    page.keyboard.press("Escape")
    expect(dialog).to_be_visible()
    inputs = dialog.locator(".api-key-row input")
    expect(inputs).to_have_count(3)
    for input, value in zip(
        inputs.all(), ("draft-second", "draft-first", "draft-third"), strict=True
    ):
        expect(input).to_have_value(value)
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    assert json.loads(
        dotenv_values_from_file(tmp_path / ".fcc/.env")["NVIDIA_NIM_API_KEY"]
    ) == ["draft-second", "draft-first", "draft-third"]
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator(".api-key-row")).to_have_count(3)
    for input in dialog.locator(".api-key-row input").all():
        expect(input).to_have_value("")


def test_actual_drag_reorders_without_exposing_saved_values(
    page: Page, admin_base_url: str, tmp_path
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    for value in ("drag-first", "drag-second", "drag-third"):
        add_key(dialog, value)
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    dialog = open_provider(page, "nvidia_nim")
    rows = dialog.locator(".api-key-row")
    icon = (
        dialog.get_by_role("button", name="Move Key 1", exact=True)
        .locator("svg")
        .bounding_box()
    )
    assert icon is not None and icon["width"] >= 12 and icon["height"] >= 12
    bounds = rows.last.bounding_box()
    assert bounds is not None
    dialog.get_by_role("button", name="Move Key 1", exact=True).drag_to(
        rows.last, target_position={"x": 10, "y": bounds["height"] - 5}
    )
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    assert json.loads(
        dotenv_values_from_file(tmp_path / ".fcc/.env")["NVIDIA_NIM_API_KEY"]
    ) == ["drag-second", "drag-third", "drag-first"]


def test_blank_duplicates_and_cancel_preserve_the_whole_draft(
    page: Page, admin_base_url: str
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    add_key(dialog, "same-key")
    add_key(dialog, " same-key ")
    add_key(dialog, " ")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.locator('.api-key-row input[aria-invalid="true"]')).to_have_count(3)
    expect(dialog.locator(".api-key-row input").first).to_be_focused()
    close_provider(page)
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator(".api-key-row")).to_have_count(0)
    expect(page.locator("#saveProvider")).to_be_disabled()


def test_warning_save_refreshes_masks_and_is_clean_then_rechecks_retained_rows(
    page: Page, admin_base_url: str, monkeypatch, tmp_path
):
    checked = []

    async def warnings(settings, submissions):
        checked.append([key.get_secret_value() for key in submissions[0].credentials])
        return tuple(
            CredentialCheck(
                submission.key,
                CredentialStatus.UNVERIFIED if index else CredentialStatus.VERIFIED,
                "Verification timed out." if index else "Accepted.",
                index,
                submission.provider_id,
            )
            for submission in submissions
            for index, _ in enumerate(submission.credentials)
        )

    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials", warnings
    )
    page.goto(admin_base_url + "/admin")
    page.locator("#field-HTTP_READ_TIMEOUT").fill("181")
    dialog = open_provider(page, "nvidia_nim")
    add_key(dialog, "warning-first")
    add_key(dialog, "warning-second")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.locator("#providerMessage")).to_contain_text("Saved")
    expect(dialog.locator(".api-key-result").nth(1)).to_contain_text("timed out")
    expect(page.locator("#saveProvider")).to_be_disabled()
    expect(dialog.locator(".api-key-row input").nth(1)).to_have_value("")
    dialog.locator("#field-NVIDIA_NIM_PROXY").fill("http://localhost:8888")
    with page.expect_response("**/admin/api/config/apply") as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok
    expect(dialog.locator("#providerMessage")).to_contain_text("Saved")
    expect(page.locator("#saveProvider")).to_be_disabled()
    assert checked == [
        ["warning-first", "warning-second"],
        ["warning-first", "warning-second"],
    ]
    expect(page.locator("#field-HTTP_READ_TIMEOUT")).to_have_value("181")
    assert "warning-second" not in page.locator("#providerMessage").inner_text()


@pytest.mark.parametrize(
    "admin_base_url",
    [{"NVIDIA_NIM_API_KEY": '["locked-one","locked-two"]'}],
    indirect=True,
)
def test_process_owned_list_disables_all_editing_controls(
    page: Page, admin_base_url: str
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator(".api-key-row input")).to_have_count(2)
    for control in dialog.locator(
        ".api-key-editor input, .api-key-editor button"
    ).all():
        expect(control).to_be_disabled()


def test_conflict_preserves_draft_and_reopening_loads_fresh_references(
    page: Page, admin_base_url: str
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    add_key(dialog, "stale-draft")
    page.route(
        "**/admin/api/config/apply",
        lambda route: route.fulfill(
            status=409, json={"detail": "Configuration changed. Reopen the modal."}
        ),
    )
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.locator("#providerMessage")).to_contain_text("Reopen")
    expect(dialog.locator(".api-key-row input")).to_have_value("stale-draft")
    expect(page.locator("#saveProvider")).to_be_disabled()
    close_provider(page)
    page.unroute("**/admin/api/config/apply")
    dialog = open_provider(page, "nvidia_nim")
    add_key(dialog, "fresh-draft")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()


@pytest.mark.parametrize(
    "provider_pair,key",
    [
        (("zai", "zai_api"), "ZAI_API_KEY"),
        (("opencode_zen", "opencode_go"), "OPENCODE_API_KEY"),
    ],
)
def test_either_shared_provider_edits_the_same_ordered_list(
    page: Page, admin_base_url: str, tmp_path, provider_pair, key
):
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, provider_pair[0])
    add_key(dialog, "shared-first")
    add_key(dialog, "shared-second")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    dialog = open_provider(page, provider_pair[1])
    expect(dialog.locator(".api-key-row")).to_have_count(2)
    handle = dialog.get_by_role("button", name="Move Key 2", exact=True)
    handle.press("Space")
    handle.press("ArrowUp")
    page.keyboard.press("Space")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    assert json.loads(dotenv_values_from_file(tmp_path / ".fcc/.env")[key]) == [
        "shared-second",
        "shared-first",
    ]


def test_long_list_scrolls_and_escape_cancels_only_the_keyboard_move(
    page: Page, admin_base_url: str, tmp_path
):
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    for number in range(33):
        add_key(dialog, f"long-list-key-{number}")
    expect(dialog.locator(".api-key-number").last).to_have_text("33")
    assert dialog.evaluate("element => element.scrollHeight > element.clientHeight")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog).not_to_be_visible()
    assert (
        len(
            json.loads(
                dotenv_values_from_file(tmp_path / ".fcc/.env")["NVIDIA_NIM_API_KEY"]
            )
        )
        == 33
    )
    dialog = open_provider(page, "nvidia_nim")
    handle = dialog.get_by_role("button", name="Move Key 33", exact=True)
    handle.press("Space")
    handle.press("ArrowUp")
    page.keyboard.press("Escape")
    expect(dialog).to_be_visible()
    expect(page.locator("#saveProvider")).to_be_disabled()
    dialog.get_by_role("button", name="Cancel", exact=True).scroll_into_view_if_needed()
    expect(dialog.get_by_role("button", name="Cancel", exact=True)).to_be_in_viewport()


def test_custom_warning_save_and_rename_retain_masked_rows(
    page: Page, admin_base_url: str, monkeypatch, tmp_path
):
    async def unavailable(settings, submissions):
        return tuple(
            CredentialCheck(
                submission.key,
                CredentialStatus.UNVERIFIED,
                "Verification unavailable.",
                index,
                submission.provider_id,
            )
            for submission in submissions
            for index, _ in enumerate(submission.credentials)
        )

    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials", unavailable
    )
    page.goto(admin_base_url + "/admin")
    page.get_by_role("button", name="Add provider", exact=True).click()
    dialog = page.locator("#providerDialog")
    dialog.get_by_label("Name", exact=True).fill("Warning gateway")
    dialog.get_by_label("Base URL", exact=True).fill("https://gateway.example/v1")
    dialog.get_by_label("Model IDs", exact=True).fill("one")
    add_key(dialog, "custom-first")
    add_key(dialog, "custom-second")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.locator("#providerMessage")).to_contain_text("Saved")
    expect(dialog.locator(".api-key-result").last).to_contain_text("unavailable")
    expect(page.locator("#saveProvider")).to_be_disabled()
    dialog.get_by_label("Name", exact=True).fill("Renamed gateway")
    with page.expect_response("**/admin/api/config/apply") as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.json()["applied"] is True
    expect(dialog.locator("#providerMessage")).to_contain_text("Saved")
    expect(page.locator("#saveProvider")).to_be_disabled()
    expect(dialog).to_have_attribute("open", "")
    definitions = json.loads(
        dotenv_values_from_file(tmp_path / ".fcc/.env")["FCC_CUSTOM_PROVIDERS"]
    )
    assert definitions[0]["display_name"] == "Renamed gateway"
    assert definitions[0]["api_keys"] == ["custom-first", "custom-second"]


def test_saved_warning_followed_by_reload_failure_requires_fresh_masks(
    page: Page, admin_base_url: str, monkeypatch
):
    async def unavailable(settings, submissions):
        return tuple(
            CredentialCheck(
                submission.key,
                CredentialStatus.UNVERIFIED,
                "Verification unavailable.",
                index,
                submission.provider_id,
            )
            for submission in submissions
            for index, _ in enumerate(submission.credentials)
        )

    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials", unavailable
    )
    page.goto(admin_base_url + "/admin")
    dialog = open_provider(page, "nvidia_nim")
    add_key(dialog, "saved-before-reload-failed")
    page.route(
        "**/admin/api/config",
        lambda route: route.fulfill(status=503, json={"detail": "Reload unavailable"}),
    )
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.locator("#providerMessage")).to_contain_text(
        "could not reload settings"
    )
    expect(page.locator("#saveProvider")).to_be_disabled()
    expect(dialog.locator(".api-key-row input")).to_have_value(
        "saved-before-reload-failed"
    )
    close_provider(page)
    page.unroute("**/admin/api/config")
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator(".api-key-row input")).to_have_count(1)
    expect(dialog.locator(".api-key-row input")).to_have_value("")
    expect(page.locator("#saveProvider")).to_be_disabled()
