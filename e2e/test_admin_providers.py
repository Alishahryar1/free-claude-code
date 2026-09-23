"""Rendered provider-setup regressions for the local Admin UI."""

import pytest
from playwright.sync_api import ConsoleMessage, Page, Route, ViewportSize, expect

from e2e.provider_support import close_provider, open_provider


@pytest.mark.parametrize(
    "admin_base_url", [{"MODEL": "github_models/openai/old"}], indirect=True
)
def test_retired_provider_is_absent_and_default_setup_remains_available(
    page: Page, admin_base_url: str
):
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    expect(page.locator('[data-provider="github_models"]')).to_have_count(0)
    expect(page.locator("#field-GITHUB_MODELS_TOKEN")).to_have_count(0)
    card = page.locator('[data-provider="nvidia_nim"]')
    expect(card.get_by_role("button", name="Configure", exact=True)).to_have_class(
        "primary-button"
    )
    card.locator("[data-provider-settings]").click()
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_focused()
    close_provider(page)
    page.get_by_role("button", name="Model Config", exact=True).click()
    expect(page.locator("#field-MODEL")).to_have_value(
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    )


def _open_admin(
    page: Page,
    admin_base_url: str,
    viewport: ViewportSize,
) -> None:
    page.set_viewport_size(viewport)
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")


@pytest.mark.parametrize(
    ("viewport", "desktop"),
    (
        ({"width": 1280, "height": 720}, True),
        ({"width": 390, "height": 844}, False),
    ),
)
def test_missing_provider_configuration_opens_modal_and_focuses_exact_field(
    page: Page,
    admin_base_url: str,
    viewport: ViewportSize,
    desktop: bool,
) -> None:
    _open_admin(page, admin_base_url, viewport)
    card = page.locator('[data-provider="nvidia_nim"]')
    key_input = page.locator("#field-NVIDIA_NIM_API_KEY")

    expect(card.get_by_role("button", name="Configure", exact=True)).to_have_class(
        "primary-button"
    )
    expect(card.locator("[data-provider-settings]")).to_have_attribute(
        "aria-haspopup", "dialog"
    )
    expect(card.get_by_role("button", name="Refresh models", exact=True)).to_have_count(
        0
    )
    expect(key_input).to_have_count(0)

    card.locator("[data-provider-settings]").click()

    expect(key_input).to_be_in_viewport()
    expect(key_input).to_be_focused()
    if desktop:
        sidebar = page.locator(".sidebar")
        expect(sidebar).to_have_css("position", "sticky")
        assert (
            round(
                float(
                    sidebar.evaluate("element => element.getBoundingClientRect().top")
                )
            )
            == 0
        )


def test_desktop_sidebar_stays_pinned_at_document_bottom(
    page: Page,
    admin_base_url: str,
) -> None:
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    sidebar = page.locator(".sidebar")

    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")

    sidebar_top = float(
        sidebar.evaluate("element => element.getBoundingClientRect().top")
    )
    assert sidebar_top == pytest.approx(0, abs=0.5)


def test_configured_provider_check_keeps_readiness_and_adds_models(
    page: Page,
    admin_base_url: str,
) -> None:
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="open_router"]')

    expect(card.locator(".provider-check-result")).to_have_text("3 models available")
    expect(card.get_by_role("button", name="Edit", exact=True)).to_have_class(
        "secondary-button"
    )
    expect(card.locator("[data-provider-settings]")).to_have_attribute(
        "aria-haspopup", "dialog"
    )
    card.locator("[data-provider-settings]").click()
    page.locator("#providerDialog").get_by_role(
        "button", name="Refresh models", exact=True
    ).click()

    expect(card.locator(".provider-check-result")).to_have_text("3 models available")
    expect(card.get_by_role("button", name="Edit", exact=True)).to_have_class(
        "secondary-button"
    )

    close_provider(page)
    page.get_by_role("button", name="Model Config", exact=True).click()
    fable = page.get_by_role(
        "combobox",
        name="Fable Override default",
        exact=True,
    )
    page.get_by_role("button", name="Show Fable Override options", exact=True).click()
    expect(page.get_by_role("listbox").get_by_role("option")).to_have_count(1)
    expect(page.get_by_role("option", name="None", exact=True)).to_be_visible()
    fable.fill("vendor/model-a")
    expect(
        page.get_by_role("option", name="open_router/vendor/model-a", exact=True)
    ).to_be_visible()


@pytest.mark.parametrize("models", [[], ["only-model"]])
def test_startup_model_count_handles_empty_and_single_model_catalogs(
    page: Page, admin_base_url: str, models: list[str]
) -> None:
    def status(route: Route) -> None:
        data = route.fetch().json()
        data["startup"]["providers"]["open_router"] = "ready"
        data["cached_models"]["open_router"] = models
        route.fulfill(json=data)

    page.route("**/admin/api/status", status)
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    expect(page.locator('[data-provider-check-result="open_router"]')).to_have_text(
        "1 model available" if models else "0 models available"
    )
    expect(page.locator('[data-provider-check-result="nvidia_nim"]')).to_be_hidden()


@pytest.mark.parametrize("manual_result", ["pending", "success", "failure"])
def test_delayed_startup_status_does_not_replace_a_manual_provider_check(
    page: Page, admin_base_url: str, manual_result: str
) -> None:
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    page.wait_for_function("!state.startupRequest && !state.startupTimer")
    snapshot = page.request.get(f"{admin_base_url}/admin/api/status").json()
    startup: list[Route] = []
    manual: list[Route] = []
    page.route("**/admin/api/status", lambda route: startup.append(route))
    page.route(
        "**/admin/api/providers/open_router/test", lambda route: manual.append(route)
    )
    with page.expect_request("**/admin/api/status"):
        page.evaluate("void refreshStartup()")
    dialog = open_provider(page, "open_router")
    with page.expect_request("**/admin/api/providers/open_router/test"):
        dialog.get_by_role("button", name="Refresh models", exact=True).click()
    expected = "Checking..."
    if manual_result != "pending":
        manual.pop().fulfill(
            json={
                "ok": manual_result == "success",
                "models": ["one", "two"],
                "message": "Could not refresh this provider's models.",
            }
        )
        expected = (
            "2 models available"
            if manual_result == "success"
            else "Unavailable: Could not refresh this provider's models."
        )
    result = page.locator('[data-provider-check-result="open_router"]')
    expect(result).to_have_text(expected)
    with page.expect_response("**/admin/api/status") as response:
        startup.pop(0).fulfill(json=snapshot)
    response.value.finished()
    page.wait_for_function("!state.startupRequest")
    expect(result).to_have_text(expected)
    expect(dialog.locator("#providerDialogCheck")).to_have_text(expected)
    if manual_result == "pending":
        manual.pop().fulfill(json={"ok": True, "models": ["one", "two"]})
        expect(result).to_have_text("2 models available")


@pytest.mark.parametrize("availability_first", [False, True])
def test_local_model_discovery_takes_precedence_over_reachability(
    page: Page, admin_base_url: str, availability_first: bool
) -> None:
    startup: list[Route] = []
    availability: list[Route] = []
    page.route("**/admin/api/status", lambda route: startup.append(route))
    page.route(
        "**/admin/api/providers/local-status", lambda route: availability.append(route)
    )
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    page.wait_for_function("!!state.startupRequest && !!state.localStatusRequest")
    snapshot = page.request.get(f"{admin_base_url}/admin/api/status").json()
    snapshot["startup"]["providers"]["lmstudio"] = "ready"
    snapshot["cached_models"]["lmstudio"] = ["local-model"]
    result = page.locator('[data-provider-check-result="lmstudio"]')
    if availability_first:
        availability.pop().continue_()
        expect(result).to_have_text("Reachable: http://localhost:1234/v1")
    startup.pop(0).fulfill(json=snapshot)
    expect(result).to_have_text("1 model available")
    if not availability_first:
        availability.pop().continue_()
        page.wait_for_function("!state.localStatusRequest")
        expect(result).to_have_text("1 model available")


def test_provider_check_failure_is_separate_and_never_exposes_exception_text(
    page: Page,
    admin_base_url: str,
) -> None:
    console_messages: list[str] = []

    def record_console(message: ConsoleMessage) -> None:
        console_messages.append(message.text)

    page.on("console", record_console)
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="groq"]')
    expect(card.locator(".provider-check-result")).to_have_text(
        "Could not load models. Check the provider's settings and retry."
    )
    card.locator("[data-provider-settings]").click()
    page.locator("#providerDialog").get_by_role(
        "button", name="Refresh models", exact=True
    ).click()

    result = card.locator(".provider-check-result")
    expect(result).to_have_text(
        "Unavailable: Could not refresh this provider's models. "
        "Verify its configuration and access."
    )
    expect(card.get_by_role("button", name="Edit", exact=True)).to_have_class(
        "secondary-button"
    )
    page_text = page.locator("body").inner_text()
    secret = "CREDENTIAL[unrecognized-format-987654321]"
    assert secret not in page_text
    assert "RuntimeError" not in page_text
    assert secret not in "\n".join(console_messages)


def test_multi_field_provider_targets_first_missing_configuration(
    page: Page,
    admin_base_url: str,
) -> None:
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="cloudflare"]')
    account_input = page.locator("#field-CLOUDFLARE_ACCOUNT_ID")

    expect(card.get_by_role("button", name="Configure", exact=True)).to_have_class(
        "primary-button"
    )
    expect(account_input).to_have_count(0)

    card.locator("[data-provider-settings]").click()

    expect(account_input).to_be_in_viewport()
    expect(account_input).to_be_focused()


def test_admin_loading_finishes_before_local_availability_checks(
    page: Page, admin_base_url: str
) -> None:
    pending: list[Route] = []
    page.route(
        "**/admin/api/providers/local-status", lambda route: pending.append(route)
    )
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    open_provider(page, "nvidia_nim")
    key = page.locator("#field-NVIDIA_NIM_API_KEY")
    key.fill("unsaved-key")
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    expect(page.locator("#saveProvider")).to_be_enabled()

    route = pending.pop()
    payload = route.fetch().json()
    providers = {provider["provider_id"]: provider for provider in payload["providers"]}
    providers["llamacpp"].update(status="offline", label="Offline", status_code=503)
    providers["ollama"].update(status="missing_url", label="Missing URL", base_url="")
    route.fulfill(json=payload)
    expect(page.locator('[data-provider-check-result="lmstudio"]')).to_have_text(
        "Reachable: http://localhost:1234/v1"
    )
    expect(page.locator('[data-provider-check-result="llamacpp"]')).to_have_text(
        "Unavailable: http://localhost:8080/v1 returned HTTP 503"
    )
    expect(page.locator('[data-provider-check-result="ollama"]')).to_be_hidden()
    expect(
        page.locator('[data-provider="lmstudio"]').get_by_role(
            "button", name="Edit", exact=True
        )
    ).to_have_class("secondary-button")
    expect(key).to_have_value("unsaved-key")
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    expect(page.locator("#messageArea")).to_have_text("")


@pytest.mark.parametrize("failure", ["http", "network"])
def test_local_availability_failure_does_not_fail_admin_loading(
    page: Page, admin_base_url: str, failure: str
) -> None:
    pending: list[Route] = []
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route(
        "**/admin/api/providers/local-status", lambda route: pending.append(route)
    )
    with page.expect_request("**/admin/api/providers/local-status"):
        page.goto(f"{admin_base_url}/admin")
    open_provider(page, "nvidia_nim")
    expect(page.locator("#field-NVIDIA_NIM_API_KEY")).to_be_editable()
    close_provider(page)
    if failure == "http":
        pending.pop().fulfill(status=503, json={"detail": "private-diagnostic-marker"})
    else:
        pending.pop().abort()

    for provider_id in ("lmstudio", "llamacpp", "ollama"):
        card = page.locator(f'[data-provider="{provider_id}"]')
        expect(card.locator(".provider-check-result")).to_have_text(
            "Availability check failed. Use Test to retry."
        )
        expect(card.get_by_role("button", name="Edit", exact=True)).to_have_class(
            "secondary-button"
        )
        dialog = open_provider(page, provider_id)
        expect(dialog.get_by_role("button", name="Test", exact=True)).to_be_enabled()
        close_provider(page)
    expect(page.locator('[data-provider-check-result="open_router"]')).to_have_text(
        "3 models available"
    )
    expect(page.locator("#messageArea")).to_have_text("")
    assert "private-diagnostic-marker" not in page.locator("body").inner_text()
    assert errors == []


@pytest.mark.parametrize("manual_finished", [False, True])
def test_manual_provider_test_takes_precedence_over_automatic_availability(
    page: Page, admin_base_url: str, manual_finished: bool
) -> None:
    availability: list[Route] = []
    manual: list[Route] = []
    page.route(
        "**/admin/api/providers/local-status", lambda route: availability.append(route)
    )
    page.route(
        "**/admin/api/providers/lmstudio/test", lambda route: manual.append(route)
    )
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="lmstudio"]')
    dialog = open_provider(page, "lmstudio")
    with page.expect_request("**/admin/api/providers/lmstudio/test"):
        dialog.get_by_role("button", name="Test", exact=True).click()
    result = card.locator(".provider-check-result")
    expect(result).to_have_text("Checking...")
    if manual_finished:
        manual.pop().fulfill(
            json={
                "provider_id": "lmstudio",
                "ok": False,
                "message": "Could not refresh this provider's models.",
            }
        )
        expect(result).to_have_text(
            "Unavailable: Could not refresh this provider's models."
        )

    with page.expect_response("**/admin/api/providers/local-status") as response:
        if manual_finished:
            availability.pop().fulfill(status=503, json={"detail": "Check failed"})
        else:
            availability.pop().continue_()
    response.value.finished()
    page.evaluate("() => new Promise(requestAnimationFrame)")
    other = page.locator('[data-provider-check-result="ollama"]')
    if manual_finished:
        expect(result).to_have_text(
            "Unavailable: Could not refresh this provider's models."
        )
        expect(other).to_have_text("Availability check failed. Use Test to retry.")
    else:
        expect(result).to_have_text("Checking...")
        expect(other).to_have_text("Reachable: http://localhost:11434")
        manual.pop().fulfill(
            json={"provider_id": "lmstudio", "ok": True, "models": ["local-model"]}
        )
        expect(result).to_have_text("1 model available")
    expect(dialog.get_by_role("button", name="Test", exact=True)).to_be_enabled()
