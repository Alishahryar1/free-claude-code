"""Endpoint diagnostics remain visible beside selected-provider discovery."""

import re

import httpx
import pytest
from playwright.sync_api import Page, Route, expect

from e2e.provider_support import close_provider, open_provider
from free_claude_code.providers.runtime import ProviderRuntime


@pytest.fixture
def admin_client_files(monkeypatch):
    # The fixture invokes this hook before startup. Only the external provider
    # boundary fails; the manager, status API, and browser remain real.
    resolve = ProviderRuntime.resolve_provider

    async def timeout_local(runtime, provider_id, api_key_number=1):
        if provider_id in {"lmstudio", "llamacpp", "ollama"}:
            raise httpx.ReadTimeout("private-provider-error")
        return await resolve(runtime, provider_id, api_key_number)

    monkeypatch.setattr(ProviderRuntime, "resolve_provider", timeout_local)


def timeout_probe(provider_id):
    return {
        "provider_id": provider_id,
        "status": "offline",
        "message": "The local provider availability check timed out. Check that its server is responsive and retry.",
        "probe_path": "/api/tags" if provider_id == "ollama" else "/models",
        "latency_ms": 1500,
        "error_type": "timeout",
    }


@pytest.mark.parametrize("provider_id", ["lmstudio", "llamacpp", "ollama"])
def test_optional_local_absence_is_neutral_and_manual_test_reports_error(
    page: Page, admin_base_url: str, provider_id: str
):
    page.route(
        f"**/admin/api/providers/{provider_id}/local-status",
        lambda route: route.fulfill(json=timeout_probe(provider_id)),
    )
    page.goto(f"{admin_base_url}/admin")
    primary = page.locator(f'[data-provider-check-result="{provider_id}"]')
    detail = page.locator(f'[data-provider-availability-result="{provider_id}"]')
    expect(primary).to_have_text("Local server unavailable")
    expect(primary).not_to_have_class(re.compile(r"\berror\b"))
    expect(detail).to_contain_text("1500 ms")
    expect(detail).not_to_have_class(re.compile(r"\berror\b"))
    dialog = open_provider(page, provider_id)
    expect(dialog.locator("#providerDialogAvailability")).to_have_text(
        detail.inner_text()
    )
    dialog.get_by_role("button", name="Test", exact=True).click()
    expect(primary).to_contain_text("timed out")
    expect(primary).to_have_class(re.compile(r"\berror\b"))
    expect(detail).to_be_hidden()
    assert "private-provider-error" not in page.locator("body").inner_text()
    close_provider(page)
    page.evaluate("load({providersOnly: true})")
    expect(primary).to_have_text("Local server unavailable")


@pytest.mark.parametrize(
    ("admin_base_url", "provider_id"),
    [
        ({"MODEL": f"{provider}/local-model"}, provider)
        for provider in ("lmstudio", "llamacpp", "ollama")
    ],
    indirect=["admin_base_url"],
)
def test_selected_local_provider_keeps_endpoint_diagnostics_and_manual_ownership(
    page: Page, admin_base_url: str, provider_id: str
):
    page.route(
        f"**/admin/api/providers/{provider_id}/local-status",
        lambda route: route.fulfill(json=timeout_probe(provider_id)),
    )
    page.goto(f"{admin_base_url}/admin")
    primary = page.locator(f'[data-provider-check-result="{provider_id}"]')
    detail = page.locator(f'[data-provider-availability-result="{provider_id}"]')
    expect(primary).to_contain_text("timed out")
    expect(detail).to_contain_text("1500 ms")
    expect(detail).to_contain_text(
        "GET /api/tags" if provider_id == "ollama" else "GET /models"
    )
    dialog = open_provider(page, provider_id)
    expect(dialog.locator("#providerDialogAvailability")).to_have_text(
        detail.inner_text()
    )
    expect(dialog.locator("#providerDialogAvailability")).to_be_in_viewport()
    assert "private-provider-error" not in page.locator("body").inner_text()

    manual: list[Route] = []
    page.route(
        f"**/admin/api/providers/{provider_id}/test", lambda route: manual.append(route)
    )
    with page.expect_request(f"**/admin/api/providers/{provider_id}/test"):
        dialog.get_by_role("button", name="Test", exact=True).click()
    expect(primary).to_have_text("Checking...")
    expect(detail).to_be_hidden()
    expect(dialog.locator("#providerDialogAvailability")).to_be_hidden()
    manual.pop().fulfill(json={"ok": True, "models": ["local-model"]})
    expect(primary).to_have_text("1 model available")
    page.wait_for_function("!state.startupRequest")
    expect(detail).to_be_hidden()


@pytest.mark.parametrize(
    "admin_base_url", [{"MODEL": "lmstudio/local-model"}], indirect=True
)
def test_availability_detail_clears_on_dialog_switch_and_configuration_reload(
    page: Page, admin_base_url: str
):
    page.route(
        "**/admin/api/providers/lmstudio/local-status",
        lambda route: route.fulfill(json=timeout_probe("lmstudio")),
    )
    page.goto(f"{admin_base_url}/admin")
    detail = page.locator('[data-provider-availability-result="lmstudio"]')
    expect(detail).to_contain_text("1500 ms")
    dialog = open_provider(page, "lmstudio")
    expect(dialog.locator("#providerDialogAvailability")).to_be_visible()
    close_provider(page)
    dialog = open_provider(page, "open_router")
    expect(dialog.locator("#providerDialogAvailability")).to_be_hidden()
    close_provider(page)
    open_provider(page, "lmstudio")
    expect(dialog.locator("#providerDialogAvailability")).to_be_visible()
    close_provider(page)
    page.locator("#addCustomProvider").click()
    expect(dialog.locator("#providerDialogAvailability")).to_be_hidden()
    close_provider(page)

    pending: list[Route] = []
    page.route(
        "**/admin/api/providers/lmstudio/local-status",
        lambda route: pending.append(route),
    )
    with page.expect_request("**/admin/api/providers/lmstudio/local-status"):
        page.evaluate("load({providersOnly: true})")
    expect(detail).to_be_hidden()
    dialog = open_provider(page, "lmstudio")
    expect(dialog.locator("#providerDialogAvailability")).to_be_hidden()
    pending.pop().fulfill(json={**timeout_probe("lmstudio"), "latency_ms": 2500})
    expect(detail).to_contain_text("2500 ms")
    expect(dialog.locator("#providerDialogAvailability")).to_have_text(
        detail.inner_text()
    )
