import pytest
from playwright.sync_api import expect

from e2e.provider_support import api_key_input, open_provider
from e2e.test_code_sessions import create_session


def test_code_bootstrap_cannot_stall_other_pages_status(page, admin_base_url):
    pending = []
    starting = True

    def status(route):
        payload = route.fetch().json()
        payload["startup"]["providers"]["open_router"] = (
            "starting" if starting else "ready"
        )
        route.fulfill(json=payload)

    def hold(route):
        pending.append(route)
        page.evaluate("window.heldBootstrap = true")

    page.route("**/admin/api/status", status)
    page.route("**/admin/api/code/bootstrap", hold)
    page.goto(f"{admin_base_url}/admin")
    page.wait_for_function("window.heldBootstrap === true")
    expect(page.locator('[data-provider-check-result="open_router"]')).to_have_text(
        "Checking models…"
    )
    starting = False
    expect(page.locator('[data-provider-check-result="open_router"]')).to_have_text(
        "3 models available"
    )
    assert len(pending) == 1
    pending.pop().fulfill(
        response=page.request.get(f"{admin_base_url}/admin/api/code/bootstrap")
    )
    page.unroute("**/admin/api/code/bootstrap", hold)


def test_partial_catalog_keeps_code_selection_and_send_usable(
    page, admin_base_url, tmp_path
):
    def bootstrap(route):
        payload = route.fetch().json()
        assert payload["models"]
        payload["startup"]["catalog"] = "starting"
        route.fulfill(json=payload)

    page.route("**/admin/api/code/bootstrap", bootstrap)
    create_session(page, admin_base_url, tmp_path)
    page.locator("#codeComposer").fill("Use the available model")
    expect(page.locator("#codeProvider")).to_be_enabled()
    expect(page.locator("#codeSend")).to_be_enabled()


def test_saved_settings_finish_before_unrelated_account_check(page, admin_base_url):
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    open_provider(page, "nvidia_nim")
    api_key_input(page, "NVIDIA_NIM_API_KEY").fill("test-new-key")
    pending = []

    def hold(route):
        pending.append(route)
        page.evaluate("window.heldAccountCheck = true")

    page.route("**/admin/api/providers/github_copilot/auth", hold)
    page.locator("#saveProvider").click()
    page.wait_for_function("window.heldAccountCheck === true")
    expect(page.locator("#messageArea")).to_have_text("Applied")
    assert not page.evaluate("state.applying")
    for request in pending:
        request.fulfill(
            json={
                "provider_id": "github_copilot",
                "connected": False,
                "state": "disconnected",
            }
        )
    page.unroute("**/admin/api/providers/github_copilot/auth", hold)
    expect(page.locator("#messageArea")).to_have_text("Applied")


def test_provider_refresh_finishes_before_optional_model_suggestions(
    page, admin_base_url
):
    page.goto(f"{admin_base_url}/admin")
    dialog = open_provider(page, "open_router")
    button = dialog.get_by_role("button", name="Refresh models", exact=True)
    expect(button).to_be_enabled()
    pending = []

    def hold(route):
        pending.append(route)
        page.evaluate("window.heldModelSuggestions = true")

    page.route("**/admin/api/models", hold)
    button.click()
    page.wait_for_function("window.heldModelSuggestions === true")
    expect(button).to_be_enabled()
    expect(dialog.locator("#providerDialogCheck")).to_contain_text("models available")
    for request in tuple(pending):
        request.fulfill(response=request.fetch())
    page.unroute("**/admin/api/models", hold)


def test_refresh_all_finishes_without_waiting_for_code(page, admin_base_url):
    page.goto(f"{admin_base_url}/admin")
    page.get_by_role("button", name="Model Config", exact=True).click()
    button = page.locator("[data-startup-catalog]")
    expect(button).to_be_enabled()
    pending = []

    def hold(route):
        pending.append(route)
        page.evaluate("window.heldRefreshBootstrap = true")

    page.route("**/admin/api/code/bootstrap", hold)
    button.click()
    page.wait_for_function("window.heldRefreshBootstrap === true")
    expect(button).to_be_enabled()
    expect(button).to_have_text("Refresh models")
    expect(page.locator("#messageArea")).to_contain_text("models available")
    for request in tuple(pending):
        request.fulfill(response=request.fetch())
    page.unroute("**/admin/api/code/bootstrap", hold)


@pytest.mark.parametrize(
    "held_id,held_button",
    [
        ("claude-vscode", "openClaudeIntegration"),
        ("vscode-chat", "openVSCodeChatIntegration"),
        ("codex", "openCodexIntegration"),
        ("claude-desktop", "openClaudeDesktopIntegration"),
        ("jetbrains-acp", "openJetBrainsIntegration"),
    ],
)
def test_each_integration_finishes_without_slowest_card(
    page, admin_base_url, held_id, held_button
):
    pending = []

    def hold(route):
        pending.append(route)
        page.evaluate("window.heldChecks = (window.heldChecks || 0) + 1")

    page.route(f"**/admin/api/integrations/{held_id}", hold)
    page.goto(f"{admin_base_url}/admin/integrations")
    for visit in range(2):
        if visit:
            page.get_by_role("button", name="Providers", exact=True).click()
            page.get_by_role("button", name="Integrations", exact=True).click()
        page.wait_for_function("count => window.heldChecks === count", arg=visit + 1)
        expect(page.locator(f"#{held_button}")).to_have_text("Loading…")
        for button in [
            "openClaudeIntegration",
            "openVSCodeChatIntegration",
            "openCodexIntegration",
            "openClaudeDesktopIntegration",
            "openJetBrainsIntegration",
        ]:
            if button != held_button:
                expect(page.locator(f"#{button}")).to_have_text("Connect")
                expect(page.locator(f"#{button}")).to_be_enabled()
        pending.pop().fulfill(status=503, json={"detail": "This check failed"})
        expect(page.locator(f"#{held_button}")).to_have_text("Retry")
        expect(page.locator(f"#{held_button}")).to_be_enabled()


def test_local_cards_render_before_slowest_check(page, admin_base_url):
    pending = []

    def hold(route):
        pending.append(route)
        page.evaluate("window.heldLocalCheck = true")

    page.route(
        "**/admin/api/providers/ollama/local-status",
        hold,
    )
    page.route(
        "**/admin/api/providers/lmstudio/local-status",
        lambda route: route.fulfill(
            json={
                "provider_id": "lmstudio",
                "status": "reachable",
                "base_url": "http://localhost:1234/v1",
            }
        ),
    )
    page.route(
        "**/admin/api/providers/llamacpp/local-status",
        lambda route: route.fulfill(status=503, json={"detail": "Check failed"}),
    )
    with (
        page.expect_response(
            "**/admin/api/providers/lmstudio/local-status"
        ) as lmstudio_response,
        page.expect_response(
            "**/admin/api/providers/llamacpp/local-status"
        ) as llamacpp_response,
    ):
        page.goto(f"{admin_base_url}/admin")
        page.wait_for_function("window.heldLocalCheck === true")
    lmstudio_response.value.finished()
    llamacpp_response.value.finished()
    expect(page.locator('[data-provider-availability-result="lmstudio"]')).to_have_text(
        "Reachable: http://localhost:1234/v1"
    )
    expect(page.locator('[data-provider-availability-result="llamacpp"]')).to_have_text(
        "Availability check failed. Use Test to retry."
    )
    assert len(pending) == 1
    expect(page.locator('[data-provider-availability-result="ollama"]')).to_be_hidden()
    pending.pop().fulfill(
        json={
            "provider_id": "ollama",
            "status": "reachable",
            "base_url": "http://localhost:11434",
        }
    )
    expect(page.locator('[data-provider-availability-result="ollama"]')).to_have_text(
        "Reachable: http://localhost:11434"
    )


def test_integration_check_from_previous_config_cannot_overwrite_new_result(
    page, admin_base_url
):
    pending = []

    def hold(route):
        pending.append(route)
        page.evaluate("window.statusChecks = (window.statusChecks || 0) + 1")

    page.route("**/admin/api/integrations/vscode-chat", hold)
    page.goto(f"{admin_base_url}/admin/integrations")
    page.wait_for_function("window.statusChecks === 1")
    expect(page.locator("#openJetBrainsIntegration")).to_be_enabled()
    page.evaluate("void load()")
    page.wait_for_function("window.statusChecks === 2", timeout=2000)
    old, current = pending
    current.fulfill(
        json={
            "connected": True,
            "paths": None,
            "update": {"state": "ready", "changed": False, "message": None},
        }
    )
    expect(page.locator("#openVSCodeChatIntegration")).to_have_text("Disconnect")
    with page.expect_response("**/admin/api/integrations/vscode-chat") as response:
        old.fulfill(
            json={
                "connected": False,
                "paths": None,
                "update": {"state": "ready", "changed": False, "message": None},
            }
        )
    response.value.finished()
    page.evaluate("() => new Promise(requestAnimationFrame)")
    expect(page.locator("#openVSCodeChatIntegration")).to_have_text("Disconnect")
    expect(page.locator("#openVSCodeChatIntegration")).to_be_enabled()
