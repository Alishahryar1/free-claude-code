import pytest
from playwright.sync_api import expect


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
    page.route(
        "**/admin/api/providers/ollama/local-status",
        lambda route: pending.append(route),
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
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator('[data-provider-check-result="lmstudio"]')).to_have_text(
        "Reachable: http://localhost:1234/v1", timeout=2000
    )
    expect(page.locator('[data-provider-check-result="llamacpp"]')).to_have_text(
        "Availability check failed. Use Test to retry."
    )
    assert len(pending) == 1
    expect(page.locator('[data-provider-check-result="ollama"]')).to_be_hidden()
    pending.pop().fulfill(
        json={
            "provider_id": "ollama",
            "status": "reachable",
            "base_url": "http://localhost:11434",
        }
    )
    expect(page.locator('[data-provider-check-result="ollama"]')).to_have_text(
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
