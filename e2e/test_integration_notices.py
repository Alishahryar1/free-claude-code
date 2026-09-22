import pytest
from playwright.sync_api import expect


@pytest.fixture(
    params=[
        ("claude-vscode", "openClaudeIntegration", "claudeIntegrationMessage"),
        ("codex", "openCodexIntegration", "codexIntegrationMessage"),
        ("jetbrains-acp", "openJetBrainsIntegration", "jetBrainsIntegrationMessage"),
        (
            "claude-desktop",
            "openClaudeDesktopIntegration",
            "claudeDesktopIntegrationMessage",
        ),
    ]
)
def notice(page, admin_base_url, request):
    integration, button_id, message_id = request.param
    progress = {"state": "ready", "changed": True, "message": None}
    server = {"instance_id": "first"}

    def startup(route):
        response = route.fetch()
        payload = response.json()
        payload["instance_id"] = server["instance_id"]
        payload["startup"]["integrations"][integration] = dict(progress)
        route.fulfill(response=response, json=payload)

    def status(route):
        route.fulfill(json={"connected": True, "paths": None, "update": dict(progress)})

    page.route("**/admin/api/status", startup)
    page.route(f"**/admin/api/integrations/{integration}", status)
    page.goto(f"{admin_base_url}/admin/integrations")
    button = page.locator(f"#{button_id}")
    message = page.locator(f"#{message_id}")
    expect(button).to_have_text("Disconnect")

    def observe(phase, changed=False, instance="first"):
        progress.update(
            state=phase,
            changed=changed,
            message="Update failed" if phase == "failed" else None,
        )
        server["instance_id"] = instance
        page.evaluate("() => refreshStartup()")
        page.wait_for_function(
            "([id, phase, changed, instance]) => state.startup?.instance_id === instance && state.startup.startup.integrations[id].state === phase && state.startup.startup.integrations[id].changed === changed",
            arg=[integration, phase, changed, instance],
        )

    return integration, button, message, observe


def test_notices_require_a_live_changed_completion(page, notice):
    _, button, message, observe = notice
    expect(message).to_be_hidden()
    observe("starting")
    expect(button).to_be_disabled()
    observe("ready", changed=True)
    expect(message).to_be_visible()
    expect(message).not_to_have_class("message-area error")
    expect(button).to_be_enabled()
    page.get_by_role("button", name="Providers", exact=True).click()
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(button).to_be_enabled()
    expect(message).to_be_hidden()
    observe("ready", changed=True)
    expect(message).to_be_hidden()
    page.reload()
    expect(button).to_be_enabled()
    expect(message).to_be_hidden()
    observe("starting")
    observe("ready", changed=False)
    expect(button).to_be_enabled()
    expect(message).to_be_hidden()
    observe("starting")
    observe("failed")
    expect(message).to_have_text("Update failed")
    observe("starting")
    observe("ready", changed=True)
    expect(message).to_be_visible()
    expect(message).not_to_have_text("Update failed")


@pytest.mark.parametrize("revisit_while_busy", [False, True])
def test_busy_status_recovery_preserves_notice_without_replaying_it(
    page, notice, revisit_while_busy
):
    integration, button, message, observe = notice
    observe("starting")
    pending = []

    def hold_status(route):
        pending.append(route)
        page.evaluate("window.integrationStatusIntercepted = true")

    page.route(f"**/admin/api/integrations/{integration}", hold_status)
    page.get_by_role("button", name="Providers", exact=True).click()
    with page.expect_request(f"**/admin/api/integrations/{integration}"):
        page.get_by_role("button", name="Integrations", exact=True).click()
    page.wait_for_function("window.integrationStatusIntercepted === true")
    observe("ready", changed=True)
    expect(message).to_be_visible()
    if revisit_while_busy:
        page.get_by_role("button", name="Providers", exact=True).click()
        page.get_by_role("button", name="Integrations", exact=True).click()
        expect(message).to_be_hidden()
    # Release a stale read after the completion poll. The subsequent recovery GET
    # must restore the button without erasing or replaying the completion notice.
    page.unroute(f"**/admin/api/integrations/{integration}")
    page.route(
        f"**/admin/api/integrations/{integration}",
        lambda route: route.fulfill(
            json={
                "connected": True,
                "paths": None,
                "update": {"state": "ready", "changed": True},
            }
        ),
    )
    assert len(pending) == 1
    pending[0].fulfill(
        json={
            "connected": None,
            "paths": None,
            "update": {"state": "starting", "changed": False},
        }
    )
    expect(button).to_be_enabled()
    if revisit_while_busy:
        expect(message).to_be_hidden()
    else:
        expect(message).to_be_visible()
    page.get_by_role("button", name="Providers", exact=True).click()
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(message).to_be_hidden()


def test_new_server_and_off_page_completions_stay_quiet(page, notice):
    _, button, message, observe = notice
    observe("starting")
    observe("ready", changed=True, instance="second")
    expect(button).to_be_enabled()
    expect(message).to_be_hidden()
    observe("starting", instance="second")
    page.get_by_role("button", name="Providers", exact=True).click()
    observe("ready", changed=True, instance="second")
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(button).to_be_enabled()
    expect(message).to_be_hidden()
