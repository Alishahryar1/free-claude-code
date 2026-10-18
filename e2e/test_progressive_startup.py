import pytest
from playwright.sync_api import expect

from e2e.provider_support import api_key_input, close_provider, open_provider
from e2e.test_code_sessions import create_session


@pytest.mark.parametrize(
    "admin_base_url", [{"NVIDIA_NIM_API_KEY": "e2e-nim-key"}], indirect=True
)
def test_discovery_error_is_visible_in_provider_card_and_refresh_result(
    page, admin_base_url
):
    proxy_message = "The network proxy rejected or failed the connection. Check your proxy settings."
    http_message = "The model-list request returned HTTP 403."

    def status(route):
        data = route.fetch().json()
        data["startup"]["providers"]["nvidia_nim"] = "failed"
        data["startup"]["provider_errors"] = {"nvidia_nim": proxy_message}
        route.fulfill(json=data)

    page.route("**/admin/api/status", status)
    page.route(
        "**/admin/api/providers/nvidia_nim/test",
        lambda route: route.fulfill(
            json={"provider_id": "nvidia_nim", "ok": False, "message": http_message}
        ),
    )
    page.goto(f"{admin_base_url}/admin")
    card = page.locator('[data-provider-check-result="nvidia_nim"]')
    expect(card).to_have_text(proxy_message)
    dialog = open_provider(page, "nvidia_nim")
    expect(dialog.locator("#providerDialogCheck")).to_have_text(proxy_message)
    expect(
        dialog.get_by_text(
            "Optional network proxy. Leave blank unless you use one", exact=True
        )
    ).to_be_visible()
    dialog.get_by_role("button", name="Refresh models", exact=True).click()
    expect(dialog.locator("#providerDialogCheck")).to_contain_text(http_message)
    expect(card).to_contain_text(http_message)


def test_provider_and_messaging_progress_preserve_settings(page, admin_base_url):
    starting = True
    provider_tests = []
    page.on(
        "request",
        lambda request: (
            provider_tests.append(request.url)
            if request.url.endswith("/test")
            else None
        ),
    )

    def status(route):
        data = route.fetch().json()
        data["startup"]["providers"]["open_router"] = (
            "starting" if starting else "ready"
        )
        data["startup"]["messaging"] = {"state": "starting" if starting else "ready"}
        route.fulfill(json=data)

    page.route("**/admin/api/status", status)
    page.goto(f"{admin_base_url}/admin")
    dialog = open_provider(page, "open_router")
    test = dialog.get_by_role("button", name="Refresh models", exact=True)
    expect(test).to_be_disabled()
    expect(test).to_have_attribute("aria-busy", "true")
    expect(page.locator('[data-provider-check-result="open_router"]')).to_have_text(
        "Checking models…"
    )
    expect(dialog.locator("#providerDialogCheck")).to_have_text("Checking models…")
    close_provider(page)
    page.get_by_role("button", name="Messaging", exact=True).click()
    expect(page.locator("#startupMessage")).to_be_visible()
    expect(page.locator("#startupMessage")).to_have_class(
        "startup-message startup-spinner"
    )
    page.get_by_role("button", name="Providers", exact=True).click()
    open_provider(page, "nvidia_nim")
    key = api_key_input(page, "NVIDIA_NIM_API_KEY")
    key.fill("unsaved-key")
    starting = False
    page.wait_for_function("state.startup?.startup?.providers?.open_router === 'ready'")
    expect(key).to_have_value("unsaved-key")
    close_provider(page)
    dialog = open_provider(page, "open_router")
    expect(
        dialog.get_by_role("button", name="Refresh models", exact=True)
    ).to_be_enabled()
    expect(page.locator('[data-provider-check-result="open_router"]')).to_have_text(
        "3 models available"
    )
    expect(dialog.locator("#providerDialogCheck")).to_have_text("3 models available")
    close_provider(page)
    page.get_by_role("button", name="Messaging", exact=True).click()
    expect(page.locator("#startupMessage")).to_be_hidden()
    assert provider_tests == []


def test_codex_connect_waits_for_catalog_and_recovers_from_publication_error(
    page, admin_base_url
):
    phase = "starting"

    def status(route):
        data = route.fetch().json()
        data["startup"]["catalog_file"] = phase
        route.fulfill(json=data)

    page.route("**/admin/api/status", status)
    with (
        page.expect_response(
            "**/admin/api/integrations/claude-vscode"
        ) as claude_status,
        page.expect_response("**/admin/api/integrations/codex") as codex_status,
    ):
        page.goto(f"{admin_base_url}/admin/integrations")
    claude_status.value.finished()
    codex_status.value.finished()
    page.wait_for_function("state.startup?.startup?.catalog_file === 'starting'")
    connect = page.locator("#openCodexIntegration")
    expect(connect).to_be_disabled()
    expect(connect).to_have_attribute("aria-busy", "true")
    expect(page.locator("#openClaudeIntegration")).to_be_enabled()
    phase = "failed"
    message = page.locator("#codexIntegrationMessage")
    expect(message).to_contain_text("Could not prepare")
    phase = "ready"
    page.get_by_role("button", name="Providers", exact=True).click()
    page.get_by_role("button", name="Integrations", exact=True).click()
    expect(connect).to_be_enabled()
    expect(connect).to_have_text("Connect")
    expect(message).to_be_hidden()


def test_initial_code_catalog_wait_keeps_history_and_draft(
    page, admin_base_url, tmp_path
):
    starting = True

    def bootstrap(route):
        data = route.fetch().json()
        if starting:
            data["models"] = []
            data["startup"]["catalog"] = "starting"
            data["startup"]["providers"]["provider"] = "starting"
        route.fulfill(json=data)

    def status(route):
        data = route.fetch().json()
        if starting:
            data["startup"]["catalog"] = "starting"
        route.fulfill(json=data)

    page.route("**/admin/api/code/bootstrap", bootstrap)
    page.route("**/admin/api/status", status)
    create_session(page, admin_base_url, tmp_path)
    assert page.evaluate("window.CodeSessions.refresh()") is True
    composer = page.locator("#codeComposer")
    composer.fill("Keep my draft")
    send = page.locator("#codeSend")
    expect(send).to_be_disabled()
    expect(send).to_have_attribute("aria-busy", "true")
    expect(page.locator("#codeProvider")).to_be_disabled()
    expect(page.locator("#codeMode")).to_be_enabled()
    starting = False
    expect(send).to_be_enabled()
    expect(composer).to_have_value("Keep my draft")


def test_failed_catalog_refresh_preserves_synchronized_session(
    page, admin_base_url, tmp_path
):
    page.add_init_script("""{
      const Feed = window.EventSource;
      window.codeFeedCount = 0;
      window.EventSource = class extends Feed {
        constructor(...args) { super(...args); window.codeFeedCount++; }
      };
    }""")
    create_session(page, admin_base_url, tmp_path)
    composer = page.locator("#codeComposer")
    composer.fill("Keep the draft during a failed catalog read")
    expect(page.locator("#codeSend")).to_be_enabled()
    feeds = page.evaluate("window.codeFeedCount")
    page.route(
        "**/admin/api/code/bootstrap",
        lambda route: route.fulfill(status=503, json={"detail": "Catalog unavailable"}),
    )
    assert page.evaluate("window.CodeSessions.refresh()") is False
    expect(composer).to_have_value("Keep the draft during a failed catalog read")
    expect(page.locator("#codeSend")).to_be_enabled()
    assert page.evaluate("window.codeFeedCount") == feeds


def test_code_catalog_cannot_roll_back_after_a_newer_startup_status(
    page, admin_base_url, tmp_path
):
    create_session(page, admin_base_url, tmp_path)
    composer = page.locator("#codeComposer")
    composer.fill("Preserve my selected model")
    expect(page.locator("#codeSend")).to_be_enabled()
    page.wait_for_function("!state.startupRequest && !state.startupTimer")
    data = page.request.get(f"{admin_base_url}/admin/api/code/bootstrap").json()
    status = page.request.get(f"{admin_base_url}/admin/api/status").json()
    status["startup"].update(generation_id=10, catalog_revision=10)
    stale = data | {
        "models": [],
        "startup": data["startup"] | {"generation_id": 9, "catalog_revision": 9},
    }
    page.route("**/admin/api/code/bootstrap", lambda route: route.fulfill(json=stale))
    assert (
        page.evaluate("status => window.CodeSessions.refresh(status)", status) is False
    )
    expect(composer).to_have_value("Preserve my selected model")
    expect(page.locator("#codeSend")).to_be_enabled()
    data["startup"].update(generation_id=10, catalog_revision=10)
    page.unroute("**/admin/api/code/bootstrap")
    page.route("**/admin/api/code/bootstrap", lambda route: route.fulfill(json=data))
    assert (
        page.evaluate("status => window.CodeSessions.refresh(status)", status) is True
    )
    expect(page.locator("#codeSelectionError")).to_be_hidden()


def test_code_catalog_coalesces_notifications_and_catches_up_to_latest_version(
    page, admin_base_url, tmp_path
):
    create_session(page, admin_base_url, tmp_path)
    page.wait_for_function("!state.startupRequest && !state.startupTimer")
    assert page.evaluate("window.CodeSessions.refresh()") is True
    data = page.request.get(f"{admin_base_url}/admin/api/code/bootstrap").json()
    status = page.request.get(f"{admin_base_url}/admin/api/status").json()
    generation = status["startup"]["generation_id"] + 1
    status["startup"].update(generation_id=generation, catalog_revision=1)
    first = data | {"startup": data["startup"] | status["startup"]}
    latest = data | {
        "startup": first["startup"] | {"catalog_revision": 5},
        "models": [
            *data["models"],
            data["models"][0]
            | {"id": "extra/later", "provider_id": "extra", "model_name": "later"},
        ],
    }
    pending = []
    calls = 0

    def bootstrap(route):
        nonlocal calls
        calls += 1
        if calls == 1:
            pending.append(route)
            page.evaluate("window.heldCatalogVersion = true")
        else:
            route.fulfill(json=latest)

    page.route("**/admin/api/code/bootstrap", bootstrap)
    page.evaluate("status => { void window.CodeSessions.refresh(status); }", status)
    page.wait_for_function("window.heldCatalogVersion === true")
    for revision in range(2, 6):
        status["startup"]["catalog_revision"] = revision
        page.evaluate("status => { void window.CodeSessions.refresh(status); }", status)
    assert calls == 1
    pending.pop().fulfill(json=first)
    expect(page.locator('#codeProvider option[value="extra"]')).to_have_count(1)
    assert calls == 2
