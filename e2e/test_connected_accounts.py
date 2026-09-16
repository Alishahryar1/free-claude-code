"""Rendered connected-account flows use provider-owned capabilities and identity."""

import pytest
from playwright.sync_api import Dialog, Page, Route, expect

from e2e.provider_support import open_provider
from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.core.json_types import JsonObject


def _status(provider_id: str, *, connected: bool = False) -> JsonObject:
    return ConnectedAccountStatus(
        provider_id=provider_id,
        state=(
            ConnectedAccountState.CONNECTED
            if connected
            else ConnectedAccountState.DISCONNECTED
        ),
        connected=connected,
        revision=1,
        display_identity="octocat"
        if connected and provider_id == "github_copilot"
        else None,
        email="person@example.com" if connected and provider_id == "openai" else None,
        model_count=14 if connected else 0,
        supported_login_modes=(
            (ConnectedAccountLoginMode.DEVICE,)
            if provider_id == "github_copilot"
            else (ConnectedAccountLoginMode.BROWSER, ConnectedAccountLoginMode.DEVICE)
        ),
        default_login_mode=(
            ConnectedAccountLoginMode.DEVICE
            if provider_id == "github_copilot"
            else ConnectedAccountLoginMode.BROWSER
        ),
    ).as_dict()


class _Accounts:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.statuses = {
            provider_id: _status(provider_id)
            for provider_id in ("openai", "github_copilot")
        }
        self.login_requests: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.disconnected: list[str] = []
        self.hold_status: set[str] = set()
        self.pending_status: list[Route] = []
        self.hold_login = False
        self.pending_login: list[Route] = []

    def config(self, route: Route) -> None:
        response = route.fetch()
        config = response.json()
        assert isinstance(config, dict)
        providers = config["provider_status"]
        assert isinstance(providers, list)
        config["provider_status"] = [
            provider
            for provider in providers
            if isinstance(provider, dict)
            and provider.get("provider_id") not in self.statuses
        ] + [
            {
                "provider_id": provider_id,
                "display_name": name,
                "kind": "connected_account",
                "status": "disconnected",
                "label": "Not connected",
                "settings_keys": ["OPENAI_PROXY"] if provider_id == "openai" else [],
            }
            for provider_id, name in (
                ("openai", "OpenAI / ChatGPT"),
                ("github_copilot", "GitHub Copilot"),
            )
        ]
        route.fulfill(response=response, json=config)

    def auth(self, route: Route) -> None:
        request = route.request
        provider_id = request.url.split("/providers/", 1)[1].split("/", 1)[0]
        if request.method == "GET":
            if provider_id in self.hold_status:
                self.pending_status.append(route)
                return
        elif request.url.endswith("/login"):
            payload = request.post_data_json
            assert isinstance(payload, dict)
            mode = payload["mode"]
            assert isinstance(mode, str)
            self.login_requests.append((provider_id, mode))
            status = _status(
                provider_id, connected=bool(self.statuses[provider_id]["connected"])
            )
            status.update(state="connecting", mode=mode, attempt_id="safe-test-attempt")
            if mode == "browser":
                status["authorization_url"] = f"{self.base_url}/account-test-sign-in"
            else:
                status["verification_url"] = "https://github.com/login/device"
                status["user_code"] = "ABCD-1234"
            self.statuses[provider_id] = status
            if self.hold_login:
                self.pending_login.append(route)
                return
        elif request.url.endswith("/cancel"):
            self.cancelled.append(provider_id)
            self.statuses[provider_id] = _status(provider_id)
        elif request.method == "DELETE":
            self.disconnected.append(provider_id)
            self.statuses[provider_id] = _status(provider_id)
        else:
            raise AssertionError(f"Unexpected account operation: {request.method}")
        route.fulfill(json=self.statuses[provider_id])


@pytest.fixture
def accounts(page: Page, admin_base_url: str) -> _Accounts:
    result = _Accounts(admin_base_url)
    page.route("**/admin/api/config", result.config)
    page.route("**/admin/api/providers/*/auth**", result.auth)
    page.context.route(
        "**/account-test-sign-in",
        lambda route: route.fulfill(content_type="text/html", body="Test sign-in"),
    )
    page.add_init_script(
        """Object.defineProperty(navigator, 'clipboard', {
          value: { writeText: async (text) => { window.copiedDeviceCode = text; } }
        });"""
    )
    return result


def _open(page: Page, admin_base_url: str) -> None:
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")


def test_account_modes_wait_for_status_and_recover_after_load_failure(
    page: Page, admin_base_url: str, accounts: _Accounts
) -> None:
    accounts.hold_status.add("github_copilot")
    page.goto(f"{admin_base_url}/admin")
    copilot = page.locator('[data-provider="github_copilot"]')
    openai = page.locator('[data-provider="openai"]')
    expect(copilot.get_by_role("button", name="Loading…", exact=True)).to_be_disabled()
    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_have_count(0)
    expect(openai.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    assert len(accounts.pending_status) == 1

    accounts.pending_status.pop().fulfill(
        status=503, json={"detail": "Account status unavailable."}
    )
    expect(copilot.locator(".provider-meta")).to_have_text(
        "Account status unavailable."
    )
    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_have_count(0)
    copilot.get_by_role("button", name="Retry", exact=True).click()
    expect(copilot.get_by_role("button", name="Loading…", exact=True)).to_be_disabled()
    assert len(accounts.pending_status) == 1
    accounts.hold_status.clear()
    accounts.pending_status.pop().fulfill(json=accounts.statuses["github_copilot"])

    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    expect(copilot.get_by_role("button", name="Use device code")).to_have_count(0)
    expect(openai.get_by_role("button", name="Use device code")).to_be_enabled()
    expect(copilot.locator(".provider-meta")).to_contain_text("GitHub Copilot")
    assert "ChatGPT" not in copilot.inner_text()


def test_device_code_connect_copy_and_cancel_ignore_an_old_poll(
    page: Page, admin_base_url: str, accounts: _Accounts
) -> None:
    _open(page, admin_base_url)
    copilot = page.locator('[data-provider="github_copilot"]')
    copilot.get_by_role("button", name="Connect", exact=True).click()
    expect(copilot.locator(".provider-meta")).to_have_text(
        "Enter code ABCD-1234 at https://github.com/login/device"
    )
    assert accounts.login_requests == [("github_copilot", "device")]
    assert len(page.context.pages) == 1
    expect(copilot.get_by_role("button", name="Open sign-in")).to_be_visible()
    copilot.get_by_role("button", name="Copy code").click()
    assert page.evaluate("window.copiedDeviceCode") == "ABCD-1234"

    stale_status = dict(accounts.statuses["github_copilot"])
    accounts.hold_status.add("github_copilot")
    with page.expect_request("**/admin/api/providers/github_copilot/auth"):
        pass
    copilot.get_by_role("button", name="Cancel sign-in", exact=True).click()
    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    assert accounts.cancelled == ["github_copilot"]
    assert len(accounts.pending_status) == 1
    accounts.hold_status.clear()
    with page.expect_response("**/admin/api/providers/github_copilot/auth") as response:
        accounts.pending_status.pop().fulfill(json=stale_status)
    response.value.finished()
    page.evaluate("() => new Promise(requestAnimationFrame)")
    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    expect(copilot.get_by_role("button", name="Copy code")).to_have_count(0)


def test_openai_browser_login_preopens_popup_and_retains_device_choice(
    page: Page, admin_base_url: str, accounts: _Accounts
) -> None:
    accounts.hold_login = True
    _open(page, admin_base_url)
    openai = page.locator('[data-provider="openai"]')
    with page.expect_popup() as opened:
        openai.get_by_role("button", name="Connect", exact=True).click()
    popup = opened.value
    try:
        assert popup.url == "about:blank"
        assert popup.evaluate("window.opener === null") is True
        assert accounts.login_requests == [("openai", "browser")]
        assert len(accounts.pending_login) == 1
        accounts.pending_login.pop().fulfill(json=accounts.statuses["openai"])
        popup.wait_for_url(f"{admin_base_url}/account-test-sign-in")
    finally:
        popup.close()
    expect(
        openai.get_by_role("button", name="Cancel sign-in", exact=True)
    ).to_be_visible()
    expect(openai.locator(".provider-meta")).to_have_text(
        "Finish signing in, then return to this page."
    )
    openai.get_by_role("button", name="Cancel sign-in", exact=True).click()
    accounts.hold_login = False
    openai.get_by_role("button", name="Use device code", exact=True).click()
    expect(openai.get_by_role("button", name="Copy code")).to_be_visible()
    assert accounts.login_requests[-1] == ("openai", "device")
    assert len(page.context.pages) == 1
    openai.get_by_role("button", name="Cancel sign-in", exact=True).click()


@pytest.mark.parametrize("restart", [False, True])
def test_connected_counts_and_modes_survive_apply_and_disconnect_independently(
    page: Page, admin_base_url: str, accounts: _Accounts, restart: bool
) -> None:
    accounts.statuses = {
        provider_id: _status(provider_id, connected=True)
        for provider_id in accounts.statuses
    }
    page.route(
        "**/admin/api/config/apply",
        lambda route: route.fulfill(
            json={
                "applied": True,
                "restart": {
                    "required": restart,
                    "automatic": restart,
                    "admin_url": "/admin",
                    "instance_id": "before-restart",
                },
                "credential_checks": [],
            }
        ),
    )
    page.route(
        "**/admin/api/status",
        lambda route: route.fulfill(
            json={"status": "running", "instance_id": "after-restart"}
        ),
    )
    _open(page, admin_base_url)
    open_provider(page, "nvidia_nim")
    page.locator("#field-NVIDIA_NIM_API_KEY").fill("unused-test-key")
    page.get_by_role("button", name="Save", exact=True).click()
    expect(page.locator("#providerDialog")).not_to_be_visible()
    expect(page.locator("#messageArea")).to_have_text("Applied")
    expect(page.locator("#dirtyState")).to_have_text("No changes")
    copilot = page.locator('[data-provider="github_copilot"]')
    openai = page.locator('[data-provider="openai"]')
    expect(copilot.locator(".provider-meta")).to_have_text("14 models available")
    expect(openai.locator(".provider-meta")).to_have_text("14 models available")
    assert "ChatGPT" not in copilot.inner_text()
    expect(page.get_by_role("button", name="Reconnect", exact=True)).to_have_count(0)
    connected = page.locator(
        '[data-provider-group="oauth"] [data-provider-subgroup="configured"]'
    )
    expect(connected.locator(".provider-title strong")).to_have_text(
        ["GitHub Copilot", "OpenAI / ChatGPT"]
    )
    confirmations: list[str] = []

    def accept_disconnect(dialog: Dialog) -> None:
        confirmations.append(dialog.message)
        dialog.accept()

    page.once("dialog", accept_disconnect)
    copilot.get_by_role("button", name="Disconnect", exact=True).click()
    expect(copilot.get_by_role("button", name="Connect", exact=True)).to_be_enabled()
    assert confirmations == ["Disconnect this GitHub Copilot account from FCC?"]
    assert accounts.disconnected == ["github_copilot"]
    expect(connected.locator(".provider-title strong")).to_have_text(
        ["OpenAI / ChatGPT"]
    )
    expect(
        page.locator(
            '[data-provider-group="oauth"] [data-provider-subgroup="unconfigured"] .provider-title strong'
        )
    ).to_have_text(["GitHub Copilot"])
    expect(openai.get_by_role("button", name="Disconnect", exact=True)).to_have_class(
        "danger-button"
    )
    expect(openai.locator(".provider-meta")).to_have_text("14 models available")
    copilot.get_by_role("button", name="Connect", exact=True).click()
    expect(copilot.get_by_role("button", name="Copy code")).to_be_visible()
    assert accounts.login_requests[-1] == ("github_copilot", "device")
    copilot.get_by_role("button", name="Cancel sign-in", exact=True).click()


def test_auth_status_refresh_preserves_proxy_edit_and_selection(
    page: Page, admin_base_url: str, accounts: _Accounts
) -> None:
    accounts.hold_status.add("openai")
    page.goto(f"{admin_base_url}/admin")
    dialog = open_provider(page, "openai")
    proxy = dialog.locator("#field-OPENAI_PROXY")
    proxy.fill("http://pending-proxy:8080")
    proxy.evaluate("input => input.setSelectionRange(7, 14)")
    accounts.hold_status.clear()
    accounts.pending_status.pop().fulfill(json=_status("openai", connected=True))
    expect(
        page.locator('[data-provider="openai"]').get_by_role(
            "button", name="Disconnect", exact=True
        )
    ).to_have_count(1)
    expect(proxy).to_have_value("http://pending-proxy:8080")
    expect(proxy).to_be_focused()
    assert proxy.evaluate("input => [input.selectionStart, input.selectionEnd]") == [
        7,
        14,
    ]
    expect(dialog.get_by_role("button", name="Connect", exact=True)).to_have_count(0)


def test_slow_initial_account_check_does_not_erase_a_later_save_warning(
    page: Page, admin_base_url: str, accounts: _Accounts
) -> None:
    accounts.hold_status.add("github_copilot")
    page.route(
        "**/admin/api/config/apply",
        lambda route: route.fulfill(
            json={
                "applied": True,
                "credential_checks": [
                    {
                        "key": "NVIDIA_NIM_API_KEY",
                        "status": "unverified",
                        "message": "Verification unavailable.",
                    }
                ],
            }
        ),
    )
    page.goto(f"{admin_base_url}/admin")
    dialog = open_provider(page, "nvidia_nim")
    assert len(accounts.pending_status) == 1
    accounts.hold_status.clear()
    dialog.locator("#field-NVIDIA_NIM_API_KEY").fill("new-key")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
    with page.expect_response("**/admin/api/providers/github_copilot/auth") as response:
        accounts.pending_status.pop().fulfill(json=_status("github_copilot"))
    response.value.finished()
    page.evaluate("() => new Promise(requestAnimationFrame)")
    expect(page.locator("#messageArea")).to_contain_text("Verification unavailable.")
