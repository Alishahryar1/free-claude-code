"""Safe connected-account capabilities preserve provider identity semantics."""

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)


def test_existing_account_defaults_remain_browser_and_device() -> None:
    status = ConnectedAccountStatus(
        provider_id="openai",
        state=ConnectedAccountState.CONNECTED,
        connected=True,
        revision=1,
        email="person@example.com",
    )

    payload = status.as_dict()

    assert payload["email"] == "person@example.com"
    assert payload["supported_login_modes"] == ["browser", "device"]
    assert payload["default_login_mode"] == "browser"
    assert "display_identity" not in payload


def test_display_identity_does_not_impersonate_an_email_address() -> None:
    status = ConnectedAccountStatus(
        provider_id="github_copilot",
        state=ConnectedAccountState.CONNECTED,
        connected=True,
        revision=2,
        display_identity="octocat",
        supported_login_modes=(ConnectedAccountLoginMode.DEVICE,),
        default_login_mode=ConnectedAccountLoginMode.DEVICE,
    )

    assert status.as_dict() == {
        "provider_id": "github_copilot",
        "state": "connected",
        "connected": True,
        "revision": 2,
        "display_identity": "octocat",
        "supported_login_modes": ["device"],
        "default_login_mode": "device",
    }
