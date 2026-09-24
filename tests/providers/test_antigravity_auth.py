"""FCC opt-in behavior for a native Antigravity account."""

import json
from pathlib import Path

import pytest

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
)
from free_claude_code.providers.antigravity.auth import AntigravityAuthManager
from free_claude_code.providers.antigravity.credentials import (
    AntigravityCredentialError,
    AntigravityCredentials,
)

DEVICE = ConnectedAccountLoginMode.DEVICE


def credentials() -> AntigravityCredentials:
    return AntigravityCredentials(
        access_token="access",
        refresh_token="refresh",
        expires_at=2_000_000_000.0,
        source="test",
    )


def expired_credentials() -> AntigravityCredentials:
    return AntigravityCredentials(
        access_token="expired",
        refresh_token="refresh",
        expires_at=0.0,
        source="test",
    )


@pytest.mark.asyncio
async def test_connect_persists_only_safe_opt_in_state(tmp_path: Path) -> None:
    path = tmp_path / "antigravity.json"
    manager = AntigravityAuthManager(
        state_path=path,
        credential_loader=credentials,
    )

    status = await manager.start_login(DEVICE)

    assert status.state is ConnectedAccountState.CONNECTED
    assert manager.connected_provider_ids() == ("antigravity",)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"schema_version": 1, "enabled": True, "revision": 1}
    assert "access" not in path.read_text(encoding="utf-8")
    assert "refresh" not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_connect_rejects_expired_native_session(tmp_path: Path) -> None:
    manager = AntigravityAuthManager(
        state_path=tmp_path / "antigravity.json",
        credential_loader=expired_credentials,
    )

    status = await manager.start_login(DEVICE)

    assert status.state is ConnectedAccountState.ERROR
    assert not manager.is_connected()
    assert status.message is not None
    assert "expired or expiring" in status.message


@pytest.mark.asyncio
async def test_restart_keeps_opt_in_without_copying_credentials(tmp_path: Path) -> None:
    path = tmp_path / "antigravity.json"
    first = AntigravityAuthManager(state_path=path, credential_loader=credentials)
    await first.start_login(DEVICE)
    await first.close()

    restarted = AntigravityAuthManager(
        state_path=path,
        credential_loader=credentials,
    )

    assert restarted.is_connected()
    assert (await restarted.credentials()).access_token == "access"


@pytest.mark.asyncio
async def test_disconnect_only_disables_fcc_state(tmp_path: Path) -> None:
    path = tmp_path / "antigravity.json"
    calls = 0

    def loader() -> AntigravityCredentials:
        nonlocal calls
        calls += 1
        return credentials()

    manager = AntigravityAuthManager(state_path=path, credential_loader=loader)
    await manager.start_login(DEVICE)
    status = await manager.disconnect()

    assert status.state is ConnectedAccountState.DISCONNECTED
    assert calls == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "enabled": False,
        "revision": 2,
    }


@pytest.mark.asyncio
async def test_missing_native_session_surfaces_error_without_enabling(tmp_path: Path) -> None:
    def missing() -> AntigravityCredentials:
        raise AntigravityCredentialError("Sign in with `agy` first.")

    manager = AntigravityAuthManager(
        state_path=tmp_path / "antigravity.json",
        credential_loader=missing,
    )

    status = await manager.start_login(DEVICE)

    assert status.state is ConnectedAccountState.ERROR
    assert not manager.is_connected()
    assert status.message == "Sign in with `agy` first."


@pytest.mark.asyncio
async def test_lost_native_session_invalidates_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "antigravity.json"
    available = True

    def loader() -> AntigravityCredentials:
        if not available:
            raise AntigravityCredentialError("Native session disappeared.")
        return credentials()

    manager = AntigravityAuthManager(state_path=path, credential_loader=loader)
    await manager.start_login(DEVICE)
    available = False

    with pytest.raises(AntigravityCredentialError, match="disappeared"):
        await manager.credentials()

    assert not manager.is_connected()
    assert json.loads(path.read_text(encoding="utf-8"))["enabled"] is False
