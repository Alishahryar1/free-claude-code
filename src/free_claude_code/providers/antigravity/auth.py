"""FCC opt-in state over credentials owned by the native Antigravity CLI."""

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.paths import antigravity_auth_path
from free_claude_code.core.async_tasks import run_sync_owned

from .credentials import (
    AntigravityCredentialError,
    AntigravityCredentials,
    load_native_credentials,
)

CredentialLoader = Callable[[], AntigravityCredentials]


class AntigravityAuthManager:
    """Own FCC's opt-in while leaving Google credentials under ``agy`` ownership."""

    provider_id = "antigravity"

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        credential_loader: CredentialLoader = load_native_credentials,
    ) -> None:
        self._state_path = state_path or antigravity_auth_path()
        self._credential_loader = credential_loader
        self._enabled = False
        self._revision = 0
        self._last_error: str | None = None
        self._operation_lock = asyncio.Lock()
        self._closed = False
        try:
            self._read_state()
        except (OSError, ValueError, UnicodeError):
            self._enabled = False
            self._last_error = (
                "Saved Antigravity connection state is unreadable. Connect again."
            )

    def is_connected(self) -> bool:
        return self._enabled and not self._closed

    def connected_provider_ids(self) -> tuple[str, ...]:
        return (self.provider_id,) if self.is_connected() else ()

    def status(self) -> ConnectedAccountStatus:
        state = (
            ConnectedAccountState.ERROR
            if self._last_error
            else ConnectedAccountState.CONNECTED
            if self.is_connected()
            else ConnectedAccountState.DISCONNECTED
        )
        return ConnectedAccountStatus(
            provider_id=self.provider_id,
            state=state,
            connected=self.is_connected(),
            revision=self._revision,
            supported_login_modes=(ConnectedAccountLoginMode.DEVICE,),
            default_login_mode=ConnectedAccountLoginMode.DEVICE,
            message=self._last_error,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        if mode is not ConnectedAccountLoginMode.DEVICE:
            raise InvalidRequestError(
                "Antigravity uses the native `agy` account. Sign in with `agy` first."
            )
        async with self._operation_lock:
            self._ensure_open()
            self._last_error = None
            try:
                await run_sync_owned(self._credential_loader)
            except AntigravityCredentialError as error:
                self._enabled = False
                self._last_error = str(error)
                return self.status()
            revision = self._revision + 1
            try:
                self._write_state(enabled=True, revision=revision)
            except OSError:
                self._enabled = False
                self._last_error = (
                    "FCC could not save Antigravity connection state. "
                    "Check ~/.fcc permissions and retry Connect."
                )
                return self.status()
            self._enabled = True
            self._revision = revision
            return self.status()

    async def cancel_login(self) -> ConnectedAccountStatus:
        return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        async with self._operation_lock:
            self._ensure_open()
            revision = self._revision + 1
            try:
                self._write_state(enabled=False, revision=revision)
            except OSError:
                self._last_error = (
                    "FCC could not save Antigravity disconnected state. "
                    "Check ~/.fcc permissions and retry Disconnect."
                )
                return self.status()
            self._enabled = False
            self._revision = revision
            self._last_error = None
            return self.status()

    async def close(self) -> None:
        self._closed = True

    async def credentials(self) -> AntigravityCredentials:
        """Return a fresh native credential snapshot for one upstream attempt."""

        self._ensure_open()
        if not self._enabled:
            raise AntigravityCredentialError(
                "Connect the Antigravity account in FCC Admin first."
            )
        try:
            return await run_sync_owned(self._credential_loader)
        except AntigravityCredentialError as error:
            self._enabled = False
            self._last_error = str(error)
            try:
                self._write_state(enabled=False, revision=self._revision + 1)
            except OSError:
                pass
            else:
                self._revision += 1
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise AntigravityCredentialError("Antigravity account manager is closing.")

    def _read_state(self) -> None:
        try:
            payload: object = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Invalid Antigravity state schema")
        enabled = payload.get("enabled")
        revision = payload.get("revision")
        if (
            not isinstance(enabled, bool)
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
        ):
            raise ValueError("Invalid Antigravity state")
        self._enabled = enabled
        self._revision = revision

    def _write_state(self, *, enabled: bool, revision: int) -> None:
        payload = {
            "schema_version": 1,
            "enabled": enabled,
            "revision": revision,
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_path.parent,
                prefix=".antigravity-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
