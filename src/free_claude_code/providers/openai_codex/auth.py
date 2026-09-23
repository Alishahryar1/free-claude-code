"""FCC-owned ChatGPT credential lifecycle."""

import asyncio
import base64
import json
import os
import time
import uuid
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.config.paths import openai_auth_lock_path, openai_auth_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock

from .login import (
    OPENAI_CODEX_ORIGINATOR,
    OPENAI_OAUTH_CLIENT_ID,
    OPENAI_REVOKE_URL,
    OPENAI_TOKEN_URL,
    AuthorizationGrant,
    BrowserAuthorization,
    OpenAILoginError,
    request_device_authorization,
    wait_for_device_grant,
)

REFRESH_EARLY_SECONDS = 5 * 60
_UNAUTHORIZED_REFRESH_TOTAL_ATTEMPTS = 2
_UNAUTHORIZED_REFRESH_RETRY_DELAY_SECONDS = 1.0


class OpenAIReconnectRequired(RuntimeError):
    """The saved ChatGPT session is absent or no longer renewable."""


@dataclass(frozen=True, slots=True, repr=False)
class OpenAIAccess:
    """Current upstream authorization headers."""

    access_token: str
    account_id: str
    fedramp: bool


@dataclass(frozen=True, slots=True, repr=False)
class _Credentials:
    access_token: str
    refresh_token: str
    id_token: str
    account_id: str
    email: str | None
    expires_at: int
    fedramp: bool

    @classmethod
    def from_tokens(
        cls,
        *,
        access_token: str,
        refresh_token: str,
        id_token: str,
    ) -> _Credentials:
        access_claims = _optional_jwt_claims(access_token)
        id_claims = _jwt_claims(id_token)
        auth_claims = _mapping_claim(
            access_claims, "https://api.openai.com/auth"
        ) or _mapping_claim(id_claims, "https://api.openai.com/auth")
        account_id = _string_claim(auth_claims, "chatgpt_account_id")
        if not account_id:
            raise OpenAILoginError(
                "OpenAI sign-in did not return a ChatGPT account identifier."
            )
        profile = _mapping_claim(id_claims, "https://api.openai.com/profile")
        email = _string_claim(id_claims, "email") or _string_claim(profile, "email")
        expires_at = _integer_claim(access_claims, "exp")
        if expires_at is None:
            expires_at = int(time.time()) + 8 * 24 * 60 * 60
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=account_id,
            email=email,
            expires_at=expires_at,
            fedramp=auth_claims.get("chatgpt_account_is_fedramp") is True,
        )

    @classmethod
    def from_json(cls, payload: Any) -> _Credentials:
        if not isinstance(payload, dict):
            raise ValueError("credential document must be an object")
        required = ("access_token", "refresh_token", "id_token", "account_id")
        values = {key: payload.get(key) for key in required}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("credential document is missing a token field")
        expires_at = payload.get("expires_at")
        if not isinstance(expires_at, int) or isinstance(expires_at, bool):
            raise ValueError("credential document is missing expires_at")
        id_claims = _jwt_claims(values["id_token"])
        profile = _mapping_claim(id_claims, "https://api.openai.com/profile")
        email = _string_claim(id_claims, "email") or _string_claim(profile, "email")
        return cls(
            access_token=values["access_token"],
            refresh_token=values["refresh_token"],
            id_token=values["id_token"],
            account_id=values["account_id"],
            email=email if isinstance(email, str) else None,
            expires_at=expires_at,
            fedramp=payload.get("fedramp") is True,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "account_id": self.account_id,
            "expires_at": self.expires_at,
            "fedramp": self.fedramp,
        }


class OpenAIAuthManager:
    """Own credentials, interactive login, refresh, and revocation."""

    provider_id = "openai"

    def __init__(
        self,
        *,
        proxy: str | None = None,
        credential_path: Path | None = None,
        lock_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._credential_path = credential_path or openai_auth_path()
        self._lock_path = lock_path or openai_auth_lock_path()
        self._client = client or httpx.AsyncClient(
            proxy=proxy,
            timeout=httpx.Timeout(30.0),
            headers={"originator": OPENAI_CODEX_ORIGINATOR},
        )
        self._owns_client = client is None
        self._credentials: _Credentials | None = None
        self._revision = 0
        self._operation_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._login_task: asyncio.Task[None] | None = None
        self._browser_login: BrowserAuthorization | None = None
        self._attempt_id: str | None = None
        self._mode: ConnectedAccountLoginMode | None = None
        self._authorization_url: str | None = None
        self._verification_url: str | None = None
        self._user_code: str | None = None
        self._expires_at: int | None = None
        self._last_error: str | None = None
        self._closed = False
        try:
            self._credentials = self._read_credentials()
        except ValueError:
            self._last_error = (
                "Saved OpenAI credentials are invalid. Disconnect and sign in again."
            )
        if self._credentials is not None:
            self._revision = 1

    def is_connected(self) -> bool:
        """Return whether renewable credentials are present."""

        return self._credentials is not None

    def connected_provider_ids(self) -> tuple[str, ...]:
        """Return the provider availability contributed by this manager."""

        return (self.provider_id,) if self.is_connected() else ()

    def status(self) -> ConnectedAccountStatus:
        """Return a credential-free snapshot."""

        credentials = self._credentials
        connecting = self._login_task is not None and not self._login_task.done()
        if connecting:
            state = ConnectedAccountState.CONNECTING
        elif self._last_error:
            state = ConnectedAccountState.ERROR
        elif credentials is not None:
            state = ConnectedAccountState.CONNECTED
        else:
            state = ConnectedAccountState.DISCONNECTED
        return ConnectedAccountStatus(
            provider_id=self.provider_id,
            state=state,
            connected=credentials is not None,
            revision=self._revision,
            attempt_id=self._attempt_id if connecting else None,
            email=credentials.email if credentials is not None else None,
            mode=self._mode if connecting else None,
            authorization_url=self._authorization_url if connecting else None,
            verification_url=self._verification_url if connecting else None,
            user_code=self._user_code if connecting else None,
            expires_at=self._expires_at if connecting else None,
            message=self._last_error,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        """Start browser PKCE or device authorization without exposing secrets."""

        async with self._operation_lock:
            async with self._state_lock:
                self._ensure_open()
                if self._login_task is not None and not self._login_task.done():
                    return self.status()
                self._clear_attempt()
                self._last_error = None
                self._attempt_id = f"login_{uuid.uuid4().hex}"
                self._mode = mode
            browser: BrowserAuthorization | None = None
            device = None
            try:
                if mode is ConnectedAccountLoginMode.BROWSER:
                    browser = await BrowserAuthorization.start()
                else:
                    device = await request_device_authorization(self._client)
                async with self._state_lock:
                    if browser is not None:
                        self._browser_login = browser
                        self._authorization_url = browser.auth_url
                        self._expires_at = int(time.time()) + 15 * 60
                        self._login_task = asyncio.create_task(
                            self._complete_browser_login(browser),
                            name="openai-browser-login",
                        )
                    else:
                        assert device is not None
                        self._verification_url = device.verification_url
                        self._user_code = device.user_code
                        self._expires_at = device.expires_at
                        self._login_task = asyncio.create_task(
                            self._complete_login(
                                wait_for_device_grant(self._client, device)
                            ),
                            name="openai-device-login",
                        )
                    return self.status()
            except asyncio.CancelledError:
                if browser is not None:
                    with suppress(Exception):
                        await browser.close()
                async with self._state_lock:
                    self._clear_attempt()
                raise
            except Exception:
                if browser is not None:
                    with suppress(Exception):
                        await browser.close()
                async with self._state_lock:
                    self._clear_attempt()
                    self._last_error = "OpenAI sign-in could not start."
                raise

    async def cancel_login(self) -> ConnectedAccountStatus:
        """Cancel a pending login while preserving current credentials."""

        async with self._operation_lock:
            async with self._state_lock:
                login = self._detach_login_locked()
                self._last_error = None
            await self._close_detached_login(*login)
            return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        """Best-effort revoke, then always remove FCC-owned credentials."""

        async with self._operation_lock:
            async with self._state_lock:
                self._ensure_open()
                login = self._detach_login_locked()
            await self._close_detached_login(*login)
            async with self._state_lock:
                credentials = self._credentials
                if credentials is not None:
                    with suppress(httpx.HTTPError):
                        await self._client.post(
                            OPENAI_REVOKE_URL,
                            json={
                                "client_id": OPENAI_OAUTH_CLIENT_ID,
                                "token": credentials.refresh_token,
                                "token_type_hint": "refresh_token",
                            },
                        )
                await self._delete_credentials()
                self._credentials = None
                self._revision += 1
                self._last_error = None
                return self.status()

    async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
        """Return current request headers, refreshing once before expiry."""

        async with self._state_lock:
            self._ensure_open()
            credentials = self._credentials
            if credentials is None:
                raise OpenAIReconnectRequired(
                    "Connect an OpenAI ChatGPT account in the FCC Admin UI."
                )
            if force_refresh:
                credentials = await self._refresh_locked(credentials)
            elif credentials.expires_at <= int(time.time()) + REFRESH_EARLY_SECONDS:
                try:
                    credentials = await self._refresh_locked(credentials)
                except httpx.HTTPError as exc:
                    if not _is_transient_refresh_error(exc):
                        raise
            return OpenAIAccess(
                access_token=credentials.access_token,
                account_id=credentials.account_id,
                fedramp=credentials.fedramp,
            )

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        """Reload cross-process state, then recover one rejected access token."""

        async with self._state_lock:
            current = self._credentials
            reloaded = await asyncio.to_thread(self._read_credentials)
            if reloaded is not None and (
                current is None or reloaded.access_token != current.access_token
            ):
                self._credentials = reloaded
                current = reloaded
            if current is None:
                raise OpenAIReconnectRequired(
                    "OpenAI credentials are no longer available. Reconnect in Admin."
                )
            if current.access_token != rejected_token:
                return OpenAIAccess(
                    current.access_token, current.account_id, current.fedramp
                )
            refreshed = await self._recover_refresh_locked(current)
            return OpenAIAccess(
                refreshed.access_token, refreshed.account_id, refreshed.fedramp
            )

    async def _recover_refresh_locked(self, current: _Credentials) -> _Credentials:
        """Retry one transient reactive refresh without repeating a provider call."""
        for attempt in range(_UNAUTHORIZED_REFRESH_TOTAL_ATTEMPTS):
            try:
                return await self._refresh_locked(current)
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as error:
                if (
                    not _is_transient_refresh_error(error)
                    or attempt + 1 == _UNAUTHORIZED_REFRESH_TOTAL_ATTEMPTS
                ):
                    raise
                await asyncio.sleep(_UNAUTHORIZED_REFRESH_RETRY_DELAY_SECONDS)
        raise RuntimeError("OpenAI refresh recovery ended without an outcome")

    async def close(self) -> None:
        """Cancel login resources and close the owned HTTP client."""

        async with self._operation_lock:
            async with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                login = self._detach_login_locked()
            await self._close_detached_login(*login)
            if self._owns_client:
                await self._client.aclose()

    async def _complete_browser_login(self, browser: BrowserAuthorization) -> None:
        try:
            await self._complete_login(browser.wait())
        finally:
            await browser.close()

    async def _complete_login(
        self, grant_awaitable: Awaitable[AuthorizationGrant]
    ) -> None:
        try:
            grant = await grant_awaitable
            credentials = await self._exchange_code(grant)
            await self._write_credentials(credentials)
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._state_lock:
                self._last_error = (
                    "OpenAI sign-in failed. Retry or use the device-code option."
                )
        else:
            async with self._state_lock:
                self._credentials = credentials
                self._revision += 1
                self._last_error = None
        finally:
            async with self._state_lock:
                if self._login_task is asyncio.current_task():
                    self._login_task = None
                    self._browser_login = None
                    self._clear_attempt()

    async def _exchange_code(self, grant: AuthorizationGrant) -> _Credentials:
        response = await self._client.post(
            OPENAI_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": OPENAI_OAUTH_CLIENT_ID,
                "code": grant.code,
                "redirect_uri": grant.redirect_uri,
                "code_verifier": grant.code_verifier,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return _credentials_from_token_response(payload)

    async def _refresh_locked(self, current: _Credentials) -> _Credentials:
        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=30.0)
        if not acquired:
            raise OpenAIReconnectRequired(
                "Timed out waiting for another FCC process to refresh OpenAI."
            )
        try:
            reloaded = await asyncio.to_thread(self._read_credentials)
            if reloaded is not None and (
                reloaded != current
                and reloaded.expires_at > int(time.time()) + REFRESH_EARLY_SECONDS
            ):
                self._credentials = reloaded
                return reloaded
            response = await self._client.post(
                OPENAI_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "client_id": OPENAI_OAUTH_CLIENT_ID,
                    "refresh_token": current.refresh_token,
                },
            )
            if response.status_code in {400, 401, 403}:
                await asyncio.to_thread(self._delete_credentials_unlocked)
                self._credentials = None
                self._revision += 1
                self._last_error = (
                    "OpenAI sign-in expired. Reconnect the account in Admin."
                )
                raise OpenAIReconnectRequired(self._last_error)
            response.raise_for_status()
            payload = response.json()
            refreshed = _credentials_from_token_response(payload, previous=current)
            await asyncio.to_thread(self._write_credentials_unlocked, refreshed)
            self._credentials = refreshed
            self._revision += 1
            return refreshed
        finally:
            await asyncio.to_thread(file_lock.release)

    async def _write_credentials(self, credentials: _Credentials) -> None:
        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=30.0)
        if not acquired:
            raise OpenAILoginError("Could not lock the OpenAI credential file.")
        try:
            await asyncio.to_thread(self._write_credentials_unlocked, credentials)
        finally:
            await asyncio.to_thread(file_lock.release)

    def _write_credentials_unlocked(self, credentials: _Credentials) -> None:
        path = self._credential_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(path.parent, 0o700)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {"version": 1, "credentials": credentials.as_json()},
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    async def _delete_credentials(self) -> None:
        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=30.0)
        if not acquired:
            raise OpenAIReconnectRequired("Could not lock the OpenAI credential file.")
        try:
            await asyncio.to_thread(self._delete_credentials_unlocked)
        finally:
            await asyncio.to_thread(file_lock.release)

    def _delete_credentials_unlocked(self) -> None:
        self._credential_path.unlink(missing_ok=True)

    def _read_credentials(self) -> _Credentials | None:
        if not self._credential_path.is_file():
            return None
        try:
            payload = json.loads(self._credential_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                raise ValueError("credential document has an unsupported schema")
            return _Credentials.from_json(payload.get("credentials"))
        except (OSError, json.JSONDecodeError, OpenAILoginError, ValueError) as exc:
            raise ValueError("OpenAI credential file is invalid") from exc

    def _detach_login_locked(
        self,
    ) -> tuple[asyncio.Task[None] | None, BrowserAuthorization | None]:
        task = self._login_task
        browser = self._browser_login
        self._login_task = None
        self._browser_login = None
        self._clear_attempt()
        return task, browser

    @staticmethod
    async def _close_detached_login(
        task: asyncio.Task[None] | None,
        browser: BrowserAuthorization | None,
    ) -> None:
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if browser is not None:
            await browser.close()

    def _clear_attempt(self) -> None:
        self._attempt_id = None
        self._mode = None
        self._authorization_url = None
        self._verification_url = None
        self._user_code = None
        self._expires_at = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise OpenAIReconnectRequired("OpenAI authentication is shutting down.")


def _credentials_from_token_response(
    payload: Any,
    *,
    previous: _Credentials | None = None,
) -> _Credentials:
    if not isinstance(payload, dict):
        raise OpenAILoginError("OpenAI returned an invalid token response.")
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    id_token = payload.get("id_token")
    if previous is not None:
        access_token = access_token or previous.access_token
        refresh_token = refresh_token or previous.refresh_token
        id_token = id_token or previous.id_token
    if not all(
        isinstance(value, str) and value
        for value in (access_token, refresh_token, id_token)
    ):
        raise OpenAILoginError("OpenAI token response is missing credentials.")
    return _Credentials.from_tokens(
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
    )


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        encoded = token.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded))
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise OpenAILoginError("OpenAI returned an invalid identity token.") from exc
    if not isinstance(payload, dict):
        raise OpenAILoginError("OpenAI returned invalid identity claims.")
    return payload


def _optional_jwt_claims(token: str) -> dict[str, Any]:
    try:
        return _jwt_claims(token)
    except OpenAILoginError:
        return {}


def _mapping_claim(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _string_claim(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _integer_claim(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) else None


def _is_transient_refresh_error(error: httpx.HTTPError) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)
