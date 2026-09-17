"""FCC-owned CodeBuddy credential lifecycle."""

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.paths import codebuddy_auth_lock_path, codebuddy_auth_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock

from .login import (
    DEVICE_POLL_INTERVAL_SECONDS,
    CodeBuddyLoginError,
    DeviceAuthorization,
    fetch_account_profile,
    poll_device_tokens,
    start_device_authorization,
)
from .site import CodeBuddySite, default_site

REFRESH_EARLY_SECONDS = 5 * 60
_REFRESH_TIMEOUT_SECONDS = 20.0
_UNAUTHORIZED_REFRESH_TOTAL_ATTEMPTS = 2
_UNAUTHORIZED_REFRESH_RETRY_DELAY_SECONDS = 1.0


class CodeBuddyReconnectRequired(RuntimeError):
    """The saved CodeBuddy session is absent or no longer renewable."""


@dataclass(frozen=True, slots=True, repr=False)
class CodeBuddyAccess:
    """Current upstream authorization and account-header identity."""

    access_token: str
    uid: str | None
    enterprise_id: str | None
    domain: str | None


@dataclass(frozen=True, slots=True, repr=False)
class _Credentials:
    access_token: str
    refresh_token: str
    expires_at: int | None
    uid: str | None
    enterprise_id: str | None
    domain: str | None
    nickname: str | None

    @classmethod
    def from_json(cls, payload: Any) -> _Credentials:
        if not isinstance(payload, dict):
            raise ValueError("credential document must be an object")
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("credential document is missing a token field")
        refresh_token = payload.get("refresh_token")
        expires_at = payload.get("expires_at")
        return cls(
            access_token=access_token,
            refresh_token=refresh_token if isinstance(refresh_token, str) else "",
            expires_at=(
                expires_at
                if isinstance(expires_at, int) and not isinstance(expires_at, bool)
                else None
            ),
            uid=_optional_str(payload.get("uid")),
            enterprise_id=_optional_str(payload.get("enterprise_id")),
            domain=_optional_str(payload.get("domain")),
            nickname=_optional_str(payload.get("nickname")),
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "uid": self.uid,
            "enterprise_id": self.enterprise_id,
            "domain": self.domain,
            "nickname": self.nickname,
        }

    def hydrated(self) -> _Credentials:
        """Fill identity fields from unsigned JWT claims when absent."""

        claims = _optional_jwt_claims(self.access_token)
        if not claims:
            return self
        expires_at = self.expires_at
        exp = claims.get("exp")
        if expires_at is None and isinstance(exp, int | float):
            expires_at = int(exp)
        uid = self.uid
        sub = claims.get("sub")
        if uid is None and sub is not None:
            uid = str(sub)
        enterprise_id = self.enterprise_id
        domain = self.domain
        iss = str(claims.get("iss") or "")
        if enterprise_id is None:
            marker = iss.rfind("/sso-")
            if marker >= 0:
                enterprise_id = iss[marker + len("/sso-") :]
        if domain is None:
            host = iss.split("/", 3)
            if host[0] in {"http:", "https:"} and len(host) > 2:
                domain = host[2]
        return _Credentials(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=expires_at,
            uid=uid,
            enterprise_id=enterprise_id,
            domain=domain,
            nickname=self.nickname,
        )


class CodeBuddyAuthManager:
    """Own credentials, device login, refresh, and disconnection."""

    provider_id = "codebuddy"

    def __init__(
        self,
        *,
        site: CodeBuddySite | None = None,
        proxy: str | None = None,
        credential_path: Path | None = None,
        lock_path: Path | None = None,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._site = site or default_site()
        self._credential_path = credential_path or codebuddy_auth_path()
        self._lock_path = lock_path or codebuddy_auth_lock_path()
        self._client = client or httpx2.AsyncClient(
            proxy=proxy,
            timeout=httpx2.Timeout(30.0),
        )
        self._owns_client = client is None
        self._credentials: _Credentials | None = None
        self._revision = 0
        self._operation_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._login_task: asyncio.Task[None] | None = None
        self._attempt_id: str | None = None
        self._mode: ConnectedAccountLoginMode | None = None
        self._verification_url: str | None = None
        self._expires_at: int | None = None
        self._last_error: str | None = None
        self._closed = False
        try:
            self._credentials = self._read_credentials()
        except ValueError:
            self._last_error = (
                "Saved CodeBuddy credentials are invalid. Disconnect and sign in again."
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
            email=credentials.nickname if credentials is not None else None,
            display_identity=credentials.nickname if credentials is not None else None,
            mode=self._mode if connecting else None,
            verification_url=self._verification_url if connecting else None,
            expires_at=self._expires_at if connecting else None,
            message=self._last_error,
            supported_login_modes=(ConnectedAccountLoginMode.DEVICE,),
            default_login_mode=ConnectedAccountLoginMode.DEVICE,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        """Start the device authorization flow without exposing secrets."""

        if mode is not ConnectedAccountLoginMode.DEVICE:
            raise InvalidRequestError("CodeBuddy supports device-code login.")
        async with self._operation_lock:
            async with self._state_lock:
                self._ensure_open()
                if self._login_task is not None and not self._login_task.done():
                    return self.status()
                self._clear_attempt()
                self._last_error = None
                self._attempt_id = f"login_{uuid.uuid4().hex}"
                self._mode = mode
            try:
                authorization = await start_device_authorization(
                    self._client, self._site
                )
                async with self._state_lock:
                    self._verification_url = authorization.verification_url
                    self._expires_at = authorization.expires_at
                    self._login_task = asyncio.create_task(
                        self._complete_login(authorization),
                        name="codebuddy-device-login",
                    )
                    return self.status()
            except asyncio.CancelledError:
                async with self._state_lock:
                    self._clear_attempt()
                raise
            except Exception:
                async with self._state_lock:
                    self._clear_attempt()
                    self._last_error = "CodeBuddy sign-in could not start."
                raise

    async def cancel_login(self) -> ConnectedAccountStatus:
        """Cancel a pending login while preserving current credentials."""

        async with self._operation_lock:
            async with self._state_lock:
                task = self._login_task
                self._login_task = None
                self._clear_attempt()
                self._last_error = None
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        """Remove FCC-owned credentials; CodeBuddy has no FCC-initiated revoke."""

        async with self._operation_lock:
            async with self._state_lock:
                self._ensure_open()
                task = self._login_task
                self._login_task = None
                self._clear_attempt()
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            async with self._state_lock:
                await self._delete_credentials()
                self._credentials = None
                self._revision += 1
                self._last_error = None
                return self.status()

    async def access(self, *, force_refresh: bool = False) -> CodeBuddyAccess:
        """Return current identity headers, refreshing once before expiry."""

        async with self._state_lock:
            self._ensure_open()
            credentials = self._credentials
            if credentials is None:
                raise CodeBuddyReconnectRequired(
                    "Connect a CodeBuddy account in the FCC Admin UI."
                )
            if force_refresh:
                credentials = await self._refresh_locked(credentials)
            elif _credentials_expiring(credentials, REFRESH_EARLY_SECONDS):
                try:
                    credentials = await self._refresh_locked(credentials)
                except httpx2.HTTPError as exc:
                    if not _is_transient_refresh_error(exc):
                        raise
            return _access_from(credentials)

    async def recover_unauthorized(self, rejected_token: str) -> CodeBuddyAccess:
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
                raise CodeBuddyReconnectRequired(
                    "CodeBuddy credentials are no longer available. Reconnect in Admin."
                )
            if current.access_token != rejected_token:
                return _access_from(current)
            for attempt in range(_UNAUTHORIZED_REFRESH_TOTAL_ATTEMPTS):
                try:
                    return _access_from(await self._refresh_locked(current))
                except asyncio.CancelledError:
                    raise
                except httpx2.HTTPError as error:
                    if (
                        not _is_transient_refresh_error(error)
                        or attempt + 1 == _UNAUTHORIZED_REFRESH_TOTAL_ATTEMPTS
                    ):
                        raise
                await asyncio.sleep(_UNAUTHORIZED_REFRESH_RETRY_DELAY_SECONDS)
            raise RuntimeError("CodeBuddy refresh recovery ended without an outcome")

    async def close(self) -> None:
        """Cancel a pending login and close the owned HTTP client."""

        async with self._operation_lock:
            async with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                task = self._login_task
                self._login_task = None
                self._clear_attempt()
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if self._owns_client:
                await self._client.aclose()

    async def _complete_login(self, authorization: DeviceAuthorization) -> None:
        try:
            while time.time() < authorization.expires_at:
                tokens = await poll_device_tokens(
                    self._client, authorization, self._site
                )
                if tokens is not None:
                    profile = await fetch_account_profile(
                        self._client, authorization, tokens, self._site
                    )
                    credentials = _Credentials(
                        access_token=tokens.access_token,
                        refresh_token=tokens.refresh_token,
                        expires_at=tokens.expires_at,
                        uid=profile.uid if profile is not None else None,
                        enterprise_id=(
                            profile.enterprise_id if profile is not None else None
                        ),
                        domain=tokens.domain,
                        nickname=profile.nickname if profile is not None else None,
                    ).hydrated()
                    await self._write_credentials(credentials)
                    break
                await asyncio.sleep(DEVICE_POLL_INTERVAL_SECONDS)
            else:
                raise CodeBuddyLoginError("CodeBuddy device sign-in timed out.")
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._state_lock:
                self._last_error = (
                    "CodeBuddy sign-in failed or timed out. Retry Connect."
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
                    self._clear_attempt()

    async def _refresh_locked(self, current: _Credentials) -> _Credentials:
        if not current.refresh_token:
            await asyncio.to_thread(self._delete_credentials_unlocked)
            self._credentials = None
            self._revision += 1
            self._last_error = (
                "CodeBuddy sign-in expired. Reconnect the account in Admin."
            )
            raise CodeBuddyReconnectRequired(self._last_error)
        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=30.0)
        if not acquired:
            raise CodeBuddyReconnectRequired(
                "Timed out waiting for another FCC process to refresh CodeBuddy."
            )
        try:
            reloaded = await asyncio.to_thread(self._read_credentials)
            if reloaded is not None and (
                reloaded != current
                and not _credentials_expiring(reloaded, REFRESH_EARLY_SECONDS)
            ):
                self._credentials = reloaded
                return reloaded
            response = await self._client.post(
                f"{self._site.api_base}/v2/plugin/auth/token/refresh",
                headers=_refresh_headers(self._site, current),
                content="",
                timeout=httpx2.Timeout(_REFRESH_TIMEOUT_SECONDS),
            )
            if response.status_code in {400, 401, 403}:
                await asyncio.to_thread(self._delete_credentials_unlocked)
                self._credentials = None
                self._revision += 1
                self._last_error = (
                    "CodeBuddy sign-in expired. Reconnect the account in Admin."
                )
                raise CodeBuddyReconnectRequired(self._last_error)
            payload = _refresh_payload(response)
            refreshed = _Credentials(
                access_token=payload.access_token,
                refresh_token=payload.refresh_token or current.refresh_token,
                expires_at=payload.expires_at,
                uid=current.uid,
                enterprise_id=current.enterprise_id,
                domain=payload.domain or current.domain,
                nickname=current.nickname,
            ).hydrated()
            await asyncio.to_thread(self._write_credentials_unlocked, refreshed)
            self._credentials = refreshed
            self._revision += 1
            self._last_error = None
            return refreshed
        finally:
            await asyncio.to_thread(file_lock.release)

    async def _write_credentials(self, credentials: _Credentials) -> None:
        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=30.0)
        if not acquired:
            raise CodeBuddyLoginError("Could not lock the CodeBuddy credential file.")
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
            raise CodeBuddyReconnectRequired(
                "Could not lock the CodeBuddy credential file."
            )
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
            return _Credentials.from_json(payload.get("credentials")).hydrated()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("CodeBuddy credential file is invalid") from exc

    def _clear_attempt(self) -> None:
        self._attempt_id = None
        self._mode = None
        self._verification_url = None
        self._expires_at = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise CodeBuddyReconnectRequired(
                "CodeBuddy authentication is shutting down."
            )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _access_from(credentials: _Credentials) -> CodeBuddyAccess:
    return CodeBuddyAccess(
        access_token=credentials.access_token,
        uid=credentials.uid,
        enterprise_id=credentials.enterprise_id,
        domain=credentials.domain,
    )


def _credentials_expiring(credentials: _Credentials, early_seconds: int) -> bool:
    if not credentials.access_token:
        return True
    if credentials.expires_at is not None:
        return time.time() > credentials.expires_at - early_seconds
    claims = _optional_jwt_claims(credentials.access_token)
    exp = claims.get("exp") if claims else None
    if isinstance(exp, int | float):
        return time.time() > float(exp) - early_seconds
    return False


def _refresh_headers(site: CodeBuddySite, credentials: _Credentials) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": site.origin,
        "Referer": f"{site.origin}/",
        "User-Agent": site.user_agent,
        "X-Refresh-Token": credentials.refresh_token,
        "X-Auth-Refresh-Source": "workbuddy",
    }
    if credentials.enterprise_id:
        headers["X-Enterprise-Id"] = credentials.enterprise_id
    return headers


@dataclass(frozen=True, slots=True, repr=False)
class _RefreshPayload:
    access_token: str
    refresh_token: str | None
    expires_at: int | None
    domain: str | None


def _refresh_payload(response: httpx2.Response) -> _RefreshPayload:
    try:
        payload = response.json()
    except ValueError as exc:
        raise CodeBuddyReconnectRequired(
            f"CodeBuddy refresh response was not JSON (HTTP {response.status_code})."
        ) from exc
    if response.status_code >= 400 or not isinstance(payload, dict):
        raise CodeBuddyReconnectRequired(
            f"CodeBuddy token refresh failed (HTTP {response.status_code})."
        )
    if payload.get("code") != 0:
        raise CodeBuddyReconnectRequired(
            f"CodeBuddy token refresh was rejected (code={payload.get('code')})."
        )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise CodeBuddyReconnectRequired("CodeBuddy token refresh returned no data.")
    access_token = data.get("accessToken")
    if not isinstance(access_token, str) or not access_token:
        raise CodeBuddyReconnectRequired(
            "CodeBuddy token refresh returned no access token."
        )
    expires_in = data.get("expiresIn")
    expires_at = (
        int(time.time()) + int(expires_in)
        if isinstance(expires_in, int | float) and not isinstance(expires_in, bool)
        else None
    )
    refresh_token = data.get("refreshToken")
    domain = data.get("domain")
    return _RefreshPayload(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        expires_at=expires_at,
        domain=domain if isinstance(domain, str) else None,
    )


def _optional_jwt_claims(token: str) -> dict[str, Any]:
    try:
        encoded = token.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded))
    except IndexError, ValueError, json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_transient_refresh_error(error: httpx2.HTTPError) -> bool:
    if isinstance(error, httpx2.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx2.TransportError)
