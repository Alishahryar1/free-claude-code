"""Renewable Google Application Default Credentials for Vertex AI."""

import asyncio
from collections.abc import Callable

import google.auth
import requests
from google.auth.credentials import Credentials
from google.auth.exceptions import (
    DefaultCredentialsError,
    GoogleAuthError,
    RefreshError,
    TransportError,
)
from google.auth.transport.requests import Request

from free_claude_code.core.failures import ExecutionFailure, FailureKind

GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

CredentialsLoader = Callable[[], Credentials]


def load_application_default_credentials() -> Credentials:
    """Load ADC with the scope required by Vertex AI."""
    credentials, _project = google.auth.default(scopes=(GOOGLE_CLOUD_PLATFORM_SCOPE,))
    return credentials


class GoogleAccessTokenProvider:
    """Return a valid ADC access token, refreshing it without blocking the event loop."""

    def __init__(
        self,
        credentials_loader: CredentialsLoader = load_application_default_credentials,
        *,
        proxy: str | None = None,
    ) -> None:
        self._credentials_loader = credentials_loader
        self._proxy = proxy
        self._credentials: Credentials | None = None
        self._refresh_lock = asyncio.Lock()

    async def __call__(self) -> str:
        credentials = self._credentials
        if credentials is not None and credentials.valid and credentials.token:
            return credentials.token

        async with self._refresh_lock:
            credentials = self._credentials
            if credentials is not None and credentials.valid and credentials.token:
                return credentials.token
            try:
                if credentials is None:
                    credentials = await asyncio.to_thread(self._credentials_loader)
                    self._credentials = credentials
                if not credentials.valid or not credentials.token:
                    await asyncio.to_thread(self._refresh, credentials)
                token = credentials.token
                if not isinstance(token, str) or not token:
                    raise RefreshError("Google credentials returned no access token.")
                return token
            except ExecutionFailure:
                raise
            except GoogleAuthError as exc:
                raise _google_auth_failure(exc) from exc

    def _refresh(self, credentials: Credentials) -> None:
        with requests.Session() as session:
            if self._proxy:
                session.proxies.update({"http": self._proxy, "https": self._proxy})
            credentials.refresh(Request(session=session))


def _google_auth_failure(exc: GoogleAuthError) -> ExecutionFailure:
    if isinstance(exc, TransportError) or (
        isinstance(exc, RefreshError) and bool(getattr(exc, "retryable", False))
    ):
        return ExecutionFailure(
            kind=FailureKind.UNAVAILABLE,
            status_code=503,
            message=(
                "Google authentication is temporarily unavailable while refreshing "
                "Application Default Credentials."
            ),
            retryable=True,
        )
    if isinstance(exc, DefaultCredentialsError):
        message = (
            "Google Application Default Credentials were not found. Run "
            "`gcloud auth application-default login`, set "
            "GOOGLE_APPLICATION_CREDENTIALS, or attach a service account."
        )
    else:
        message = (
            "Google Application Default Credentials could not be refreshed. "
            "Reauthenticate with `gcloud auth application-default login` or check "
            "the configured service account."
        )
    return ExecutionFailure(
        kind=FailureKind.AUTHENTICATION,
        status_code=401,
        message=message,
        retryable=False,
    )
