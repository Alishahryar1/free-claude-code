"""Async client for the Google Antigravity Cloud Code API."""

from __future__ import annotations

import json
import platform
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from .auth import AntigravityAuthManager
from .credentials import AntigravityCredentialError

REQUEST_USER_AGENT = "antigravity"
REQUEST_TYPE = "agent"


class AntigravityUpstreamError(RuntimeError):
    """A Cloud Code request failed."""

    def __init__(self, status_code: int, message: str, *, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AntigravityClient:
    """Resolve native credentials and call Antigravity's Cloud Code endpoints."""

    def __init__(
        self,
        *,
        auth: AntigravityAuthManager,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def load_code_assist(self) -> Mapping[str, Any]:
        response = await self._json_request(
            "/v1internal:loadCodeAssist",
            {"metadata": {"ideType": "ANTIGRAVITY"}},
        )
        return _mapping(response, "loadCodeAssist response")

    async def fetch_available_models(self) -> Mapping[str, Any]:
        response = await self._json_request("/v1internal:fetchAvailableModels", {})
        return _mapping(response, "fetchAvailableModels response")

    async def generate_content(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = await self._json_request(
            "/v1internal:generateContent",
            self.prepare_request(payload),
        )
        return _mapping(response, "generateContent response")

    async def stream_generate_content(
        self, payload: Mapping[str, Any]
    ) -> AsyncIterator[Mapping[str, Any]]:
        response = await self._request(
            "POST",
            "/v1internal:streamGenerateContent?alt=sse",
            self.prepare_request(payload),
            stream=True,
        )
        if response.is_error:
            body = await response.aread()
            error = _upstream_error(response, body=body)
            await response.aclose()
            raise error
        try:
            async for line in response.aiter_lines():
                text = line.strip()
                if not text or text.startswith(":") or not text.startswith("data:"):
                    continue
                data = text.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError as error:
                    raise AntigravityUpstreamError(
                        502, "Antigravity returned an invalid SSE event."
                    ) from error
                yield _mapping(parsed, "stream event")
        finally:
            await response.aclose()

    def prepare_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        prepared = dict(payload)
        prepared.setdefault("requestId", f"agent-{uuid.uuid4()}")
        prepared["userAgent"] = REQUEST_USER_AGENT
        prepared["requestType"] = REQUEST_TYPE
        return prepared

    async def _json_request(
        self, path: str, payload: Mapping[str, Any]
    ) -> Any:
        response = await self._request("POST", path, payload)
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            raise _upstream_error(response) from error
        except ValueError as error:
            raise AntigravityUpstreamError(
                502, "Antigravity returned invalid JSON."
            ) from error
        finally:
            await response.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        stream: bool = False,
    ) -> httpx.Response:
        credentials = await self._auth.credentials()
        if credentials.expires_soon():
            raise AntigravityCredentialError(
                "The native Antigravity token is expired or expiring. "
                "Refresh the account with `agy`, then reconnect in FCC Admin."
            )
        response = await self._send(
            method,
            path,
            payload,
            credentials.access_token,
            stream=stream,
        )
        if response.status_code != 401:
            return response

        # The native CLI may have refreshed the token since our first read.
        await response.aclose()
        credentials = await self._auth.credentials()
        retry = await self._send(
            method,
            path,
            payload,
            credentials.access_token,
            stream=stream,
        )
        if retry.status_code == 401:
            await retry.aclose()
            raise AntigravityCredentialError(
                "Antigravity rejected the native `agy` session. "
                "Sign in with `agy` again, then reconnect in FCC Admin."
            )
        return retry

    async def _send(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
        token: str,
        *,
        stream: bool,
    ) -> httpx.Response:
        request = self._client.build_request(
            method,
            f"{self._base_url}{path}",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": _platform_user_agent(),
                "Accept": "text/event-stream" if stream else "application/json",
            },
        )
        return await self._client.send(request, stream=stream)


def _platform_user_agent() -> str:
    return (
        "antigravity/cli/1.1.13 "
        f"(aidev_client; os_type={platform.system().lower()}; "
        f"arch={platform.machine().lower()}; auth_method=consumer)"
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AntigravityUpstreamError(502, f"Invalid {label}.")
    return value


def _upstream_error(
    response: httpx.Response, *, body: bytes | None = None
) -> AntigravityUpstreamError:
    raw = response.content if body is None else body
    text = raw.decode("utf-8", errors="replace")
    preview = text[:1024]
    return AntigravityUpstreamError(
        response.status_code,
        f"Antigravity upstream returned {response.status_code}: {preview}",
        body=text,
    )
