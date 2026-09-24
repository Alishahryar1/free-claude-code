"""Async client for the Google Antigravity Cloud Code API."""

import asyncio
import hashlib
import json
import logging
import platform
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from .auth import AntigravityAuthManager
from .credentials import AntigravityCredentialError

REQUEST_USER_AGENT = "antigravity"
REQUEST_TYPE = "agent"
MODEL_CACHE_TTL_SECONDS = 3600.0

_LOGGER = logging.getLogger(__name__)


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
        read_timeout: float = 60.0,
        write_timeout: float = 60.0,
        connect_timeout: float = 60.0,
        proxy: str | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        timeout = httpx.Timeout(
            read_timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=connect_timeout,
        )
        self._client = client or httpx.AsyncClient(timeout=timeout, proxy=proxy)
        self._owns_client = client is None
        self._models_cache: Mapping[str, Any] | None = None
        self._models_cache_time = 0.0
        self._models_cache_revision = -1
        self._models_lock = asyncio.Lock()

    @property
    def auth_revision(self) -> int:
        """Return the FCC account revision used to invalidate account-scoped caches."""

        return self._auth.status().revision

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
        revision = self.auth_revision
        now = time.monotonic()
        if (
            self._models_cache is not None
            and self._models_cache_revision == revision
            and now - self._models_cache_time < MODEL_CACHE_TTL_SECONDS
        ):
            return self._models_cache

        # Admin/model-picker requests can arrive together. Collapse them into one
        # expensive Cloud Code fetch and share the result for the next hour, but
        # never across FCC account revisions.
        async with self._models_lock:
            revision = self.auth_revision
            now = time.monotonic()
            if (
                self._models_cache is not None
                and self._models_cache_revision == revision
                and now - self._models_cache_time < MODEL_CACHE_TTL_SECONDS
            ):
                return self._models_cache
            response = await self._json_request("/v1internal:fetchAvailableModels", {})
            models = _mapping(response, "fetchAvailableModels response")
            self._models_cache = models
            self._models_cache_time = time.monotonic()
            self._models_cache_revision = revision
            return models

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
        prepared.setdefault("requestId", _new_request_id())
        prepared["userAgent"] = REQUEST_USER_AGENT
        prepared["requestType"] = REQUEST_TYPE

        request_value = prepared.get("request")
        if isinstance(request_value, Mapping):
            request = dict(request_value)
            request.setdefault("sessionId", _derive_session_id(request.get("contents")))
            prepared["request"] = request
        return prepared

    async def _json_request(self, path: str, payload: Mapping[str, Any]) -> Any:
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


def _new_request_id() -> str:
    conversation_id = uuid.uuid4()
    trajectory_id = uuid.uuid4()
    return f"agent/{conversation_id}/{int(time.time() * 1000)}/{trajectory_id}/1"


def _derive_session_id(contents: Any) -> str:
    if isinstance(contents, list):
        for message in contents:
            if not isinstance(message, Mapping):
                continue
            if str(message.get("role") or "").lower() != "user":
                continue
            parts = message.get("parts")
            if not isinstance(parts, list):
                continue
            texts = [
                part["text"]
                for part in parts
                if isinstance(part, Mapping)
                and isinstance(part.get("text"), str)
                and part["text"]
            ]
            if texts:
                digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()
                return digest[:32]
    return str(uuid.uuid4())


def _platform_user_agent() -> str:
    return (
        "antigravity/cli/1.1.13 "
        f"(aidev_client; os_type={platform.system().lower()}; "
        f"arch={platform.machine().lower()}; cl=964361259; auth_method=consumer)"
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
    _LOGGER.warning(
        "Antigravity upstream error: status=%s body_chars=%s",
        response.status_code,
        len(text),
    )
    return AntigravityUpstreamError(
        response.status_code,
        f"Antigravity upstream returned {response.status_code}.",
        body=text,
    )
