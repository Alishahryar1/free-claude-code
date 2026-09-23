"""HTTP helpers for live smoke requests."""

from typing import Any

import httpx

from free_claude_code.core.anthropic.stream_contracts import SSEEvent, parse_sse_lines

from .config import SmokeConfig, auth_headers, redacted
from .server import RunningServer


def message_payload(
    text: str,
    *,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 128,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": text}],
    }
    if extra:
        payload.update(extra)
    return payload


def post_json(
    server: RunningServer,
    path: str,
    payload: dict[str, Any],
    config: SmokeConfig,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request_headers = headers or auth_headers()
    response = httpx.post(
        f"{server.base_url}{path}",
        headers=request_headers,
        json=payload,
        timeout=config.timeout_s,
    )
    return response


def collect_message_stream(
    server: RunningServer,
    payload: dict[str, Any],
    config: SmokeConfig,
    *,
    headers: dict[str, str] | None = None,
) -> list[SSEEvent]:
    request_headers = headers or auth_headers()
    stream_payload = {**payload, "stream": True}
    with httpx.stream(
        "POST",
        f"{server.base_url}/v1/messages",
        headers=request_headers,
        json=stream_payload,
        timeout=config.timeout_s,
    ) as response:
        if response.status_code != 200:
            body = response.read().decode("utf-8", errors="replace")
            raise AssertionError(
                f"stream request failed: HTTP {response.status_code} "
                f"{redacted(body[:1000])}"
            )
        return parse_sse_lines(response.iter_lines())
