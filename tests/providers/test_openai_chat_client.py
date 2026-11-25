"""Tests for OpenAI-chat SDK client construction."""

import pytest

from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import create_chat_client


def _config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test-key",
        base_url="http://localhost:20128/v1",
        rate_limit=120,
        rate_window=60,
        max_concurrency=8,
        http_read_timeout=120.0,
        http_write_timeout=10.0,
        http_connect_timeout=10.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("http://localhost:20128/v1", "http://127.0.0.1:20128/v1"),
        ("http://localhost:20128", "http://127.0.0.1:20128"),
        ("http://localhost/v1", "http://127.0.0.1/v1"),
        ("https://localhost:8443/v1", "https://127.0.0.1:8443/v1"),
        ("http://.localhost:20128/v1", "http://127.0.0.1:20128/v1"),
        ("https://api.tokenrouter.com/v1", "https://api.tokenrouter.com/v1"),
        ("http://127.0.0.1:20128/v1", "http://127.0.0.1:20128/v1"),
        ("http://192.168.1.5:8080/v1", "http://192.168.1.5:8080/v1"),
        ("http://[::1]:20128/v1", "http://[::1]:20128/v1"),
    ],
)
def test_normalize_localhost_base_url(source: str, expected: str) -> None:
    from free_claude_code.providers.openai_chat.client import (
        _normalize_localhost_base_url,
    )

    assert _normalize_localhost_base_url(source) == expected


def test_create_chat_client_normalizes_localhost_base_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        FakeClient,
    )
    create_chat_client(
        _config(),
        base_url="http://localhost:20128/v1",
        provider_name="tokenrouter",
    )
    assert captured["base_url"] == "http://127.0.0.1:20128/v1"