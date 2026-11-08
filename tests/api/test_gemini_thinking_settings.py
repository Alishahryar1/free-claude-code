"""Configured and native thinking controls retain their distinct ownership."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.config.settings import Settings
from free_claude_code.providers.gemini import GeminiProvider
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_gemini import _sse


@pytest.mark.parametrize(
    "root, route, fallback, expected",
    [
        ("client", "inherit", None, {"thinkingLevel": "HIGH"}),
        ("high", "inherit", None, {"thinkingBudget": 2048}),
        ("client", "high", None, {"thinkingBudget": 2048}),
        ("high", "client", None, {"thinkingLevel": "HIGH"}),
        ("max", "inherit", None, {"thinkingBudget": 8192}),
        ("off", "inherit", None, {"thinkingBudget": 0}),
        ("high", "inherit", "inherit", {"thinkingBudget": 2048}),
        ("high", "inherit", "client", {"thinkingLevel": "HIGH"}),
        ("client", "inherit", "low", {"thinkingBudget": 512}),
        ("client", "inherit", "off", {"thinkingBudget": 0}),
    ],
)
@pytest.mark.parametrize("visible", [False, True])
def test_native_root_route_and_fallback_precedence(
    root, route, fallback, expected, visible
):
    sent = []

    def reply(request):
        body = json.loads(request.content)
        sent.append((request.url.path, body))
        if fallback is not None and "/primary:" in request.url.path:
            return httpx.Response(
                503, json={"error": {"message": "temporarily unavailable"}}
            )
        return _sse([{"text": "answer"}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=1),
        transport=httpx.MockTransport(reply),
    )
    settings = Settings(
        MODEL="gemini/primary",
        REASONING_POLICY=root,
        REASONING_SONNET=route,
        MODEL_FALLBACKS=(
            [{"model": "gemini/secondary", "reasoning": fallback}]
            if fallback is not None
            else []
        ),
    )
    app = create_test_app(settings, providers={"gemini": provider})
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "generationConfig": {
            "temperature": 0.3,
            "thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": visible},
        },
    }
    with TestClient(app) as client:
        response = client.post("/v1beta/models/sonnet:generateContent", json=payload)
    assert response.status_code == 200, response.text
    assert len(sent) == (2 if fallback is not None else 1)
    config = sent[-1][1]["generationConfig"]
    assert config["thinkingConfig"] == {
        "includeThoughts": False if expected.get("thinkingBudget") == 0 else visible,
        **expected,
    }
    assert config["temperature"] == 0.3
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
