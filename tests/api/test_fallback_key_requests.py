"""Numbered fallback credentials reach actual HTTP requests across transports."""

import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.providers.test_anthropic_messages_transport import _sse

KEYS = ["synthetic-first", "synthetic-second", "synthetic-third"]
CUSTOM_ID = "custom_" + "d" * 32


@pytest.fixture
def upstream(monkeypatch):
    calls = []

    def payload(request):
        if request.method == "GET":
            return 200, {"id": "same-model", "data": []}
        token = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        body = json.loads(request.content)
        calls.append((token, body))
        if token != "synthetic-third":
            return 403, {"error": {"type": "permission_error", "message": "No quota"}}
        protocol = (
            "messages"
            if request.url.path.endswith("/messages")
            else "responses"
            if request.url.path.endswith("/responses")
            else "chat"
        )
        events = text_events(protocol, "selected key worked")
        raw = (
            "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
            if protocol == "chat"
            else _sse(*events)
        )
        return 200, raw

    def response(request, library):
        status, value = payload(request)
        if isinstance(value, dict):
            return library.Response(status, json=value)
        return library.Response(
            status, headers={"content-type": "text/event-stream"}, content=value
        )

    init = httpx.AsyncClient.__init__
    init2 = httpx2.AsyncClient.__init__

    def local_http(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda request: response(request, httpx)
        )
        init(self, *args, **kwargs)

    def local_http2(self, *args, **kwargs):
        kwargs["transport"] = httpx2.MockTransport(
            lambda request: response(request, httpx2)
        )
        init2(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", local_http)
    monkeypatch.setattr(httpx2.AsyncClient, "__init__", local_http2)
    return calls


@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    ("provider", "api_format"),
    [
        ("groq", "openai_chat"),
        ("openai_api", "openai_responses"),
        ("anthropic", "anthropic_messages"),
        (CUSTOM_ID, "openai_chat"),
        (CUSTOM_ID, "openai_responses"),
        (CUSTOM_ID, "anthropic_messages"),
    ],
)
def test_fallback_sequence_uses_each_rows_selected_key(
    upstream, provider, api_format, wire
):
    model = f"{provider}/same-model"
    custom = (
        [
            {
                "provider_id": CUSTOM_ID,
                "display_name": "Gateway",
                "base_url": "https://gateway.example/v1",
                "api_format": api_format,
                "api_keys": KEYS,
            }
        ]
        if provider == CUSTOM_ID
        else []
    )
    settings = Settings(
        MODEL=model,
        GROQ_API_KEY=KEYS,
        OPENAI_API_KEY=KEYS,
        ANTHROPIC_API_KEY=KEYS,
        PROVIDER_RATE_LIMIT=1000,
        PROVIDER_RATE_WINDOW=1,
        FCC_CUSTOM_PROVIDERS=custom,
        MODEL_FALLBACKS=[
            {"model": model, "reasoning": "off", "api_key_number": 2},
            {"model": model, "reasoning": "off", "api_key_number": 3},
        ],
    )
    body = (
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 64}
        if wire == "messages"
        else {"input": "hello"}
    )
    app = create_test_app(settings)
    with TestClient(app) as client:
        result = client.post(
            f"/v1/{wire}", json={"model": model, "stream": True, **body}
        )
    assert result.status_code == 200, result.text
    assert "selected key worked" in result.text
    tokens = [token for token, _ in upstream]
    assert [
        token
        for index, token in enumerate(tokens)
        if not index or token != tokens[index - 1]
    ] == KEYS
    assert all(body["model"] == "same-model" for _, body in upstream)
    assert all(key not in result.text for key in KEYS)
