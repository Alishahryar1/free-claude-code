"""Numbered fallback credentials reach actual HTTP requests across transports."""

import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Unpack

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from typing_extensions import TypedDict

from free_claude_code.config.constants import PROVIDER_MAX_ATTEMPTS_DEFAULT
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.admission_capacity import ProviderCapacityLimiter
from tests.api.support import create_test_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.fixture_parameters import fixture_parameter
from tests.providers.test_anthropic_messages_transport import _sse
from tests.wire_payload import decode_object

KEYS = ["synthetic-first", "synthetic-second", "synthetic-third"]
CUSTOM_ID = "custom_" + "d" * 32


class _NativeClientOptions(TypedDict, total=False, closed=True):
    timeout: httpx.Timeout
    follow_redirects: bool
    event_hooks: Mapping[str, list[Callable[[httpx.Request], Awaitable[None]]]]


class _SDKClientOptions(TypedDict, total=False, closed=True):
    timeout: httpx2.Timeout | float | None
    follow_redirects: bool
    limits: httpx2.Limits


@pytest.fixture(params=[403, 429], ids=["permission-rejection", "rate-limit"])
def upstream(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, tmp_path: Path
):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    calls = []
    failure_status: int = fixture_parameter(request, int)
    build = ProviderAdmissionController.__init__

    def fast_backoff(
        self: ProviderAdmissionController,
        *,
        provider_name: str,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
        max_attempts: int = PROVIDER_MAX_ATTEMPTS_DEFAULT,
        capacity: ProviderCapacityLimiter | None = None,
    ) -> None:
        build(
            self,
            provider_name=provider_name,
            rate_limit=rate_limit,
            rate_window=rate_window,
            max_concurrency=max_concurrency,
            max_attempts=max_attempts,
            base_delay=0.1,
            max_delay=0.1,
            jitter=0.0,
            capacity=capacity,
        )

    monkeypatch.setattr(ProviderAdmissionController, "__init__", fast_backoff)

    def payload(
        request: httpx.Request | httpx2.Request,
    ) -> tuple[int, JsonObject | bytes]:
        if request.method == "GET":
            return 200, {"id": "same-model", "data": []}
        token = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        body = decode_object(request.content)
        calls.append((token, body))
        if token != "synthetic-third":
            return failure_status, {
                "error": {
                    "type": "rate_limit_error"
                    if failure_status == 429
                    else "permission_error",
                    "message": "No quota",
                }
            }
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

    def response[T: (httpx.Response, httpx2.Response)](
        request: httpx.Request | httpx2.Request, response_type: type[T]
    ) -> T:
        status, value = payload(request)
        if isinstance(value, dict):
            return response_type(status, json=value)
        return response_type(
            status, headers={"content-type": "text/event-stream"}, content=value
        )

    native_transport = httpx.MockTransport(
        lambda request: response(request, httpx.Response)
    )
    sdk_transport = httpx2.MockTransport(
        lambda request: response(request, httpx2.Response)
    )

    native_init = httpx.AsyncClient.__init__
    sdk_init = httpx2.AsyncClient.__init__

    # Inject the mocks before the clients can load TLS settings or mount proxies.
    def init_native_client(
        client: httpx.AsyncClient,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        proxy: str | httpx.URL | httpx.Proxy | None = None,
        **options: Unpack[_NativeClientOptions],
    ) -> None:
        native_init(client, transport=native_transport, **options)

    def init_sdk_client(
        client: httpx2.AsyncClient,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
        proxy: str | httpx2.URL | httpx2.Proxy | None = None,
        **options: Unpack[_SDKClientOptions],
    ) -> None:
        sdk_init(client, transport=sdk_transport, **options)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init_native_client)
    monkeypatch.setattr(httpx2.AsyncClient, "__init__", init_sdk_client)
    return calls, failure_status


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
    upstream: tuple[list[tuple[str, JsonObject]], int],
    provider: str,
    api_format: str,
    wire: str,
) -> None:
    calls, failure_status = upstream
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
    tokens = [token for token, _ in calls]
    assert [
        token
        for index, token in enumerate(tokens)
        if not index or token != tokens[index - 1]
    ] == KEYS
    if failure_status == 429:
        assert tokens == (
            [KEYS[0]] * PROVIDER_MAX_ATTEMPTS_DEFAULT
            + [KEYS[1]] * PROVIDER_MAX_ATTEMPTS_DEFAULT
            + [KEYS[2]]
        )
    assert all(body["model"] == "same-model" for _, body in calls)
    assert all(key not in result.text for key in KEYS)
