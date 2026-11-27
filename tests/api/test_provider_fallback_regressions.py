"""End-to-end regressions: quota/TPM failures, provider fallback, compaction."""

import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Unpack

import httpx
import httpx2
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from free_claude_code.application.key_rotation import (
    KeyRotationRegistry,
    RotationScope,
    cooldown_seconds,
    failure_category,
    is_quota_failure,
    key_rotation_registry,
    rotation_scope,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.admission_capacity import ProviderCapacityLimiter
from free_claude_code.providers.failure_classification import classify
from tests.api.support import create_test_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_fallback_key_requests import (
    _NativeClientOptions,
    _SDKClientOptions,
)
from tests.providers.test_anthropic_messages_transport import _sse
from tests.providers.upstream_errors import http_error
from tests.wire_payload import decode_object

GROQ_KEYS = ["synthetic-groq-1", "synthetic-groq-2", "synthetic-groq-3"]
OPENAI_KEYS = ["synthetic-openai-1"]
ALL_KEYS = GROQ_KEYS + OPENAI_KEYS
MODEL = "groq/same-model"

type Failure = tuple[int, JsonObject, dict[str, str]]
type Handler = Callable[[str], Failure | None]


def groq_tpm_413(limit: int = 8000, requested: int = 52172) -> Failure:
    return (
        413,
        {
            "error": {
                "message": (
                    "Request too large for model on tokens per minute (TPM): "
                    f"Limit {limit}, Requested {requested}"
                ),
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
        {"retry-after": "30"},
    )


def coarena_daily_429() -> Failure:
    return (
        429,
        {
            "error": {
                "type": "rate_limit_error",
                "message": "Today's free usage is used up.",
            }
        },
        {},
    )


def plain_413() -> Failure:
    return (
        413,
        {
            "error": {
                "type": "invalid_request_error",
                "code": "request_too_large",
                "message": "Payload too large",
            }
        },
        {},
    )


def failing(provider: str, failure: Failure) -> Handler:
    return lambda name: failure if name == provider else None


@dataclass
class Wire:
    """Recorded upstream calls as ``(provider, token)`` and the failure policy."""

    calls: list[tuple[str, str]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    handler: Handler = lambda _provider: None


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Iterator[Wire]:
    """Mock both HTTP stacks; ``handler(provider)`` may fail a call."""
    state = Wire()
    build = ProviderAdmissionController.__init__

    def fast_backoff(
        self: ProviderAdmissionController,
        *,
        provider_name: str,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
        max_attempts: int = 5,
        capacity: ProviderCapacityLimiter | None = None,
    ) -> None:
        build(
            self,
            provider_name=provider_name,
            rate_limit=rate_limit,
            rate_window=rate_window,
            max_concurrency=max_concurrency,
            max_attempts=max_attempts,
            base_delay=0.01,
            max_delay=0.01,
            jitter=0.0,
            capacity=capacity,
        )

    monkeypatch.setattr(ProviderAdmissionController, "__init__", fast_backoff)

    def respond[T: (httpx.Response, httpx2.Response)](
        request: httpx.Request | httpx2.Request, response_type: type[T]
    ) -> T:
        if request.method == "GET":
            return response_type(200, json={"id": "same-model", "data": []})
        token = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        provider = (
            "groq" if request.url.path.endswith("/chat/completions") else "openai_api"
        )
        state.calls.append((provider, token))
        payload = decode_object(request.content)
        model = payload["model"]
        assert isinstance(model, str)
        state.models.append(model)
        failure = state.handler(provider)
        if failure is not None:
            status, body, headers = failure
            return response_type(status, json=body, headers=headers)
        protocol = "chat" if provider == "groq" else "responses"
        events = text_events(protocol, f"served by {provider}")
        raw = (
            "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
            if protocol == "chat"
            else _sse(*events)
        )
        return response_type(
            200, headers={"content-type": "text/event-stream"}, content=raw
        )

    native = httpx.MockTransport(lambda request: respond(request, httpx.Response))
    sdk = httpx2.MockTransport(lambda request: respond(request, httpx2.Response))
    native_init = httpx.AsyncClient.__init__
    sdk_init = httpx2.AsyncClient.__init__

    def init_native(
        client: httpx.AsyncClient,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        proxy: str | httpx.URL | httpx.Proxy | None = None,
        **options: Unpack[_NativeClientOptions],
    ) -> None:
        native_init(client, transport=native, **options)

    def init_sdk(
        client: httpx2.AsyncClient,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
        proxy: str | httpx2.URL | httpx2.Proxy | None = None,
        **options: Unpack[_SDKClientOptions],
    ) -> None:
        sdk_init(client, transport=sdk, **options)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init_native)
    monkeypatch.setattr(httpx2.AsyncClient, "__init__", init_sdk)
    yield state


def settings(fallback_model: str = "same-model") -> Settings:
    return Settings(
        MODEL=MODEL,
        GROQ_API_KEY=GROQ_KEYS,
        OPENAI_API_KEY=OPENAI_KEYS,
        PROVIDER_RATE_LIMIT=1000,
        PROVIDER_RATE_WINDOW=1,
        MODEL_FALLBACKS=[{"model": f"openai_api/{fallback_model}", "reasoning": "off"}],
    )


def post(app: FastAPI, content: str = "hello") -> httpx.Response:
    with TestClient(app) as client:
        return client.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": content}],
            },
        )


def groq_calls(wire: Wire) -> list[str]:
    return [token for provider, token in wire.calls if provider == "groq"]


def test_groq_tpm_413_falls_back_to_other_provider_without_sibling_keys(
    wire: Wire,
) -> None:
    wire.handler = failing("groq", groq_tpm_413())
    result = post(create_test_app(settings()))
    assert result.status_code == 200, result.text
    assert "served by openai_api" in result.text
    # An org-wide TPM cap is not worked around by rotating sibling keys.
    assert len(groq_calls(wire)) == 1
    assert wire.calls[-1][0] == "openai_api"
    assert all(key not in result.text for key in ALL_KEYS)


def test_groq_tpm_cools_all_groq_keys_so_next_request_skips_groq(wire: Wire) -> None:
    wire.handler = failing("groq", groq_tpm_413())
    app = create_test_app(settings())
    post(app)
    wire.calls.clear()
    second = post(app)
    assert second.status_code == 200
    assert [provider for provider, _ in wire.calls] == ["openai_api"]


@pytest.mark.parametrize("endpoint", ["messages", "responses"])
def test_cooling_primary_sends_fallback_model_on_first_available_route(
    wire: Wire, endpoint: str
) -> None:
    wire.handler = failing("groq", groq_tpm_413())
    app = create_test_app(settings("fallback-model"))
    payload: JsonObject = {"model": MODEL, "stream": True}
    if endpoint == "messages":
        payload.update(
            max_tokens=64,
            messages=[{"role": "user", "content": "hello"}],
        )
    else:
        payload["input"] = "hello"
    with TestClient(app) as client:
        first = client.post(f"/v1/{endpoint}", json=payload)
        assert first.status_code == 200, first.text
        wire.calls.clear()
        wire.models.clear()
        second = client.post(f"/v1/{endpoint}", json=payload)
    assert second.status_code == 200, second.text
    assert [provider for provider, _ in wire.calls] == ["openai_api"]
    assert wire.models == ["fallback-model"]


def test_oversized_request_skips_provider_with_learned_tpm_cap(
    wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compaction-sized request is not sent where it can never fit."""
    wire.handler = failing("groq", groq_tpm_413())
    app = create_test_app(settings())
    post(app)  # learns Groq's cap (8000) and cools its keys
    real = time.time
    monkeypatch.setattr(
        "free_claude_code.application.key_rotation.time.time", lambda: real() + 120
    )  # the 30s Retry-After cooldown has expired; the learned cap has not
    wire.calls.clear()
    result = post(app, "x" * 200_000)  # ~50k tokens, like the reported 52172
    assert result.status_code == 200, result.text
    assert [provider for provider, _ in wire.calls] == ["openai_api"]


def test_coarena_style_daily_429_rotates_keys_then_falls_back(wire: Wire) -> None:
    wire.handler = failing("groq", coarena_daily_429())
    result = post(create_test_app(settings()))
    assert result.status_code == 200, result.text
    assert "served by openai_api" in result.text
    assert set(groq_calls(wire)) == set(GROQ_KEYS)  # every key, then the next provider
    rows = [
        row
        for row in key_rotation_registry.status(settings())
        if row["provider_id"] == "groq"
    ]
    assert len(rows) == len(GROQ_KEYS)
    for row in rows:
        assert row["status"] == "cooldown"
        cooldown = row["cooldown_seconds"]
        assert isinstance(cooldown, int)
        assert cooldown > 3600  # daily cooldown
    assert all(key not in str(rows) for key in GROQ_KEYS)


def test_ordinary_413_does_not_rotate_or_cool_keys(wire: Wire) -> None:
    wire.handler = failing("groq", plain_413())
    post(create_test_app(settings()))
    assert len(groq_calls(wire)) == 1
    assert all(
        row["status"] == "available" for row in key_rotation_registry.status(settings())
    )


@pytest.mark.parametrize("status", [400, 401])
def test_auth_and_bad_request_do_not_rotate_keys(wire: Wire, status: int) -> None:
    wire.handler = failing(
        "groq",
        (
            status,
            {"error": {"type": "invalid_request_error", "message": "rate limit"}},
            {},
        ),
    )
    post(create_test_app(settings()))
    assert groq_calls(wire) == [GROQ_KEYS[0]]
    assert all(
        row["status"] == "available" for row in key_rotation_registry.status(settings())
    )


def test_everything_exhausted_fails_bounded_and_cleanly(wire: Wire) -> None:
    wire.handler = lambda _provider: coarena_daily_429()
    app = create_test_app(settings())
    result = post(app)
    assert result.status_code == 429
    assert len(wire.calls) <= 5 * len(ALL_KEYS)  # bounded: no endless retry
    after = len(wire.calls)
    again = post(app)
    assert again.status_code == 429
    assert len(wire.calls) == after  # cooled routes answer without upstream calls
    assert all(key not in result.text + again.text for key in ALL_KEYS)


def test_concurrent_requests_do_not_corrupt_registry() -> None:
    registry = KeyRotationRegistry()

    def work(index: int) -> None:
        for _ in range(50):
            registry.mark("p", index % 3 + 1, f"k{index % 3}", http_error(429))
            registry.available("p", index % 3 + 1, f"k{index % 3}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not any(registry.available("p", n + 1, f"k{n}") for n in range(3))


def _raw(status: int, body: JsonObject) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://upstream.example/v1/chat/completions")
    response = httpx2.Response(status, json={"error": body}, request=request)
    return openai.APIStatusError(f"Error code: {status}", response=response, body=body)


def _error_body(failure: Failure) -> JsonObject:
    body = failure[1]["error"]
    assert isinstance(body, dict)
    return body


def test_structured_413_keeps_invalid_request_verdict_but_scopes_provider() -> None:
    """Client-facing 413 semantics are unchanged; rotation sees the TPM cap."""
    error = _raw(413, _error_body(groq_tpm_413()))
    assert classify(error).kind is FailureKind.INVALID_REQUEST
    assert rotation_scope(error) is RotationScope.PROVIDER


def test_classifier_keeps_ordinary_413_as_oversized_payload() -> None:
    error = _raw(413, _error_body(plain_413()))
    verdict = classify(error)
    assert verdict.kind is FailureKind.INVALID_REQUEST
    assert verdict.status_code == 413
    assert not is_quota_failure(error)


def test_classifier_coarena_429_is_retryable_rate_limit_with_daily_cooldown() -> None:
    error = _raw(429, _error_body(coarena_daily_429()))
    assert classify(error).kind is FailureKind.RATE_LIMIT
    assert classify(error).retryable is True
    assert rotation_scope(error) is RotationScope.KEY
    assert cooldown_seconds(error) == 86400
    assert failure_category(error) == "daily_quota"


def test_retry_after_is_respected_for_tpm_cooldown() -> None:
    error = http_error(
        413,
        code="rate_limit_exceeded",
        message="tokens per minute: Limit 8000, Requested 9000",
        headers={"retry-after": "42"},
    )
    assert rotation_scope(error) is RotationScope.PROVIDER
    assert cooldown_seconds(error) == 42


def test_no_key_material_in_logs_status_or_responses(wire: Wire) -> None:
    captured: list[str] = []
    sink = logger.add(lambda message: captured.append(str(message)), level="DEBUG")
    wire.handler = failing("groq", coarena_daily_429())
    try:
        result = post(create_test_app(settings()))
    finally:
        logger.remove(sink)
    blob = (
        "\n".join(captured)
        + result.text
        + str(key_rotation_registry.status(settings()))
    )
    assert result.status_code == 200
    assert all(key not in blob for key in ALL_KEYS)
