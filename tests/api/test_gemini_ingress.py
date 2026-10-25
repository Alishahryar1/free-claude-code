"""Documented Gemini generation routes own authentication and response lifetime."""

import json
import logging
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from free_claude_code.cli.uvicorn_server import (
    QueryCredentialFilter,
    uvicorn_log_config,
)
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, provider_manager_for_app


def body():
    return {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model", ["model:free", "model%3Afree", "model:region:free"])
def test_routed_model_preserves_colons(streaming, model):
    provider, calls = fixture_provider()
    app = create_test_app(Settings(MODEL="groq/backend"), providers={"groq": provider})
    action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    with TestClient(app) as client:
        response = client.post(f"/v1beta/models/groq/{model}:{action}", json=body())
    assert response.status_code == 200, response.text
    assert calls[0][0].model == model.replace("%3A", ":")
    assert calls[0][1]["response_model"] == "groq/" + model.replace("%3A", ":")


def fixture_provider():
    provider = Mock()
    provider.cleanup = AsyncMock()
    provider.list_model_infos = AsyncMock(return_value=frozenset())
    calls = []

    async def stream(request, **kwargs):
        calls.append((request, kwargs))
        for part in ({"text": "Hello "}, {"text": "world"}):
            yield (
                "data: "
                + json.dumps(
                    {
                        "responseId": "fcc_test",
                        "modelVersion": kwargs["response_model"],
                        "candidates": [
                            {"index": 0, "content": {"role": "model", "parts": [part]}}
                        ],
                    }
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {
                    "responseId": "fcc_test",
                    "modelVersion": kwargs["response_model"],
                    "candidates": [{"index": 0, "finishReason": "STOP"}],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 5,
                    },
                }
            )
            + "\n\n"
        )

    provider.stream_gemini = stream
    return provider, calls


@pytest.mark.parametrize("streaming", [False, True])
def test_generation_routes_use_the_selected_backend_and_release_lease(streaming):
    provider, calls = fixture_provider()
    app = create_test_app(Settings(MODEL="groq/backend"), providers={"groq": provider})
    suffix = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    with TestClient(app) as client:
        response = client.post(f"/v1beta/models/public-model:{suffix}", json=body())
    assert response.status_code == 200, response.text
    if streaming:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "[DONE]" not in response.text and "event:" not in response.text
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["candidates"][0]["finishReason"] == "STOP"
    else:
        value = response.json()
        assert value["candidates"][0]["content"]["parts"] == [{"text": "Hello world"}]
        assert value["usageMetadata"]["totalTokenCount"] == 5
    assert len(calls) == 1
    assert calls[0][0].model == "backend"
    assert calls[0][1]["response_model"] == "public-model"
    assert provider_manager_for_app(app)._current.active_leases == 0


@pytest.mark.parametrize("credential", ["header", "query", "bearer"])
def test_gemini_credentials_authenticate_the_proxy_not_the_upstream(credential):
    provider, calls = fixture_provider()
    app = create_test_app(
        Settings(
            MODEL="groq/backend",
            PROXY_AUTH_ENABLED=True,
            proxy_auth_token="fcc-test-token",
        ),
        providers={"groq": provider},
    )
    headers = (
        {"x-goog-api-key": "fcc-test-token"}
        if credential == "header"
        else {"authorization": "Bearer fcc-test-token"}
        if credential == "bearer"
        else {}
    )
    query = "?key=fcc-test-token" if credential == "query" else ""
    with TestClient(app) as client:
        response = client.post(
            "/v1beta/models/model:generateContent" + query, headers=headers, json=body()
        )
    assert response.status_code == 200, response.text
    forwarded = calls[0][1].get("request_headers") or {}
    assert "x-goog-api-key" not in forwarded
    assert "authorization" not in forwarded


@pytest.mark.parametrize(
    "headers,query",
    [
        ({}, ""),
        ({"x-goog-api-key": "wrong"}, "?key=fcc-test-token"),
        ({"authorization": "Bearer wrong", "x-goog-api-key": "fcc-test-token"}, ""),
        ({}, "?key=fcc-test-token&key=other"),
    ],
)
def test_invalid_auth_is_a_google_error_before_inference(headers, query):
    provider, calls = fixture_provider()
    app = create_test_app(
        Settings(
            MODEL="groq/backend",
            PROXY_AUTH_ENABLED=True,
            proxy_auth_token="fcc-test-token",
        ),
        providers={"groq": provider},
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1beta/models/model:generateContent" + query, headers=headers, json=body()
        )
    assert response.status_code in {400, 401}, response.text
    assert response.json()["error"]["code"] == response.status_code
    assert response.json()["error"]["status"] in {"UNAUTHENTICATED", "INVALID_ARGUMENT"}
    assert not calls


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {"contents": []},
        {**body(), "cachedContent": "caches/foreign"},
        {**body(), "generationConfig": {"candidateCount": 2}},
    ],
)
def test_invalid_or_unsupported_requests_are_rejected_before_inference(request_body):
    provider, calls = fixture_provider()
    app = create_test_app(Settings(MODEL="groq/backend"), providers={"groq": provider})
    with TestClient(app) as client:
        response = client.post(
            "/v1beta/models/model:generateContent", json=request_body
        )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert not calls


@pytest.mark.parametrize(
    "method,path,status",
    [
        ("GET", "/v1beta/models/model:generateContent", 405),
        ("POST", "/v1beta/models/model:unknown", 404),
        ("POST", "/v1beta/models/:generateContent", 400),
        ("POST", "/v1beta/models/model:streamGenerateContent", 400),
        ("POST", "/v1beta/models/model:generateContent?alt=sse", 400),
    ],
)
def test_gemini_route_family_errors_use_google_envelopes(method, path, status):
    provider, calls = fixture_provider()
    app = create_test_app(Settings(MODEL="groq/backend"), providers={"groq": provider})
    with TestClient(app) as client:
        response = client.request(method, path, json=body())
    assert response.status_code == status
    assert response.json()["error"]["code"] == status
    assert response.headers["request-id"] == response.headers["x-request-id"]
    assert not calls


def test_repeated_material_json_key_is_rejected_before_inference():
    provider, calls = fixture_provider()
    app = create_test_app(Settings(MODEL="groq/backend"), providers={"groq": provider})
    with TestClient(app) as client:
        response = client.post(
            "/v1beta/models/model:generateContent",
            content='{"contents":[],"contents":[{"parts":[{"text":"ok"}]}]}',
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert not calls


@pytest.mark.parametrize("console", [False, True])
def test_query_api_key_is_masked_before_uvicorn_console_and_file_handlers(console):
    config = uvicorn_log_config(console=console)
    loggers = config["loggers"]
    assert isinstance(loggers, dict)
    assert loggers["uvicorn.access"]["filters"] == ["query_credentials"]
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        (
            "127.0.0.1",
            "POST",
            "/v1beta/models/m:generateContent?ke%79=SECRET&alt=json&key=SECOND",
            "1.1",
            400,
        ),
        None,
    )
    assert QueryCredentialFilter().filter(record)
    assert "SECRET" not in record.getMessage() and "SECOND" not in record.getMessage()
    assert "alt=json" in record.getMessage()
