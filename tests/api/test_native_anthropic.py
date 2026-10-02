"""Messages ingress chooses one native contract before lossy processing."""

import json
from typing import cast
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider
from tests.api.support import create_test_app
from tests.providers.test_anthropic_provider import native_body, provider


def test_native_ingress_preserves_extensions_and_bypasses_local_processing():
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "type": "message",
                "model": "claude-test",
                "content": [{"type": "new_hosted_result", "value": 42}],
                "stop_reason": "pause_turn",
            },
        )

    p = provider(handle)
    settings = Settings(
        MODEL="anthropic/claude-test",
        ANTHROPIC_API_KEY="key",
        ENABLE_WEB_SERVER_TOOLS=False,
    )
    app = create_test_app(settings, providers={"anthropic": p})
    body = native_body()
    body["model"] = "claude-sonnet-client-alias"
    with (
        patch(
            "free_claude_code.api.handlers.messages.try_optimizations",
            side_effect=AssertionError("native optimization"),
        ),
        patch(
            "free_claude_code.runtime.provider_manager.ProviderGenerationLease.wait_for_token_estimation",
            side_effect=AssertionError("native token wait"),
        ),
        patch(
            "free_claude_code.api.routes.get_token_count",
            side_effect=AssertionError("native counting"),
        ),
        patch(
            "free_claude_code.application.web_tools.service.WebToolService.try_stream_messages",
            side_effect=AssertionError("local tools"),
        ),
        TestClient(app) as client,
    ):
        response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == body["model"]
    assert response.json()["stop_reason"] == "pause_turn"
    assert sent == [{**body, "model": "claude-test"}]


def test_native_fallback_excludes_compatibility_and_preserves_alias():
    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        if body["model"] == "primary":
            return httpx.Response(
                403,
                json={
                    "type": "error",
                    "error": {"type": "permission_error", "message": "unavailable"},
                },
            )
        return httpx.Response(
            200, json={"type": "message", "model": "backup", "content": []}
        )

    settings = Settings(
        MODEL="anthropic/primary",
        ANTHROPIC_API_KEY="key",
        MODEL_FALLBACKS=["nvidia_nim/never", "anthropic/backup"],
    )
    app = create_test_app(settings, providers={"anthropic": provider(handle)})
    body = native_body()
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == body["model"]
    assert [item["model"] for item in sent] == ["primary", "backup"]
    assert sent[-1]["tools"] == body["tools"]


def test_compatibility_ingress_still_rejects_unknown_blocks():
    app = create_test_app(Settings())
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=native_body())
    assert response.status_code == 422


def test_leading_pings_allow_exhausted_primary_to_reach_native_fallback():
    from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
    from tests.providers.test_anthropic_messages_transport import Wire, _events, _sse

    calls = []
    wires = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        events = (
            [{"type": "error", "error": {"type": "overloaded_error"}}]
            if model == "primary"
            else _events()
        )
        wire = Wire([_sse({"type": "ping"}, *events)])
        wires.append(wire)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    app = create_test_app(
        Settings(
            MODEL="anthropic/primary",
            MODEL_FALLBACKS=["nvidia_nim/excluded", "anthropic/backup"],
        ),
        providers={"anthropic": provider(handle)},
    )
    body = native_body(True)
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    events = parse_sse_text(response.text)
    assert events[0].event == "message_start"
    assert events[0].data["message"]["model"] == body["model"]
    assert events[-1].event == "message_stop"
    assert calls == ["primary", "primary", "backup"]
    assert all(wire.closed for wire in wires)


def test_stream_failure_after_start_emits_one_error_without_fallback():
    from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
    from tests.providers.test_anthropic_messages_transport import _events, _sse

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                _events()[0],
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "late failure"},
                    "request_id": "anthropic-req-evidence",
                },
            ),
        )

    settings = Settings(
        MODEL="anthropic/primary",
        ANTHROPIC_API_KEY="key",
        MODEL_FALLBACKS=["anthropic/backup"],
    )
    app = create_test_app(settings, providers={"anthropic": provider(handle)})
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=native_body(True))
    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == ["message_start", "error"]
    assert "anthropic-req-evidence" in response.text
    assert len(requests) == 1


@pytest.mark.parametrize("view", ["claude", "claude-desktop", "messages"])
@pytest.mark.parametrize("thinking", [False, True, None])
def test_native_model_catalog_does_not_advertise_ignored_thinking_variant(
    view, thinking
):
    from free_claude_code.application.model_metadata import ProviderModelInfo
    from free_claude_code.core.gateway_model_ids import decode_gateway_model_id
    from tests.api.support import provider_manager_for_app

    app = create_test_app(Settings(MODEL="anthropic/selected", ANTHROPIC_API_KEY="key"))
    provider_manager_for_app(app).cache_model_infos(
        "anthropic", [ProviderModelInfo("selected", supports_thinking=thinking)]
    )
    with TestClient(app) as client:
        rows = client.get(f"/v1/models?view={view}").json()
    native = [row for row in rows["data"] if "anthropic" in row["display_name"]]
    assert len(native) == 1
    decoded = decode_gateway_model_id(native[0]["id"])
    assert decoded is None or not decoded.force_reasoning_off
    if view == "messages":
        assert rows["default_model_id"] == "anthropic/selected"


def test_native_proxy_auth_remains_separate_from_upstream_key():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"type": "message", "content": []})

    settings = Settings(
        MODEL="anthropic/selected",
        ANTHROPIC_API_KEY="api-key",
        ANTHROPIC_AUTH_TOKEN="proxy-secret",
        PROXY_AUTH_ENABLED=True,
    )
    app = create_test_app(settings, providers={"anthropic": provider(handle)})
    with TestClient(app) as client:
        denied = client.post(
            "/v1/messages", json=native_body(), headers={"x-api-key": "api-key"}
        )
        accepted = client.post(
            "/v1/messages",
            json=native_body(),
            headers={"Authorization": "Bearer proxy-secret"},
        )
    assert denied.status_code in {401, 403}
    assert accepted.status_code == 200
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer upstream-secret"


def test_compatibility_primary_keeps_conversion_when_anthropic_is_fallback():
    from tests.api.model_fallback_support import (
        ControlledFallbackProvider,
        execution_failure,
    )
    from tests.providers.test_anthropic_discovery import metadata
    from tests.providers.test_anthropic_messages_transport import _events, _sse

    sent = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=metadata("backup", True))
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(*_events())
        )

    primary = ControlledFallbackProvider(
        failure=execution_failure("primary unavailable")
    )
    settings = Settings(
        MODEL="nvidia_nim/primary",
        ANTHROPIC_API_KEY="key",
        MODEL_FALLBACKS=["anthropic/backup"],
        REASONING_POLICY="off",
    )
    p = provider(handle)
    app = create_test_app(
        settings, providers={"nvidia_nim": cast(BaseProvider, primary), "anthropic": p}
    )
    with (
        patch.object(
            p, "stream_native_messages", side_effect=AssertionError("contract changed")
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "client-alias",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32000,
            },
        )
    assert response.status_code == 200, response.text
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert sent[0]["max_tokens"] == 8192
    assert sent[0]["stream"] is True
    assert response.json()["model"] == "client-alias"
