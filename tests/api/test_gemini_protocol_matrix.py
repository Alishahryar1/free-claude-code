"""Real FCC HTTP requests dispatch directly through each custom transport."""

import base64
import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.settings import Settings
from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.runtime.config import build_custom_provider_config
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import create_test_app, provider_manager_for_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.providers.support import immediate_admission
from tests.providers.test_custom_provider import completion, definition, provider
from tests.providers.test_gemini_transport import chat_response


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
def test_caller_native_signature_is_rejected_before_inference(wire, streaming):
    sent = []

    def reply(request):
        sent.append(request)
        raise AssertionError("Unsupported history must be rejected before inference")

    configured = definition(api_format=wire)
    instance = provider(configured, openai_handler=reply, messages_handler=reply)
    app = create_test_app(
        Settings(
            MODEL=f"{configured.provider_id}/model", custom_providers=(configured,)
        ),
        providers={configured.provider_id: instance},
    )
    action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    with TestClient(app) as client:
        response = client.post(
            f"/v1beta/models/public:{action}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "hello"}]},
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": "previous answer",
                                "thoughtSignature": base64.b64encode(
                                    b"native-google-state"
                                ).decode(),
                            }
                        ],
                    },
                    {"role": "user", "parts": [{"text": "continue"}]},
                ]
            },
        )
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert response.status_code == 400, response.text
    assert "Caller-native Gemini signatures" in response.json()["error"]["message"]
    assert not sent


@pytest.mark.parametrize(
    "wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
@pytest.mark.parametrize("streaming", [False, True])
def test_full_http_to_upstream_matrix_keeps_route_controls_and_credentials(
    wire, streaming
):
    sent = []

    def reply(request):
        sent.append((json.loads(request.content), dict(request.headers)))
        response = httpx.Response if wire == "anthropic_messages" else httpx2.Response
        return response(
            200, text=completion(wire), headers={"content-type": "text/event-stream"}
        )

    configured = definition(api_format=wire, api_keys=["synthetic-upstream"])
    instance = provider(configured, openai_handler=reply, messages_handler=reply)
    settings = Settings(
        MODEL=f"{configured.provider_id}/default-backend",
        custom_providers=(configured,),
        PROXY_AUTH_ENABLED=True,
        proxy_auth_token="synthetic-ingress",
    )
    app = create_test_app(settings, providers={configured.provider_id: instance})
    public_model = f"{configured.provider_id}/selected-backend"
    suffix = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    payload = {
        "contents": [{"parts": [{"text": "hello"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 37},
    }
    with TestClient(app) as client:
        response = client.post(
            f"/v1beta/models/{public_model}:{suffix}",
            json=payload,
            headers={"x-goog-api-key": "synthetic-ingress"},
        )
        assert response.status_code == 200, response.text
        if streaming:
            values = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert values[-1]["candidates"][0]["finishReason"] == "STOP"
            assert all(item["modelVersion"] == public_model for item in values)
        else:
            value = response.json()
            assert value["modelVersion"] == public_model
            assert value["candidates"][0]["content"]["parts"] == [{"text": "Hello"}]
        assert provider_manager_for_app(app)._current.active_leases == 0
        # An auxiliary/unknown model uses the same configured default route.
        assert (
            client.post(
                "/v1beta/models/gemini-auxiliary:generateContent",
                json=payload,
                headers={"x-goog-api-key": "synthetic-ingress"},
            ).status_code
            == 200
        )
    assert [body["model"] for body, _ in sent] == [
        "selected-backend",
        "default-backend",
    ]
    for body, headers in sent:
        assert body["temperature"] == 0
        assert (
            body["max_output_tokens" if wire == "openai_responses" else "max_tokens"]
            == 37
        )
        assert "synthetic-ingress" not in json.dumps(headers)
        assert "x-goog-api-key" not in headers
        assert headers[
            "x-api-key" if wire == "anthropic_messages" else "authorization"
        ].endswith("synthetic-upstream")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "first_wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
def test_gemini_http_fallback_keeps_one_response_across_egresses(
    first_wire, streaming, structured, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    sent = []

    def reply(request):
        body = json.loads(request.content)
        sent.append(body)
        module = (
            httpx
            if first_wire == "anthropic_messages" and body["model"] == "first"
            else httpx2
        )
        if body["model"] == "second":
            return module.Response(
                200,
                text=completion("openai_responses"),
                headers={"content-type": "text/event-stream"},
            )
        protocol = {
            "openai_chat": "chat",
            "openai_responses": "responses",
            "anthropic_messages": "messages",
        }[first_wire]
        events = text_events(
            protocol, '{"value":' if structured else "Before ", complete=False
        )
        raw = "".join(
            (f"event: {event['type']}\n" if protocol == "messages" else "")
            + f"data: {json.dumps(event)}\n\n"
            for event in events
        )
        return module.Response(
            200, text=raw, headers={"content-type": "text/event-stream"}
        )

    configs = [
        definition(api_format=first_wire),
        definition(
            provider_id="custom_" + "b" * 32,
            display_name="Backup",
            api_format="openai_responses",
        ),
    ]
    instances: dict[str, BaseProvider] = {
        config.provider_id: CustomProvider(
            build_custom_provider_config(config, Settings()),
            definition=config,
            admission=immediate_admission(max_attempts=1),
            openai_transport=httpx2.MockTransport(reply),
            messages_transport=httpx.MockTransport(reply),
        )
        for config in configs
    }
    app = create_test_app(
        Settings(
            MODEL=f"{configs[0].provider_id}/first",
            MODEL_FALLBACKS=[ModelFallback(model=f"{configs[1].provider_id}/second")],
            custom_providers=tuple(configs),
        ),
        providers=instances,
    )
    payload = {"contents": [{"parts": [{"text": "hello"}]}]}
    if structured:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": {"type": "object"},
        }
    suffix = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    with TestClient(app) as client:
        response = client.post(f"/v1beta/models/public:{suffix}", json=payload)
        assert provider_manager_for_app(app)._current.active_leases == 0
    if structured:
        assert [body["model"] for body in sent] == ["first"]
        if streaming:
            assert '"error"' in response.text and '"finishReason"' not in response.text
        else:
            assert response.status_code >= 400
    else:
        assert response.status_code == 200, response.text
        assert [body["model"] for body in sent] == ["first", "second"]
        if streaming:
            aggregate = GeminiResponseAccumulator()
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    aggregate.feed(json.loads(line[6:]))
            value = aggregate.result()
        else:
            value = response.json()
        assert value["candidates"][0]["content"]["parts"] == [{"text": "Before Hello"}]
        assert value["modelVersion"] == "public"


@pytest.mark.parametrize("streaming", [False, True])
def test_public_fallback_keeps_completed_reasoning_from_original_provider(streaming):
    sent = []

    def reply(request):
        body = json.loads(request.content)
        sent.append(body)
        if body["model"] == "first":
            return chat_response(
                [
                    {"reasoning_content": "original hidden thought"},
                    {"content": "Before " * 10000},
                ],
                finish=None,
            )
        return httpx2.Response(
            200,
            text=completion("openai_responses"),
            headers={"content-type": "text/event-stream"},
        )

    configs = [
        definition(api_format="openai_chat"),
        definition(
            provider_id="custom_" + "b" * 32,
            display_name="Backup",
            api_format="openai_responses",
        ),
    ]
    instances: dict[str, BaseProvider] = {
        config.provider_id: CustomProvider(
            build_custom_provider_config(config, Settings()),
            definition=config,
            admission=immediate_admission(max_attempts=1),
            openai_transport=httpx2.MockTransport(reply),
        )
        for config in configs
    }
    app = create_test_app(
        Settings(
            MODEL=f"{configs[0].provider_id}/first",
            MODEL_FALLBACKS=[ModelFallback(model=f"{configs[1].provider_id}/second")],
            custom_providers=tuple(configs),
        ),
        providers=instances,
    )
    action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    with TestClient(app) as client:
        response = client.post(
            f"/v1beta/models/public:{action}",
            json={"contents": [{"parts": [{"text": "hello"}]}]},
        )
    assert response.status_code == 200, response.text
    assert [body["model"] for body in sent] == ["first", "second"]
    assert "original hidden thought" in json.dumps(sent[1]["input"])
    if streaming:
        aggregate = GeminiResponseAccumulator()
        for line in response.text.splitlines():
            if line.startswith("data: "):
                aggregate.feed(json.loads(line[6:]))
        value = aggregate.result()
    else:
        value = response.json()
    signatures = [
        part["thoughtSignature"]
        for part in value["candidates"][0]["content"]["parts"]
        if "thoughtSignature" in part
    ]
    assert len(signatures) == 1
    bundle = decode_bundle(signatures[0])
    assert bundle is not None
    assert len(bundle.records) == 1
    assert (
        bundle.records[0].origin.protocol == "chat"
        and bundle.records[0].origin.model == "first"
    )
    assert bundle.records[0].native == {"reasoning_content": "original hidden thought"}
