"""Real FCC HTTP requests dispatch directly through each custom transport."""

import base64
import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.settings import Settings
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.runtime.config import build_custom_provider_config
from tests.api.support import create_test_app, provider_manager_for_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.providers.support import immediate_admission
from tests.providers.test_custom_provider import completion, definition, provider
from tests.providers.test_gemini_transport import chat_response
from tests.wire_payload import (
    decode_object,
    object_array,
    object_payload,
    response_object,
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
def test_caller_native_signature_is_rejected_before_inference(
    wire: str, streaming: bool
) -> None:
    sent = []

    def reply(request: httpx.Request | httpx2.Request):
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
    wire: str, streaming: bool
) -> None:
    sent = []

    def reply(request: httpx.Request | httpx2.Request):
        sent.append((decode_object(request.content), dict(request.headers)))
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
                load_json_value(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert (
                object_array(object_payload(values[-1])["candidates"])[0][
                    "finishReason"
                ]
                == "STOP"
            )
            assert all(
                object_payload(item)["modelVersion"] == public_model for item in values
            )
        else:
            value = response_object(response)
            assert value["modelVersion"] == public_model
            assert object_payload(object_array(value["candidates"])[0]["content"])[
                "parts"
            ] == [{"text": "Hello"}]
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
    first_wire: str, streaming: bool, structured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = []

    def response_text(content: bytes) -> str:
        body = decode_object(content)
        sent.append(body)
        if body["model"] == "second":
            return completion("openai_responses")
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
        return raw

    def messages_reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=response_text(request.content),
            headers={"content-type": "text/event-stream"},
        )

    def openai_reply(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            text=response_text(request.content),
            headers={"content-type": "text/event-stream"},
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
            openai_transport=httpx2.MockTransport(openai_reply),
            messages_transport=httpx.MockTransport(messages_reply),
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
    payload: JsonObject = {"contents": [{"parts": [{"text": "hello"}]}]}
    if structured:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": {"type": "object"},
        }
    suffix = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    with TestClient(app) as client:
        response = client.post(f"/v1beta/models/public:{suffix}", json=payload)
        assert provider_manager_for_app(app)._current.active_leases == 0
    if structured and streaming:
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
                    aggregate.feed(decode_object(line[6:]))
            value = aggregate.result()
        else:
            value = response_object(response)
        assert object_payload(object_array(value["candidates"])[0]["content"])[
            "parts"
        ] == [{"text": "Before Hello" if streaming else "Hello"}]
        assert value["modelVersion"] == "public"


@pytest.mark.parametrize("streaming", [False, True])
def test_public_fallback_omits_hidden_unsigned_history(streaming: bool) -> None:
    sent = []

    def reply(request: httpx2.Request):
        body = decode_object(request.content)
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
    assert "original hidden thought" not in json.dumps(sent[1]["input"])
    if streaming:
        aggregate = GeminiResponseAccumulator()
        for line in response.text.splitlines():
            if line.startswith("data: "):
                aggregate.feed(decode_object(line[6:]))
        value = aggregate.result()
    else:
        value = response_object(response)
    signatures = [
        part["thoughtSignature"]
        for part in object_array(
            object_payload(object_array(value["candidates"])[0]["content"])["parts"]
        )
        if "thoughtSignature" in part
    ]
    assert not signatures
