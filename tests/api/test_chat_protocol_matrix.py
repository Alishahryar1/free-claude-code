"""The Chat endpoint uses each upstream through the real HTTP boundary."""

import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.test_custom_provider import completion, definition, provider


@pytest.mark.parametrize(
    "wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
@pytest.mark.parametrize("streaming", [False, True])
def test_chat_http_dispatch_preserves_controls_credentials_and_public_model(
    wire, streaming
):
    sent = []

    def reply(request):
        sent.append((json.loads(request.content), dict(request.headers)))
        module = httpx if wire == "anthropic_messages" else httpx2
        return module.Response(
            200, text=completion(wire), headers={"content-type": "text/event-stream"}
        )

    configured = definition(api_format=wire, api_keys=["upstream-key"])
    instance = provider(configured, openai_handler=reply, messages_handler=reply)
    app = create_test_app(
        Settings(
            MODEL=f"{configured.provider_id}/backend",
            custom_providers=(configured,),
            PROXY_AUTH_ENABLED=True,
            proxy_auth_token="ingress-key",
        ),
        providers={configured.provider_id: instance},
    )
    payload = {
        "model": "public",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "max_completion_tokens": 37,
        "stream": streaming,
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"authorization": "Bearer ingress-key"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-request-id"]
        if streaming:
            assert response.text.endswith("data: [DONE]\n\n")
            values = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert (
                "".join(
                    choice.get("delta", {}).get("content") or ""
                    for value in values
                    for choice in value.get("choices", [])
                )
                == "Hello"
            )
            assert all(value["model"] == "public" for value in values)
        else:
            assert response.json()["model"] == "public"
            assert response.json()["choices"][0]["message"]["content"] == "Hello"
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(sent) == 1
    body, headers = sent[0]
    assert body["model"] == "backend"
    assert body["temperature"] == 0
    limit = {
        "openai_chat": "max_completion_tokens",
        "openai_responses": "max_output_tokens",
        "anthropic_messages": "max_tokens",
    }[wire]
    assert body[limit] == 37
    assert "ingress-key" not in json.dumps(headers)
    assert headers[
        "x-api-key" if wire == "anthropic_messages" else "authorization"
    ].endswith("upstream-key")


def test_chat_validation_and_auth_errors_use_openai_shape():
    app = create_test_app(
        Settings(PROXY_AUTH_ENABLED=True, proxy_auth_token="ingress-key")
    )
    with TestClient(app) as client:
        unauthenticated = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauthenticated.status_code == 401
        assert "error" in unauthenticated.json()
        invalid = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
            headers={"authorization": "Bearer ingress-key"},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "changes",
    [
        {"n": 0},
        {"n": "two"},
        {"max_completion_tokens": "many"},
        {"stream_options": ["usage"]},
        {"tools": [1]},
        {"tools": [{"type": "function", "function": {}}]},
        {"tool_choice": "bogus"},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "id": "x", "function": {"arguments": "{}"}}
                    ],
                }
            ]
        },
        {"messages": [{"role": "assistant", "reasoning_details": ["bad"]}]},
    ],
)
def test_chat_malformed_fields_fail_before_provider_execution(changes):
    app = create_test_app(Settings())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public",
                "messages": [{"role": "user", "content": "hi"}],
                **changes,
            },
        )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
