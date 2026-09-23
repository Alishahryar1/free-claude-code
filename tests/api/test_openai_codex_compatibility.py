import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.openai_codex.auth import (
    OpenAIAccess,
    OpenAIAuthManager,
)
from free_claude_code.providers.openai_codex.provider import OpenAICodexProvider
from tests.api.support import create_test_app
from tests.providers.support import make_provider_config


class _FakeOpenAIAuth(OpenAIAuthManager):
    async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
        return OpenAIAccess("access", "account", False)

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        raise AssertionError("The compatibility request must not require auth recovery")


def _complete_stream(text: str) -> str:
    events = [
        (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        ),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": text},
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ),
    ]
    return "".join(
        f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        for event_type, payload in events
    )


def _openai_provider(
    upstream_requests: list[httpx2.Request],
    *,
    response_text: str,
) -> OpenAICodexProvider:
    def handler(request: httpx2.Request) -> httpx2.Response:
        upstream_requests.append(request)
        return httpx2.Response(
            200,
            content=_complete_stream(response_text).encode(),
            request=request,
        )

    provider = OpenAICodexProvider(
        make_provider_config(
            api_key="",
            base_url="https://chatgpt.com/backend-api/codex",
            rate_limit=100,
            rate_window=1,
            max_concurrency=2,
        ),
        auth=_FakeOpenAIAuth(),
        admission=ProviderAdmissionController(
            provider_name="openai",
            rate_limit=100,
            rate_window=1,
            max_concurrency=2,
            base_delay=0,
            max_delay=0,
            jitter=0,
        ),
        transport=httpx2.MockTransport(handler),
    )
    return provider


@pytest.mark.asyncio
async def test_messages_accepts_current_claude_controls_for_openai_provider() -> None:
    upstream_requests: list[httpx2.Request] = []
    provider = _openai_provider(
        upstream_requests,
        response_text="hello",
    )
    app = create_test_app(providers={"openai": provider})

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "openai/gpt-test",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "hello"}],
                    "metadata": {"user_id": "example-user"},
                    "thinking": {"type": "adaptive", "display": "omitted"},
                    "context_management": {
                        "edits": [
                            {
                                "type": "clear_thinking_20251015",
                                "keep": "all",
                            }
                        ]
                    },
                    "output_config": {"effort": "high"},
                    "stream": True,
                },
            )
    finally:
        await provider.cleanup()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse_text(response.text)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "hello"
    assert len(upstream_requests) == 1
    payload = json.loads(upstream_requests[0].content)
    assert payload["model"] == "gpt-test"
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "max_output_tokens" not in payload
    assert "metadata" not in payload
    assert "context_management" not in payload
    assert "output_config" not in payload


@pytest.mark.asyncio
async def test_messages_auto_mode_classifier_composes_with_openai_provider() -> None:
    upstream_requests: list[httpx2.Request] = []
    provider = _openai_provider(
        upstream_requests,
        response_text="<severity>0</severity>",
    )
    app = create_test_app(providers={"openai": provider})

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages?beta=true",
                json={
                    "model": "openai/gpt-test",
                    "max_tokens": 64,
                    "system": (
                        "Classify the command's safety. "
                        "Output <severity>N</severity> where N is the numeric "
                        "severity."
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "<transcript>\nUser: inspect a file\n"
                                "Bash: cat the file\n</transcript>"
                            ),
                        }
                    ],
                    "stop_sequences": ["</severity>"],
                },
            )
    finally:
        await provider.cleanup()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "<severity>0</severity>"}]
    assert body["stop_reason"] == "end_turn"
    assert len(upstream_requests) == 1
    payload = json.loads(upstream_requests[0].content)
    assert payload["model"] == "gpt-test"
    assert payload["reasoning"] == {"effort": "none"}
    assert "stop" not in payload
    assert "stop_sequences" not in payload


@pytest.mark.asyncio
async def test_messages_non_classifier_stop_remains_lossy_for_openai_provider() -> None:
    upstream_requests: list[httpx2.Request] = []
    provider = _openai_provider(
        upstream_requests,
        response_text="unreachable",
    )
    app = create_test_app(providers={"openai": provider})

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "openai/gpt-test",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stop_sequences": ["caller-owned-stop"],
                },
            )
    finally:
        await provider.cleanup()

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "cannot represent" in body["error"]["message"]
    assert "stop_sequences" in body["error"]["message"]
    assert upstream_requests == []
