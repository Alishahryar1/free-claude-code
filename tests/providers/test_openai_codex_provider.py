import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import httpx
import httpx2
import pytest
from openai import AuthenticationError

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_codex.auth import (
    OpenAIAccess,
    OpenAIAuthManager,
    OpenAIReconnectRequired,
)
from free_claude_code.providers.openai_codex.provider import OpenAICodexProvider
from tests.providers.support import immediate_admission, make_provider_config


class _FakeAuth(OpenAIAuthManager):
    def __init__(self) -> None:
        self.access_calls = 0
        self.recovery_calls = 0
        self.current_token = "access_1"

    async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
        self.access_calls += 1
        return OpenAIAccess(self.current_token, "account_1", False)

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        assert rejected_token == self.current_token
        self.recovery_calls += 1
        self.current_token = "access_2"
        return OpenAIAccess(self.current_token, "account_1", False)


class _CloseTrackingResponse(httpx2.Response):
    """HTTPX response that records close order and can fail during cleanup."""

    def __init__(
        self,
        status_code: int,
        *,
        close_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(status_code, **kwargs)
        self.close_calls = 0
        self._close_error = close_error

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()
        if self._close_error is not None:
            raise self._close_error


class _StaticAsyncBody(httpx2.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


class _CloseAwareAuth(_FakeAuth):
    def __init__(self, responses: list[_CloseTrackingResponse]) -> None:
        super().__init__()
        self._responses = responses

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        assert self._responses[-1].is_closed
        return await super().recover_unauthorized(rejected_token)


def _config() -> ProviderConfig:
    return make_provider_config(
        api_key="",
        base_url="https://chatgpt.com/backend-api/codex",
        rate_limit=100,
        rate_window=1,
        max_concurrency=2,
    )


def _admission(*, max_concurrency: int = 2) -> ProviderAdmissionController:
    return ProviderAdmissionController(
        provider_name="openai",
        rate_limit=100,
        rate_window=1,
        max_concurrency=max_concurrency,
        base_delay=0,
        max_delay=0,
        jitter=0,
    )


def _request() -> MessagesRequest:
    return MessagesRequest(
        model="gpt-test",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )


def _responses_request() -> OpenAIResponsesRequest:
    return OpenAIResponsesRequest.model_validate(
        {
            "model": "gpt-test",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "store": True,
            "previous_response_id": "resp_previous",
            "max_output_tokens": 1024,
            "metadata": {"source": "test"},
            "future_option": {"enabled": True},
        }
    )


def _client_control_request() -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "gpt-test",
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
        }
    )


def _sse(*events: tuple[str, dict[str, Any]]) -> str:
    return "".join(
        f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        for event_type, payload in events
    )


def _complete_stream(text: str) -> str:
    return _sse(
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
    )


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_ingress", [False, True])
async def test_completion_followed_by_ping_stays_successful(
    responses_ingress: bool,
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_complete_stream("hello") + _sse(("ping", {"type": "ping"})),
        )

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(),
        auth=_FakeAuth(),
        admission=immediate_admission(max_attempts=1),
        transport=pool,
    )
    stream = (
        provider.stream_responses(_responses_request())
        if responses_ingress
        else provider.stream_messages(_request())
    )
    events = parse_sse_text(await _collect(stream))
    await provider.cleanup()
    terminal = "response.completed" if responses_ingress else "message_stop"
    assert [event.event for event in events].count(terminal) == 1
    assert events[-1].event == terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_ingress", [False, True])
async def test_streamed_authentication_error_has_authentication_status(
    responses_ingress: bool,
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                (
                    "error",
                    {"type": "error", "code": "invalid_api_key", "message": "expired"},
                ),
            ),
        )

    auth = _FakeAuth()
    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(),
        auth=auth,
        admission=immediate_admission(max_attempts=1),
        transport=pool,
    )
    stream = (
        provider.stream_responses(_responses_request())
        if responses_ingress
        else provider.stream_messages(_request())
    )
    with pytest.raises(ExecutionFailure) as failure:
        await _collect(stream)
    await provider.cleanup()
    assert failure.value.status_code == 401
    assert failure.value.kind is FailureKind.AUTHENTICATION
    assert auth.recovery_calls == 0


@pytest.mark.asyncio
async def test_provider_uses_subscription_headers_and_visible_model_catalog() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx2.Response(
                200,
                json={
                    "models": [
                        {
                            "slug": "gpt-visible",
                            "visibility": "list",
                            "supported_in_api": False,
                            "supported_reasoning_levels": [{"effort": "high"}],
                            "input_modalities": ["text", "image"],
                            "context_window": 272000,
                        },
                        {
                            "slug": "gpt-no-reasoning",
                            "visibility": "list",
                            "supported_reasoning_levels": [],
                        },
                        {
                            "slug": "gpt-malformed-reasoning",
                            "visibility": "list",
                            "supported_reasoning_levels": [None],
                        },
                        {"slug": "gpt-hidden", "visibility": "hide"},
                    ]
                },
                request=request,
            )
        assert request.url.path.endswith("/responses")
        return httpx2.Response(
            200,
            content=_complete_stream("hello").encode(),
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )

    infos = await provider.list_model_infos()
    body = await _collect(
        provider.stream_messages(
            _request(),
            input_tokens=3,
            request_id="req_test",
            response_model="claude-opus-4",
            reasoning=ReasoningPolicy.provider_default(),
        )
    )

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "gpt-visible",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=272000,
            ),
            ProviderModelInfo("gpt-no-reasoning", supports_thinking=False),
            ProviderModelInfo("gpt-malformed-reasoning"),
        }
    )
    response_request = requests[-1]
    assert response_request.headers["authorization"] == "Bearer access_1"
    assert response_request.headers["chatgpt-account-id"] == "account_1"
    assert response_request.headers["originator"] == "codex_cli_rs"
    assert response_request.headers["accept"] == "text/event-stream"
    assert response_request.headers["session_id"]
    payload = json.loads(response_request.content)
    assert payload["model"] == "gpt-test"
    assert payload["store"] is False
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "hello"
    await provider.cleanup()


@pytest.mark.asyncio
async def test_generation_close_failure_preserves_success_and_permit() -> None:
    responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        response = _CloseTrackingResponse(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
            close_error=RuntimeError("cleanup failed"),
        )
        responses.append(response)
        return response

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(),
        auth=_FakeAuth(),
        admission=_admission(max_concurrency=1),
        transport=pool,
    )

    first = await asyncio.wait_for(
        _collect(provider.stream_messages(_request(), response_model="claude-opus-4")),
        timeout=1,
    )
    second = await asyncio.wait_for(
        _collect(provider.stream_messages(_request(), response_model="claude-opus-4")),
        timeout=1,
    )

    assert text_content(parse_sse_text(first)) == "hello"
    assert text_content(parse_sse_text(second)) == "hello"
    assert all(response.is_closed for response in responses)
    await provider.cleanup()


@pytest.mark.asyncio
async def test_generation_close_failure_preserves_provider_failure() -> None:
    responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        response = _CloseTrackingResponse(
            400,
            json={"error": {"message": "original provider failure"}},
            request=request,
            close_error=RuntimeError("cleanup failed"),
        )
        responses.append(response)
        return response

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_close_failure",
                response_model="claude-opus-4",
            )
        )

    assert exc_info.value.status_code == 400
    assert "original provider failure" in exc_info.value.message
    assert "cleanup failed" not in exc_info.value.message
    assert all(response.is_closed for response in responses)
    await provider.cleanup()


@pytest.mark.asyncio
async def test_provider_accepts_claude_client_controls_before_upstream_io() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )
    request = _client_control_request()
    original_request = request.model_dump()
    reasoning = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)

    stream = provider.stream_messages(
        request,
        request_id="req_client_controls",
        response_model="claude-opus-4",
        reasoning=reasoning,
    )
    assert requests == []
    body = await _collect(stream)

    assert len(requests) == 1
    upstream = requests[0]
    assert upstream.url.path.endswith("/responses")
    payload = json.loads(upstream.content)
    assert payload["model"] == "gpt-test"
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "max_output_tokens" not in payload
    assert "metadata" not in payload
    assert "context_management" not in payload
    assert "output_config" not in payload
    assert request.model_dump() == original_request
    assert_anthropic_stream_contract(parse_sse_text(body))
    await provider.cleanup()


@pytest.mark.asyncio
async def test_provider_relays_native_responses_with_private_field_policy() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )
    request = _responses_request()
    original = request.model_dump()

    stream = provider.stream_responses(
        request,
        request_id="req_native_responses",
        response_model="openai/gpt-test",
    )
    assert requests == []
    body = await _collect(stream)

    assert len(requests) == 1
    upstream = json.loads(requests[0].content)
    assert upstream["model"] == "gpt-test"
    assert upstream["input"] == [{"role": "user", "content": "hello"}]
    assert upstream["stream"] is True
    assert upstream["store"] is False
    assert upstream["future_option"] == {"enabled": True}
    assert "previous_response_id" not in upstream
    assert "max_output_tokens" not in upstream
    assert "metadata" not in upstream
    assert request.model_dump() == original

    events = parse_sse_text(body)
    assert [event.event for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0].data["response"]["id"] == "resp_1"
    assert events[0].data["response"]["model"] == "openai/gpt-test"
    assert events[-1].data["response"]["id"] == "resp_1"
    assert events[-1].data["response"]["model"] == "openai/gpt-test"
    await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "error_path"),
    [
        (
            "context_management",
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 2},
                    }
                ]
            },
            "context_management",
        ),
        (
            "output_config",
            {"format": {"type": "json_schema"}},
            "output_config.format",
        ),
    ],
)
async def test_provider_validation_rejects_unrepresentable_client_controls(
    field: str,
    value: object,
    error_path: str,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(500, request=request)

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }

    with pytest.raises(InvalidRequestError, match=error_path):
        provider.stream_messages(MessagesRequest.model_validate(payload))

    assert requests == []
    await provider.cleanup()


@pytest.mark.asyncio
async def test_provider_round_trips_portable_tool_name_alias() -> None:
    original = "mcp__codex_provider__" + "x" * 70
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "use the tool"}],
            "tools": [
                {
                    "name": original,
                    "description": "Long tool",
                    "input_schema": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "tool", "name": original},
        }
    )
    payloads: list[dict[str, Any]] = []

    def handler(http_request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(http_request.content)
        payloads.append(payload)
        alias = payload["tools"][0]["name"]
        body = _sse(
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": alias,
                    },
                },
            ),
            (
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": alias,
                        "arguments": "{}",
                    },
                },
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
        )
        return httpx2.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
            request=http_request,
        )

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )

    body = await _collect(provider.stream_messages(request))

    payload = payloads[0]
    alias = payload["tools"][0]["name"]
    assert alias != original
    assert len(alias) <= 64
    assert payload["tool_choice"] == {"type": "function", "name": alias}
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    tool_start = next(
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
    )
    assert tool_start["name"] == original
    assert alias not in body
    await provider.cleanup()


@pytest.mark.asyncio
async def test_non_streaming_success_cannot_complete_generation() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            json={"message": "upstream contract changed"},
            headers={"content-type": "application/json"},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_non_stream",
                response_model="claude-opus-4",
            )
        )

    assert attempts > 1
    assert exc_info.value.status_code == 502
    assert "Request ID: req_non_stream" in exc_info.value.message
    await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_ingress", [False, True])
@pytest.mark.parametrize("rejection", ["http_401", "http_403", "flat", "wrapped"])
async def test_unauthorized_response_forces_one_auth_refresh(
    responses_ingress: bool,
    rejection: str,
) -> None:
    authorizations: list[str] = []
    sessions: list[str] = []
    unauthorized_responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        authorizations.append(request.headers["authorization"])
        sessions.append(request.headers["session_id"])
        if len(authorizations) == 1:
            error = {"code": "invalid_api_key", "message": "expired"}
            if rejection.startswith("http_"):
                status = int(rejection.removeprefix("http_"))
                body = json.dumps({"error": error})
            else:
                status = 200
                payload = (
                    {"error": error}
                    if rejection == "wrapped"
                    else {"type": "error", **error}
                )
                body = f"data: {json.dumps(payload)}\n\n"
            response = _CloseTrackingResponse(
                status,
                stream=_StaticAsyncBody(body.encode()),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
            unauthorized_responses.append(response)
            return response
        return httpx2.Response(
            200,
            text=_complete_stream("recovered"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    auth = _CloseAwareAuth(unauthorized_responses)
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )

    body = await _collect(
        provider.stream_responses(_responses_request(), request_id="req_auth")
        if responses_ingress
        else provider.stream_messages(_request(), request_id="req_auth")
    )

    assert authorizations == ["Bearer access_1", "Bearer access_2"]
    assert auth.recovery_calls == 1
    assert sessions[0] == sessions[1]
    assert unauthorized_responses[0].is_closed
    if responses_ingress:
        assert [event.event for event in parse_sse_text(body)].count(
            "response.completed"
        ) == 1
    else:
        assert text_content(parse_sse_text(body)) == "recovered"
    await provider.cleanup()


@pytest.mark.asyncio
async def test_concurrent_credentials_keep_request_identity_and_pool_ownership() -> (
    None
):
    class ConcurrentAuth(_FakeAuth):
        async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
            self.access_calls += 1
            return OpenAIAccess(
                f"access_{self.access_calls}", f"account_{self.access_calls}", False
            )

        async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
            assert rejected_token == "access_1"
            self.recovery_calls += 1
            return OpenAIAccess("access_3", "account_3", True)

        async def close(self) -> None:
            raise AssertionError("Provider must not close the runtime-owned account")

    class Pool(httpx2.MockTransport):
        closes = 0

        async def aclose(self) -> None:
            self.closes += 1
            await super().aclose()

    second_started = asyncio.Event()
    requests: dict[str, httpx2.Request] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        token = request.headers["authorization"]
        requests[token] = request
        if token == "Bearer access_1":
            await second_started.wait()
            return httpx2.Response(401, json={"error": {"message": "expired"}})
        if token == "Bearer access_2":
            second_started.set()
        return httpx2.Response(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
        )

    auth = ConcurrentAuth()
    pool = Pool(handler)
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                _collect(provider.stream_messages(_request())),
                _collect(provider.stream_messages(_request())),
            ),
            timeout=3,
        )
        assert all(
            text_content(parse_sse_text(result)) == "hello" for result in results
        )
        first, second, refreshed = (
            requests[f"Bearer access_{number}"] for number in (1, 2, 3)
        )
        assert first.headers["session_id"] == refreshed.headers["session_id"]
        assert first.headers["session_id"] != second.headers["session_id"]
        for number, request in enumerate((first, second, refreshed), start=1):
            assert request.headers["chatgpt-account-id"] == f"account_{number}"
        assert "x-openai-fedramp" not in first.headers
        assert "x-openai-fedramp" not in second.headers
        assert refreshed.headers["x-openai-fedramp"] == "true"
        assert auth.recovery_calls == 1
        assert pool.closes == 0
    finally:
        await provider.cleanup()
    assert pool.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_ingress", [False, True])
@pytest.mark.parametrize("code", [None, "token_expired"])
@pytest.mark.parametrize(
    ("error_type", "status", "kind"),
    [
        ("authentication_error", 401, FailureKind.AUTHENTICATION),
        ("permission_error", 403, FailureKind.PERMISSION),
    ],
)
async def test_nested_authentication_error_keeps_its_type(
    responses_ingress: bool,
    code: str | None,
    error_type: str,
    status: int,
    kind: FailureKind,
) -> None:
    authorizations: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        authorizations.append(request.headers["authorization"])
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                (
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": "resp_failed",
                            "status": "failed",
                            "error": {
                                "code": code,
                                "type": error_type,
                                "message": "expired",
                            },
                        },
                    },
                )
            ),
        )

    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(),
        auth=auth,
        admission=immediate_admission(max_attempts=2),
        transport=httpx2.MockTransport(handler),
    )
    try:
        stream = (
            provider.stream_responses(_responses_request())
            if responses_ingress
            else provider.stream_messages(_request())
        )
        with pytest.raises(ExecutionFailure) as failure:
            await _collect(stream)
        assert failure.value.status_code == status
        assert failure.value.kind is kind
        assert auth.recovery_calls == 1
        assert authorizations == ["Bearer access_1", "Bearer access_2"]
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["messages", "responses", "catalog"])
async def test_disconnected_account_never_opens_or_refreshes_a_request(
    operation: str,
) -> None:
    class DisconnectedAuth(_FakeAuth):
        async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
            self.access_calls += 1
            raise OpenAIReconnectRequired("Reconnect in Admin.")

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("Disconnected accounts must not send a provider request")

    auth = DisconnectedAuth()
    provider = OpenAICodexProvider(
        _config(),
        auth=auth,
        admission=_admission(),
        transport=httpx2.MockTransport(handler),
    )
    try:
        with pytest.raises(ExecutionFailure, match="Reconnect in Admin") as failure:
            if operation == "catalog":
                await provider.list_model_infos()
            elif operation == "responses":
                await _collect(provider.stream_responses(_responses_request()))
            else:
                await _collect(provider.stream_messages(_request()))
        assert failure.value.kind is FailureKind.AUTHENTICATION
        assert auth.access_calls == 1
        assert auth.recovery_calls == 0
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_model_discovery_re_request_is_a_second_admitted_attempt() -> None:
    authorizations: list[str] = []
    unauthorized_responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        authorizations.append(request.headers["authorization"])
        if len(authorizations) == 1:
            response = _CloseTrackingResponse(401, request=request)
            unauthorized_responses.append(response)
            return response
        return httpx2.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-visible",
                        "visibility": "list",
                        "supported_in_api": False,
                        "supported_reasoning_levels": [{"effort": "high"}],
                    }
                ]
            },
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    auth = _CloseAwareAuth(unauthorized_responses)
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        infos = await provider.list_model_infos()

    assert {info.model_id for info in infos} == {"gpt-visible"}
    assert authorizations == ["Bearer access_1", "Bearer access_2"]
    assert auth.recovery_calls == 1
    assert unauthorized_responses[0].is_closed
    attempt_rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event", "").startswith("provider.attempt.")
    ]
    assert [row["event"] for row in attempt_rows] == [
        "provider.attempt.started",
        "provider.attempt.resolved",
        "provider.attempt.started",
        "provider.attempt.resolved",
    ]
    assert [row["attempt"] for row in attempt_rows] == [1, 1, 2, 2]
    assert [row.get("outcome") for row in attempt_rows] == [
        None,
        "corrected",
        None,
        "accepted",
    ]
    assert len({row["execution_id"] for row in attempt_rows}) == 1
    await provider.cleanup()


@pytest.mark.asyncio
async def test_auth_refresh_failure_does_not_repeat_rejected_provider_call() -> None:
    class FailingRecoveryAuth(_FakeAuth):
        async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
            assert rejected_token == self.current_token
            self.recovery_calls += 1
            raise httpx.ReadError(
                "refresh interrupted",
                request=httpx.Request("POST", "https://auth.openai.com/oauth/token"),
            )

    authorizations: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        authorization = request.headers["authorization"]
        authorizations.append(authorization)
        if authorization == "Bearer access_1":
            return httpx2.Response(401, request=request)
        return httpx2.Response(
            200,
            text=_complete_stream("recovered"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    auth = FailingRecoveryAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )

    with pytest.raises(ExecutionFailure, match="refresh interrupted") as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_auth_transient",
                response_model="claude-opus-4",
            )
        )

    assert isinstance(exc_info.value.__cause__, httpx.ReadError)
    assert authorizations == ["Bearer access_1"]
    assert auth.recovery_calls == 1
    await provider.cleanup()


@pytest.mark.asyncio
async def test_second_unauthorized_response_is_terminal_without_refresh_loop() -> None:
    authorizations: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        authorizations.append(request.headers["authorization"])
        return httpx2.Response(
            401,
            json={"error": {"message": "still expired"}},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_auth_terminal",
                response_model="claude-opus-4",
            )
        )

    assert exc_info.value.status_code == 401
    assert authorizations == ["Bearer access_1", "Bearer access_2"]
    assert auth.recovery_calls == 1
    await provider.cleanup()


@pytest.mark.asyncio
async def test_final_attempt_unauthorized_preserves_the_provider_401() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        status = 503 if attempts < 5 else 401
        return httpx2.Response(
            status,
            json={"error": {"message": f"attempt {attempts}"}},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), transport=pool
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_final_401",
                response_model="claude-opus-4",
            )
        )

    assert attempts == 5
    assert auth.recovery_calls == 0
    assert exc_info.value.status_code == 401
    assert "Request ID: req_final_401" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, AuthenticationError)
    assert exc_info.value.__cause__.response.status_code == 401
    await provider.cleanup()


@pytest.mark.asyncio
async def test_stream_failure_redacts_credentials_from_customer_diagnostic() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            text=_sse(
                (
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {
                                "code": "invalid_request",
                                "message": (
                                    "Authorization: Bearer "
                                    "sk-this-must-never-be-returned"
                                ),
                            }
                        },
                    },
                )
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    pool = httpx2.MockTransport(handler)
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), transport=pool
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_redaction",
                response_model="claude-opus-4",
            )
        )

    assert "sk-this-must-never-be-returned" not in exc_info.value.message
    assert "Authorization: <redacted>" in exc_info.value.message
    await provider.cleanup()
