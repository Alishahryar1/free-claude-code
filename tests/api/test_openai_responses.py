import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import DEFAULT_MODEL
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission, make_provider_config

_PUBLIC_MODEL = "nvidia_nim/test-model"
_UPSTREAM_MODEL = "test-model"
_RESPONSE_ID = "resp_test"


class FakeProvider:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.preflight_responses = MagicMock()
        self.requests: list[OpenAIResponsesRequest] = []
        self.stream_kwargs: list[dict[str, object]] = []

    async def stream_responses(
        self,
        request_data: OpenAIResponsesRequest,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        self.requests.append(request_data)
        self.stream_kwargs.append(kwargs)
        for chunk in self.chunks:
            yield chunk


class PreStartFailingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])

    async def stream_responses(
        self,
        request_data: OpenAIResponsesRequest,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        self.requests.append(request_data)
        self.stream_kwargs.append(kwargs)
        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="upstream is busy",
            retryable=True,
        )
        yield "unreachable"


@pytest.fixture
def responses_client():
    provider = FakeProvider(_responses_text_stream("Hello from provider"))
    app = create_test_app(Settings())
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        yield client, provider


def test_responses_probe_endpoints_return_204(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, _provider = responses_client

    assert client.head("/v1/responses").status_code == 204
    assert client.options("/v1/responses").status_code == 204


@pytest.mark.parametrize("prefix", ["", "anthropic/", "claude-3-freecc-no-thinking/"])
def test_retired_response_id_uses_default_provider(responses_client, prefix):
    client, provider = responses_client
    original = f"{prefix}github_models/vendor/opus"
    response = client.post("/v1/responses", json={"model": original, "input": "hello"})
    assert response.status_code == 200
    assert "response.completed" in response.text
    assert provider.requests[0].model == DEFAULT_MODEL.split("/", 1)[1]
    assert provider.stream_kwargs[0]["response_model"] == original
    if prefix == "claude-3-freecc-no-thinking/":
        assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy.off()


def test_create_response_stream_routes_native_request_through_provider(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client

    response = client.post(
        "/v1/responses",
        json={
            "model": _PUBLIC_MODEL,
            "input": "Hello",
            "max_output_tokens": 32,
        },
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["x-request-id"] == response.headers["request-id"]
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[-1].data["response"]["output"][0]["content"][0]["text"] == (
        "Hello from provider"
    )
    assert provider.preflight_responses.called
    routed = provider.requests[0]
    assert routed.model == _UPSTREAM_MODEL
    assert routed.input == "Hello"
    assert routed.max_output_tokens == 32
    assert provider.stream_kwargs[0]["request_id"] == response.headers["request-id"]


def test_create_response_stream_preserves_output_limit_as_incomplete() -> None:
    provider = FakeProvider(_responses_text_stream("partial output", incomplete=True))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": _PUBLIC_MODEL,
                "input": "Keep working",
                "max_output_tokens": 32,
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert events[-1].event == "response.incomplete"
    incomplete = events[-1].data["response"]
    assert incomplete["id"] == events[0].data["response"]["id"]
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_details"] == {"reason": "max_output_tokens"}
    assert incomplete["output"][0]["content"][0]["text"] == "partial output"


def test_create_response_preflight_rejection_stays_an_ordinary_http_error() -> None:
    provider = FakeProvider(_responses_text_stream("unused"))
    provider.preflight_responses.side_effect = InvalidRequestError("bad tool shape")
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Hello"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "bad tool shape",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
    assert "x-should-retry" not in response.headers
    assert provider.requests == []


def test_create_response_rejects_unportable_image_as_invalid_request() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=MagicMock(),
    ):
        provider = OpenAIChatProvider(
            make_provider_config(
                api_key="test-key",
                base_url="https://provider.invalid/v1",
            ),
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    provider_name="TEST_CHAT",
                    reasoning_replay=ReasoningReplayMode.DISABLED,
                ),
                NO_REASONING,
            ),
            admission=immediate_admission(provider_name="TEST_CHAT"),
        )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": _PUBLIC_MODEL,
                "input": [{"type": "input_image", "file_id": "file_1"}],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "file_id" in response.json()["error"]["message"]


def test_create_response_preserves_unknown_top_level_extensions(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client

    response = client.post(
        "/v1/responses",
        json={
            "model": _PUBLIC_MODEL,
            "input": "Hello",
            "provider_extension": {"enabled": True},
        },
    )

    assert response.status_code == 200
    assert provider.requests[0].model_dump()["provider_extension"] == {"enabled": True}


def test_create_response_pre_start_provider_error_returns_openai_error() -> None:
    provider = PreStartFailingProvider()
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        patch("free_claude_code.api.response_streams.trace_event") as trace,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Hello"},
        )

    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"] == response.headers["request-id"]
    payload = response.json()
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["message"] == "upstream is busy"
    request_id = response.headers["request-id"]
    assert provider.stream_kwargs[0]["request_id"] == request_id
    terminal_trace = next(
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event")
        == "free_claude_code.api.response.terminal_execution_error"
    )
    assert terminal_trace["wire_api"] == "responses"
    assert terminal_trace["request_id"] == request_id
    assert terminal_trace["status_code"] == 429
    assert terminal_trace["error_type"] == "rate_limit_error"
    assert terminal_trace["client_should_retry"] is False
    assert terminal_trace["failure_kind"] == "rate_limit"
    assert terminal_trace["provider_retryable"] is True


def test_create_response_relays_provider_owned_post_start_failure() -> None:
    provider = FakeProvider(
        [
            _created_event(),
            _terminal_event(
                "failed",
                error={
                    "message": "socket closed",
                    "type": "api_error",
                    "param": None,
                    "code": None,
                },
            ),
        ]
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Hello"},
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == ["response.created", "response.failed"]
    assert events[-1].data["response"]["id"] == events[0].data["response"]["id"]
    assert events[-1].data["response"]["status"] == "failed"
    assert events[-1].data["response"]["error"]["message"] == "socket closed"


def test_create_response_stream_bypasses_local_message_optimizations() -> None:
    provider = FakeProvider(_responses_text_stream("Provider response"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        patch(
            "free_claude_code.api.handlers.messages.try_optimizations",
            side_effect=AssertionError("Responses must not use message optimizations"),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "quota check"},
        )

    assert response.status_code == 200
    completed = parse_sse_text(response.text)[-1].data["response"]
    assert completed["output"][0]["content"][0]["text"] == "Provider response"
    assert provider.requests[0].input == "quota check"


def test_create_response_stream_false_returns_openai_error(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client

    response = client.post(
        "/v1/responses",
        json={"model": _PUBLIC_MODEL, "input": "Hello", "stream": False},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "streaming only" in payload["error"]["message"]
    assert provider.requests == []


def test_create_response_relays_interleaved_reasoning_order() -> None:
    output: list[JsonObject] = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "first thought"}],
        },
        _message_output("first answer", item_id="msg_1"),
        {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"value":"FCC"}',
        },
        {
            "id": "rs_2",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "second thought"}],
        },
        _message_output("final answer", item_id="msg_2"),
    ]
    provider = FakeProvider(
        [
            _created_event(),
            _event(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "sequence_number": 1,
                    "item_id": "rs_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "first thought",
                },
            ),
            _terminal_event("completed", output=output),
        ]
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Use reasoning and tools"},
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert "response.reasoning_text.delta" in [event.event for event in events]
    completed = events[-1].data["response"]
    assert [item["type"] for item in completed["output"]] == [
        "reasoning",
        "message",
        "function_call",
        "reasoning",
        "message",
    ]


def test_create_response_relays_function_call() -> None:
    call: JsonObject = {
        "id": "fc_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "toolu_1",
        "name": "echo",
        "arguments": '{"value":"FCC"}',
    }
    provider = FakeProvider(
        [_created_event(), _terminal_event("completed", output=[call])]
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": _PUBLIC_MODEL,
                "input": "Use echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "echo",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

    assert response.status_code == 200
    completed_call = parse_sse_text(response.text)[-1].data["response"]["output"][0]
    assert completed_call == call


def test_create_response_preserves_namespace_and_passive_tools() -> None:
    tools: list[JsonObject] = [
        {"type": "web_search", "external_web_access": True},
        {"type": "image_generation", "output_format": "png"},
        {
            "type": "namespace",
            "name": "mcp__node_repl",
            "tools": [
                {
                    "type": "function",
                    "name": "js",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                }
            ],
        },
    ]
    provider = FakeProvider(_responses_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Use JS", "tools": tools},
        )

    assert response.status_code == 200
    assert provider.requests[0].tools == tools


def test_create_response_preserves_muse_code_request_shape() -> None:
    request = {
        "model": _PUBLIC_MODEL,
        "input": "Read the file",
        "instructions": "Be concise.",
        "max_output_tokens": 64,
        "store": False,
        "stream": True,
        "reasoning": {"effort": "high", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "muse-session-1",
        "tools": [
            {
                "type": "namespace",
                "name": "muse",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "Read one file.",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    }
                ],
            }
        ],
    }
    provider = FakeProvider(_responses_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post("/v1/responses", json=request)

    assert response.status_code == 200
    routed = provider.requests[0]
    assert routed.max_output_tokens == 64
    assert routed.reasoning == {"effort": "high", "summary": "auto"}
    assert routed.model_dump()["include"] == ["reasoning.encrypted_content"]
    assert routed.model_dump()["prompt_cache_key"] == "muse-session-1"
    expected_policy = ReasoningPolicy(
        control=ReasoningControl.DEFAULT,
        effort=ReasoningEffort.HIGH,
    )
    assert provider.stream_kwargs[0]["reasoning"] == expected_policy
    assert provider.preflight_responses.call_args.kwargs["reasoning"] == expected_policy


def test_create_response_preserves_custom_tool_request() -> None:
    tool: JsonObject = {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply repo patches",
        "format": {"type": "text"},
    }
    choice: JsonObject = {"type": "custom", "name": "apply_patch"}
    provider = FakeProvider(_responses_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": _PUBLIC_MODEL,
                "input": "Use apply_patch",
                "tools": [tool],
                "tool_choice": choice,
            },
        )

    assert response.status_code == 200
    assert provider.requests[0].tools == [tool]
    assert provider.requests[0].tool_choice == choice


def test_create_response_relays_provider_error_lifecycle() -> None:
    error: JsonObject = {
        "message": "provider failed",
        "type": "api_error",
        "param": None,
        "code": None,
    }
    provider = FakeProvider([_created_event(), _terminal_event("failed", error=error)])
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Hello"},
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == ["response.created", "response.failed"]
    failed = events[-1].data["response"]
    assert failed["id"] == events[0].data["response"]["id"]
    assert failed["status"] == "failed"
    assert failed["error"] == error


def test_create_response_preserves_prior_reasoning_and_tool_history() -> None:
    input_items: list[JsonObject] = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Need the tool."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
        {"role": "user", "content": "continue"},
    ]
    provider = FakeProvider(_responses_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": input_items},
        )

    assert response.status_code == 200
    assert provider.requests[0].input == input_items


def test_create_response_preserves_malformed_prior_function_call() -> None:
    input_items: list[JsonObject] = [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "call_id": "call_bad",
            "name": "echo",
            "arguments": "{",
        },
        {
            "type": "function_call_output",
            "call_id": "call_bad",
            "output": "stale output",
        },
    ]
    provider = FakeProvider(_responses_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": input_items},
        )

    assert response.status_code == 200
    assert provider.requests[0].input == input_items


@pytest.mark.parametrize(
    ("reasoning", "expected_policy"),
    [
        ({"effort": "none"}, ReasoningPolicy.off()),
        (
            {"effort": "low"},
            ReasoningPolicy(
                control=ReasoningControl.DEFAULT,
                effort=ReasoningEffort.LOW,
            ),
        ),
    ],
)
def test_create_response_preserves_and_resolves_reasoning_effort(
    reasoning: JsonObject,
    expected_policy: ReasoningPolicy,
) -> None:
    provider = FakeProvider(_responses_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": _PUBLIC_MODEL,
                "input": "Hello",
                "reasoning": reasoning,
            },
        )

    assert response.status_code == 200
    assert provider.requests[0].reasoning == reasoning
    assert provider.stream_kwargs[0]["reasoning"] == expected_policy
    assert provider.preflight_responses.call_args.kwargs["reasoning"] == expected_policy


def test_create_response_relays_encrypted_reasoning() -> None:
    reasoning: JsonObject = {
        "id": "rs_1",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "encrypted_content": "opaque-redacted",
    }
    provider = FakeProvider(
        [_created_event(), _terminal_event("completed", output=[reasoning])]
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": _PUBLIC_MODEL, "input": "Continue"},
        )

    assert response.status_code == 200
    assert parse_sse_text(response.text)[-1].data["response"]["output"] == [reasoning]


def test_create_response_provider_rejects_unsupported_tool(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client
    provider.preflight_responses.side_effect = InvalidRequestError(
        "Unsupported Responses tool type: 'web_search_preview'"
    )

    response = client.post(
        "/v1/responses",
        json={
            "model": _PUBLIC_MODEL,
            "input": "Hello",
            "tools": [{"type": "web_search_preview"}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "Unsupported Responses tool type" in payload["error"]["message"]


def _event(event_type: str, payload: JsonObject) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _response(
    *,
    status: str,
    output: list[JsonObject] | None = None,
    error: JsonObject | None = None,
    incomplete_details: JsonObject | None = None,
) -> JsonObject:
    return {
        "id": _RESPONSE_ID,
        "object": "response",
        "created_at": 1,
        "model": _PUBLIC_MODEL,
        "status": status,
        "output": output or [],
        "error": error,
        "incomplete_details": incomplete_details,
        "usage": (
            None
            if status == "in_progress"
            else {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7,
            }
        ),
    }


def _created_event() -> str:
    return _event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _response(status="in_progress"),
        },
    )


def _terminal_event(
    status: str,
    *,
    output: list[JsonObject] | None = None,
    error: JsonObject | None = None,
) -> str:
    event_type = f"response.{status}"
    incomplete_details: JsonObject | None = (
        {"reason": "max_output_tokens"} if status == "incomplete" else None
    )
    return _event(
        event_type,
        {
            "type": event_type,
            "sequence_number": 2,
            "response": _response(
                status=status,
                output=output,
                error=error,
                incomplete_details=incomplete_details,
            ),
        },
    )


def _message_output(text: str, *, item_id: str = "msg_1") -> JsonObject:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _responses_text_stream(text: str, *, incomplete: bool = False) -> list[str]:
    status = "incomplete" if incomplete else "completed"
    return [
        _created_event(),
        _event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
        ),
        _terminal_event(status, output=[_message_output(text)]),
    ]
