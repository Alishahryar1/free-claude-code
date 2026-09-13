import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.responses import JSONResponse, StreamingResponse

import free_claude_code.api.web_tools.constants as web_tool_constants
from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.api.web_tools import egress as web_egress
from free_claude_code.api.web_tools.egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
)
from free_claude_code.api.web_tools.outbound import (
    _drain_response_body_capped,
    _read_response_body_capped,
    _run_web_fetch,
)
from free_claude_code.api.web_tools.request import (
    HIDDEN_WEB_SEARCH_NAME,
    is_web_server_tool_request,
    plan_automatic_web_search,
    unsupported_server_tool_error,
)
from free_claude_code.api.web_tools.streaming import stream_web_server_tool_response
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import (
    ModelRouter,
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import (
    ContentBlockServerToolUse,
    Message,
    MessagesRequest,
    Tool,
)
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.version import package_version
from free_claude_code.messaging.event_parser import parse_cli_event

_STRICT_EGRESS = WebFetchEgressPolicy(
    allow_private_network_targets=False,
    allowed_schemes=frozenset({"http", "https"}),
)
_PROVIDER_IDS = tuple(PROVIDER_CATALOG)


def test_web_tool_user_agent_reports_installed_package_version() -> None:
    assert {
        "User-Agent": (f"Mozilla/5.0 compatible; free-claude-code/{package_version()}")
    } == web_tool_constants._WEB_TOOL_HTTP_HEADERS


class FixedProviderModelRouter(ModelRouter):
    """Test double that pins provider identity."""

    def __init__(
        self,
        settings: Settings,
        provider_id: str,
        *,
        provider_model: str | None = None,
    ) -> None:
        super().__init__(settings)
        self._fixed_provider_id = provider_id
        self._fixed_provider_model = provider_model

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        provider_model = self._fixed_provider_model or request.model
        target = ProviderModelTarget(
            provider_id=self._fixed_provider_id,
            provider_model=provider_model,
            provider_model_ref=f"{self._fixed_provider_id}/{provider_model}",
        )
        resolved = ResolvedModelRoute(
            original_model=request.model,
            primary=target,
            fallbacks=(),
            reasoning_preference=ReasoningPreference.OFF,
        )
        routed = request.model_copy(deep=True)
        routed.model = resolved.primary.provider_model
        return RoutedMessagesRequest(
            request=routed,
            resolved=resolved,
            reasoning=ReasoningPolicy.off(),
        )


class ScriptedSelectionProvider:
    """One-stream provider double for the automatic WebSearch decision."""

    def __init__(
        self,
        events: list[str],
        *,
        failure: ExecutionFailure | None = None,
        failure_after_events: bool = False,
        wait_for: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.failure = failure
        self.failure_after_events = failure_after_events
        self.wait_for = wait_for
        self.started = asyncio.Event()
        self.requests: list[MessagesRequest] = []
        self.stream_kwargs: list[dict[str, object]] = []
        self.close_count = 0

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        raise AssertionError("Web-search selection received a Responses request")
        yield ""

    async def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self.requests.append(request)
        self.stream_kwargs.append(
            {
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        self.started.set()
        try:
            if self.wait_for is not None:
                await self.wait_for.wait()
            if self.failure is not None and not self.failure_after_events:
                raise self.failure
            for event in self.events:
                yield event
            if self.failure is not None:
                raise self.failure
        finally:
            self.close_count += 1


def _provider_text_events(text: str) -> list[str]:
    return [
        format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_provider",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "gateway-model",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _provider_tool_events(
    *,
    name: str = HIDDEN_WEB_SEARCH_NAME,
    arguments: dict[str, object] | None = None,
    additional_calls: int = 0,
) -> list[str]:
    calls = [(name, arguments if arguments is not None else {"query": "selected"})]
    calls.extend(
        (name, {"query": f"extra-{index}"}) for index in range(additional_calls)
    )
    events = [
        format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_provider",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "gateway-model",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 17,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 5,
                    },
                },
            },
        )
    ]
    for index, (tool_name, tool_input) in enumerate(calls):
        events.extend(
            [
                format_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": f"call_{index}",
                            "name": tool_name,
                            "input": {},
                        },
                    },
                ),
                format_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tool_input),
                        },
                    },
                ),
                format_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": index},
                ),
            ]
        )
    events.extend(
        [
            format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 9},
                },
            ),
            format_sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    return events


def _automatic_search_request(
    *,
    stream: bool = True,
    tool: Tool | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> MessagesRequest:
    return MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        stream=stream,
        messages=[
            Message(role="user", content="Prompt text must not become the query")
        ],
        tools=[tool or Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "auto"} if tool_choice is None else tool_choice,
    )


def _automatic_search_service(
    provider: ScriptedSelectionProvider,
    *,
    settings: Settings | None = None,
) -> MessagesHandler:
    effective_settings = settings or Settings()
    return MessagesHandler(
        effective_settings,
        provider_resolver=lambda _: provider,
        model_router=FixedProviderModelRouter(
            effective_settings,
            _PROVIDER_IDS[0],
            provider_model="upstream-model",
        ),
    )


def test_web_server_tool_not_detected_when_tool_only_listed():
    """Listing web_search without forcing it must not skip the upstream provider."""
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )

    assert not is_web_server_tool_request(request)


def test_web_server_tool_detected_when_tool_choice_forces_it():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert is_web_server_tool_request(request)


def test_web_server_tool_not_detected_when_forced_name_missing_from_tools():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
        tools=[Tool(name="other", type="function")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert not is_web_server_tool_request(request)


@pytest.mark.parametrize("tool_choice", [None, {"type": "auto"}])
def test_plans_exact_automatic_web_search(tool_choice: dict[str, Any] | None) -> None:
    request = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[Message(role="user", content="search")],
        tools=[
            Tool(
                name="web_search",
                type="web_search_20250305",
                max_uses=8,
                allowed_domains=["Example.com"],
            )
        ],
        tool_choice=tool_choice,
    )

    plan = plan_automatic_web_search(request, web_tools_enabled=True)

    assert plan is not None
    assert plan.domains.allowed == ("example.com",)
    assert plan.domains.blocked == ()
    assert plan.request.tool_choice == tool_choice
    assert plan.request.tools is not None
    assert [tool.name for tool in plan.request.tools] == [HIDDEN_WEB_SEARCH_NAME]
    assert plan.request.tools[0].input_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    assert request.tools is not None
    assert request.tools[0].name == "web_search"


def test_ordinary_function_named_web_search_remains_a_client_tool() -> None:
    request = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[Message(role="user", content="search")],
        tools=[
            Tool(
                name="web_search",
                description="ordinary client function",
                input_schema={"type": "object"},
            )
        ],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert plan_automatic_web_search(request, web_tools_enabled=True) is None
    assert unsupported_server_tool_error(request, web_tools_enabled=True) is None
    assert not is_web_server_tool_request(request)


@pytest.mark.parametrize(
    "tool",
    [
        Tool(name="web_fetch", type="web_fetch_20250910"),
        Tool(name="web_search", type="web_search_20260209"),
        Tool(name="WebSearch", input_schema={"type": "object"}),
        Tool(name="WebFetch", input_schema={"type": "object"}),
    ],
)
def test_does_not_plan_noncanonical_automatic_search(tool: Tool) -> None:
    request = _automatic_search_request(tool=tool)

    assert plan_automatic_web_search(request, web_tools_enabled=True) is None


def test_does_not_plan_mixed_tools_or_extended_auto_choice() -> None:
    mixed = _automatic_search_request()
    mixed.tools = [
        Tool(name="web_search", type="web_search_20250305"),
        Tool(name="client_tool", input_schema={"type": "object"}),
    ]
    extended_choice = _automatic_search_request(
        tool_choice={"type": "auto", "disable_parallel_tool_use": True}
    )

    for request in (mixed, extended_choice):
        assert plan_automatic_web_search(request, web_tools_enabled=True) is None
        assert (
            unsupported_server_tool_error(request, web_tools_enabled=True) is not None
        )


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        (
            Tool(name="other", type="web_search_20250305"),
            "must use name 'web_search'",
        ),
        (
            Tool(
                name="web_search",
                type="web_search_20250305",
                description="custom",
            ),
            "must not define description",
        ),
        (
            Tool(name="web_search", type="web_search_20250305", max_uses=0),
            "must be a positive integer",
        ),
        (
            Tool(name="web_search", type="web_search_20250305", unknown="value"),
            "unsupported option",
        ),
        (
            Tool(
                name="web_search",
                type="web_search_20250305",
                allowed_domains=["example.com"],
                blocked_domains=["blocked.test"],
            ),
            "may define allowed_domains or blocked_domains",
        ),
    ],
)
def test_rejects_malformed_automatic_web_search_definition(
    tool: Tool, message: str
) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        plan_automatic_web_search(
            _automatic_search_request(tool=tool), web_tools_enabled=True
        )


def test_rejects_automatic_web_search_with_server_tool_history() -> None:
    request = _automatic_search_request()
    request.messages.insert(
        0,
        Message(
            role="assistant",
            content=[
                ContentBlockServerToolUse(
                    type="server_tool_use",
                    id="srvtoolu_prior",
                    name="web_search",
                    input={"query": "prior"},
                )
            ],
        ),
    )

    with pytest.raises(InvalidRequestError, match="prior Anthropic server-tool"):
        plan_automatic_web_search(request, web_tools_enabled=True)


@pytest.mark.parametrize(
    "domain",
    ["*.example.com", "example.com/path", "https://example.com", "example.com:443"],
)
def test_rejects_web_search_domain_wildcards_and_url_parts(domain: str) -> None:
    request = _automatic_search_request(
        tool=Tool(
            name="web_search",
            type="web_search_20250305",
            allowed_domains=[domain],
        )
    )

    with pytest.raises(InvalidRequestError, match="literal hostname"):
        plan_automatic_web_search(request, web_tools_enabled=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
async def test_service_rejects_forced_server_tool_when_local_handler_is_disabled(
    provider_id: str,
):
    """Every provider needs FCC's local handler for forced server tools."""
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": False})
    assert settings.enable_web_server_tools is False
    service = MessagesHandler(
        settings,
        provider_resolver=lambda _: MagicMock(),
        model_router=FixedProviderModelRouter(settings, provider_id),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Perform a web search for the query: DeepSeek V4 model release 2026",
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )
    with pytest.raises(InvalidRequestError, match="ENABLE_WEB_SERVER_TOOLS"):
        await service.create(request)


@pytest.mark.asyncio
async def test_automatic_web_search_replays_provider_response_when_declined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _provider_text_events("No search needed")
    provider = ScriptedSelectionProvider(events)
    search = AsyncMock()
    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    service = _automatic_search_service(provider)

    response = await service.create(
        _automatic_search_request(), request_id="req_automatic_declined"
    )

    assert isinstance(response, StreamingResponse)
    assert await _streaming_body_text(response) == "".join(events)
    search.assert_not_awaited()
    assert provider.close_count == 1
    assert len(provider.requests) == 1
    translated = provider.requests[0]
    assert translated.model == "upstream-model"
    assert translated.tool_choice == {"type": "auto"}
    assert translated.tools is not None
    assert [tool.name for tool in translated.tools] == [HIDDEN_WEB_SEARCH_NAME]
    assert provider.stream_kwargs == [
        {
            "input_tokens": provider.stream_kwargs[0]["input_tokens"],
            "request_id": "req_automatic_declined",
            "response_model": "claude-haiku-4-5-20251001",
            "reasoning": ReasoningPolicy.off(),
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("domain_option", "domain"),
    [
        ("allowed_domains", "example.com"),
        ("blocked_domains", "unrelated.test"),
    ],
)
async def test_automatic_web_search_uses_model_query_and_filters_domains(
    domain_option: str,
    domain: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(
        _provider_tool_events(arguments={"query": "model selected query"})
    )
    seen_queries: list[str] = []

    async def fake_search(query: str) -> list[dict[str, str]]:
        seen_queries.append(query)
        return [
            {"title": "Allowed", "url": "https://docs.example.com/page"},
            {"title": "Filtered", "url": "https://unrelated.test/page"},
        ]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    service = _automatic_search_service(provider)
    request = _automatic_search_request(
        tool=Tool.model_validate(
            {
                "name": "web_search",
                "type": "web_search_20250305",
                "max_uses": 8,
                domain_option: [domain],
            }
        )
    )

    response = await service.create(request, request_id="req_automatic_selected")

    assert isinstance(response, StreamingResponse)
    raw = await _streaming_body_text(response)
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    assert seen_queries == ["model selected query"]
    assert HIDDEN_WEB_SEARCH_NAME not in raw
    assert "call_0" not in raw
    assert "Prompt text must not become the query" not in raw
    assert "docs.example.com/page" in raw
    assert "unrelated.test/page" not in raw
    assert sum(event.event == "message_start" for event in events) == 1
    assert sum(event.event == "message_stop" for event in events) == 1
    starts = [event for event in events if event.event == "content_block_start"]
    assert [start.data["content_block"]["type"] for start in starts] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert starts[0].data["content_block"]["input"] == {"query": "model selected query"}
    assert (
        starts[1].data["content_block"]["tool_use_id"]
        == starts[0].data["content_block"]["id"]
    )
    message_start = next(
        event.data["message"] for event in events if event.event == "message_start"
    )
    assert message_start["model"] == "claude-haiku-4-5-20251001"
    final_usage = next(
        event.data["usage"] for event in events if event.event == "message_delta"
    )
    assert final_usage == {
        "input_tokens": 17,
        "output_tokens": 9,
        "cache_read_input_tokens": 5,
        "server_tool_use": {"web_search_requests": 1},
    }
    assert len(provider.requests) == 1
    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_automatic_web_search_aggregates_when_stream_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(
        _provider_tool_events(arguments={"query": "json query"})
    )

    async def fake_search(_query: str) -> list[dict[str, str]]:
        return [{"title": "JSON result", "url": "https://example.com/json"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    service = _automatic_search_service(provider)

    response = await service.create(
        _automatic_search_request(stream=False),
        request_id="req_automatic_json",
    )

    assert isinstance(response, JSONResponse)
    body = _json_body(response)
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert body["content"][1]["content"][0]["url"] == "https://example.com/json"
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        _provider_tool_events(arguments={"query": ""}),
        _provider_tool_events(name="unexpected_tool"),
        _provider_tool_events(additional_calls=1),
    ],
    ids=["blank-query", "wrong-tool", "multiple-tools"],
)
async def test_automatic_web_search_rejects_malformed_provider_selection(
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(events)
    search = AsyncMock()
    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    service = _automatic_search_service(provider)

    response = await service.create(
        _automatic_search_request(), request_id="req_malformed_selection"
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["content-type"].startswith("application/json")
    assert _json_body(response)["request_id"] == "req_malformed_selection"
    search.assert_not_awaited()
    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_automatic_web_search_preserves_provider_failure_before_public_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(
        [],
        failure=ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="provider exhausted retries",
            retryable=False,
        ),
    )
    search = AsyncMock()
    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    service = _automatic_search_service(provider)

    response = await service.create(
        _automatic_search_request(), request_id="req_selection_failure"
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    assert _json_body(response)["error"]["type"] == "rate_limit_error"
    search.assert_not_awaited()
    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_automatic_web_search_converts_internal_post_start_failure_to_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(
        _provider_text_events("partial")[:2],
        failure=ExecutionFailure(
            kind=FailureKind.OVERLOADED,
            status_code=529,
            message="provider exhausted retries after starting",
            retryable=False,
        ),
        failure_after_events=True,
    )
    search = AsyncMock()
    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    service = _automatic_search_service(provider)

    response = await service.create(
        _automatic_search_request(), request_id="req_selection_post_start_failure"
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 529
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["content-type"].startswith("application/json")
    body = _json_body(response)
    assert body["error"]["type"] == "overloaded_error"
    assert body["request_id"] == "req_selection_post_start_failure"
    search.assert_not_awaited()
    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_automatic_web_search_cancellation_closes_provider_stream() -> None:
    release = asyncio.Event()
    provider = ScriptedSelectionProvider(
        _provider_text_events("unused"), wait_for=release
    )
    service = _automatic_search_service(provider)
    task = asyncio.create_task(
        service.create(
            _automatic_search_request(), request_id="req_selection_cancelled"
        )
    )
    await provider.started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.close_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://192.168.1.1/",
        "http://10.0.0.1/",
        "http://[::1]/",
        "http://localhost/foo",
        "http://mybox.local/",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_enforce_web_fetch_egress_blocks_internal_or_disallowed(url: str):
    with pytest.raises(WebFetchEgressViolation):
        enforce_web_fetch_egress(url, _STRICT_EGRESS)


def test_enforce_web_fetch_egress_allows_global_literal_ip():
    enforce_web_fetch_egress("http://8.8.8.8/", _STRICT_EGRESS)


def test_enforce_web_fetch_egress_skips_private_checks_when_opted_in():
    enforce_web_fetch_egress(
        "http://127.0.0.1/",
        WebFetchEgressPolicy(
            allow_private_network_targets=True,
            allowed_schemes=frozenset({"http", "https"}),
        ),
    )


def _cm(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _stream_cm(response: httpx.Response) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _json_body(response: JSONResponse) -> dict[str, Any]:
    payload = json.loads(bytes(response.body).decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


async def _streaming_body_text(response: StreamingResponse) -> str:
    parts = [
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ]
    return "".join(parts)


def _aiohttp_response(
    status: int,
    *,
    url: str = "http://8.8.8.8/",
    location: str | None = None,
    body: bytes = b"hello world",
) -> MagicMock:
    r = MagicMock()
    r.status = status
    r.url = url
    hdrs: dict[str, str] = {}
    if location is not None:
        hdrs["location"] = location
    r.headers = hdrs
    r.get_encoding = MagicMock(return_value="utf-8")
    r.raise_for_status = MagicMock()
    r.request_info = MagicMock()
    r.history = ()

    async def iter_chunked(_n: int) -> Any:
        yield body

    r.content.iter_chunked = MagicMock(side_effect=iter_chunked)
    return r


def _aiohttp_client_session_patch(
    *responses: MagicMock,
) -> tuple[MagicMock, MagicMock]:
    """Build ``ClientSession`` mock that serves ``responses`` to successive ``get`` calls."""
    queue = list(responses)
    n = 0

    def get_side(*_a: Any, **_k: Any) -> Any:
        nonlocal n
        resp = queue[n] if n < len(queue) else queue[-1]
        n += 1
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    session = MagicMock()
    session.get = MagicMock(side_effect=get_side)

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=session)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    return client_cm, session


def test_enforce_web_fetch_egress_documents_connect_time_pinning():
    assert enforce_web_fetch_egress.__doc__ and "resolved addresses" in (
        enforce_web_fetch_egress.__doc__ or ""
    )
    assert (
        web_egress.get_validated_stream_addrinfos_for_egress.__doc__
        and "pinning"
        in (web_egress.get_validated_stream_addrinfos_for_egress.__doc__ or "")
    )
    assert "DNS-pinned" in (_run_web_fetch.__doc__ or "")


@pytest.mark.asyncio
async def test_run_web_fetch_follows_redirect_when_each_hop_is_allowed():
    res_redirect = _aiohttp_response(
        302, url="http://8.8.8.8/start", location="/final", body=b""
    )
    res_ok = _aiohttp_response(200, url="http://8.8.8.8/final", body=b"hello world")
    client_cm, session = _aiohttp_client_session_patch(res_redirect, res_ok)
    with patch(
        "free_claude_code.api.web_tools.outbound.ClientSession", return_value=client_cm
    ):
        out = await _run_web_fetch("http://8.8.8.8/start", _STRICT_EGRESS)

    assert out["data"] == "hello world"
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_run_web_fetch_truncates_large_body_to_byte_cap(monkeypatch):
    huge = b"x" * 5000
    res_ok = _aiohttp_response(200, url="http://8.8.8.8/big", body=huge)
    client_cm, _ = _aiohttp_client_session_patch(res_ok)
    monkeypatch.setattr(web_tool_constants, "_MAX_WEB_FETCH_RESPONSE_BYTES", 100)
    with patch(
        "free_claude_code.api.web_tools.outbound.ClientSession", return_value=client_cm
    ):
        out = await _run_web_fetch("http://8.8.8.8/big", _STRICT_EGRESS)

    assert len(out["data"]) <= 100
    assert out["data"] == "x" * 100


@pytest.mark.asyncio
async def test_run_web_fetch_redirect_to_blocked_host_raises():
    res_redirect = _aiohttp_response(
        302,
        url="http://8.8.8.8/start",
        location="http://127.0.0.1/secret",
        body=b"",
    )
    client_cm, session = _aiohttp_client_session_patch(res_redirect)
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation),
    ):
        await _run_web_fetch("http://8.8.8.8/start", _STRICT_EGRESS)

    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_run_web_fetch_redirect_without_location_raises():
    res_bad = _aiohttp_response(302, url="http://8.8.8.8/here", body=b"")
    client_cm, _ = _aiohttp_client_session_patch(res_bad)
    with (
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation, match="missing Location"),
    ):
        await _run_web_fetch("http://8.8.8.8/here", _STRICT_EGRESS)


@pytest.mark.asyncio
async def test_run_web_fetch_excess_redirects_raises():
    res1 = _aiohttp_response(302, url="http://8.8.8.8/a", location="/b", body=b"")
    res2 = _aiohttp_response(302, url="http://8.8.8.8/b", location="/c", body=b"")
    client_cm, _ = _aiohttp_client_session_patch(res1, res2)
    with (
        patch("free_claude_code.api.web_tools.constants._MAX_WEB_FETCH_REDIRECTS", 1),
        patch(
            "free_claude_code.api.web_tools.outbound.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation, match="exceeded maximum redirects"),
    ):
        await _run_web_fetch("http://8.8.8.8/a", _STRICT_EGRESS)


@pytest.mark.asyncio
async def test_streams_web_search_server_tool_result(monkeypatch):
    async def fake_search(query: str) -> list[dict[str, str]]:
        assert query == "DeepSeek V4 model release 2026"
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content=(
                    "Perform a web search for the query: DeepSeek V4 model release 2026"
                ),
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=42, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    starts = [e for e in events if e.event == "content_block_start"]
    assert starts[0].data["content_block"]["type"] == "server_tool_use"
    assert starts[0].data["content_block"]["name"] == "web_search"
    tool_use_id = starts[0].data["content_block"]["id"]
    assert starts[1].data["content_block"]["type"] == "web_search_tool_result"
    assert starts[1].data["content_block"]["tool_use_id"] == tool_use_id
    assert starts[1].data["content_block"]["content"][0]["url"] == (
        "https://example.com/v4"
    )
    text_deltas = [
        e
        for e in events
        if e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
    ]
    assert text_deltas, "summary must be streamed as text_delta"
    assert "example.com" in text_content(events)
    cli_text: list[str] = []
    for ev in events:
        cli_text.extend(
            str(p.get("text", ""))
            for p in parse_cli_event(ev.data)
            if p.get("type") == "text_delta"
        )
    assert "example.com" in "".join(cli_text)
    deltas = [e for e in events if e.event == "message_delta"]
    assert deltas[-1].data["usage"]["server_tool_use"] == {"web_search_requests": 1}


@pytest.mark.asyncio
async def test_service_streams_forced_web_search_by_default(monkeypatch):
    async def fake_search(_query: str) -> list[dict[str, str]]:
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = MagicMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(
            settings,
            _PROVIDER_IDS[0],
            provider_model="upstream-model",
        ),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        stream=True,
        messages=[Message(role="user", content="Search for DeepSeek V4")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    response = await service.create(request)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    raw = await _streaming_body_text(response)
    assert "event: message_start" in raw
    assert "DeepSeek V4 Released" in raw
    message_start = next(
        event.data["message"]
        for event in parse_sse_text(raw)
        if event.event == "message_start"
    )
    assert message_start["model"] == request.model
    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_service_aggregates_forced_web_search_when_stream_false(monkeypatch):
    async def fake_search(_query: str) -> list[dict[str, str]]:
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", fake_search
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = MagicMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(settings, _PROVIDER_IDS[0]),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="Search for DeepSeek V4")],
        stream=False,
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    response = await service.create(request)

    assert isinstance(response, JSONResponse)
    assert response.headers["content-type"].startswith("application/json")
    body = _json_body(response)
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert body["content"][1]["content"][0]["url"] == "https://example.com/v4"
    assert "DeepSeek V4 Released" in body["content"][2]["text"]
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}
    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_forced_web_fetch_ignores_stale_url_from_prior_user_turns(monkeypatch):
    """Only the latest user message supplies the URL (not earlier transcript text)."""
    target = "https://new-only.example.com/page"

    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        assert url == target
        return {
            "url": url,
            "title": "T",
            "media_type": "text/plain",
            "data": "x",
        }

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_fetch", fake_fetch
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Earlier turn https://stale.com/old-article ignore this",
            ),
            Message(role="assistant", content="ok"),
            Message(
                role="user",
                content=f"Please fetch {target} for the summary",
            ),
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=1, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    assert target in raw


@pytest.mark.asyncio
async def test_service_aggregates_forced_web_fetch_when_stream_false(monkeypatch):
    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        return {
            "url": url,
            "title": "Example Article",
            "media_type": "text/plain",
            "data": "Article body",
        }

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_fetch", fake_fetch
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = MagicMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(settings, _PROVIDER_IDS[0]),
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="Fetch https://example.com/article")],
        stream=False,
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    response = await service.create(request)

    assert isinstance(response, JSONResponse)
    assert response.headers["content-type"].startswith("application/json")
    body = _json_body(response)
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_fetch_tool_result",
        "text",
    ]
    assert body["content"][1]["content"]["content"]["title"] == "Example Article"
    assert body["content"][2]["text"] == "Article body"
    assert body["usage"]["server_tool_use"] == {"web_fetch_requests": 1}
    provider_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_streams_web_fetch_server_tool_result(monkeypatch):
    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        assert url == "https://example.com/article"
        return {
            "url": url,
            "title": "Example Article",
            "media_type": "text/plain",
            "data": "Article body",
        }

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_fetch", fake_fetch
    )
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(role="user", content="Fetch https://example.com/article please")
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=42, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    starts = [e for e in events if e.event == "content_block_start"]
    assert starts[0].data["content_block"]["type"] == "server_tool_use"
    tool_use_id = starts[0].data["content_block"]["id"]
    assert starts[1].data["content_block"]["type"] == "web_fetch_tool_result"
    assert starts[1].data["content_block"]["tool_use_id"] == tool_use_id
    assert starts[1].data["content_block"]["content"]["content"]["title"] == (
        "Example Article"
    )
    assert any(
        e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
        for e in events
    )
    assert "Article body" in text_content(events)
    cli_text: list[str] = []
    for ev in events:
        cli_text.extend(
            str(p.get("text", ""))
            for p in parse_cli_event(ev.data)
            if p.get("type") == "text_delta"
        )
    assert "Article body" in "".join(cli_text)
    deltas = [e for e in events if e.event == "message_delta"]
    assert deltas[-1].data["usage"]["server_tool_use"] == {"web_fetch_requests": 1}


@pytest.mark.asyncio
async def test_streams_web_fetch_error_summary_generic_by_default(monkeypatch):
    secret = "sensitive-upstream-token"

    async def boom(_url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        raise ValueError(secret)

    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", boom)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Fetch https://example.com/sensitive-path?x=1 please",
            )
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    with patch("free_claude_code.api.web_tools.outbound.logger.warning") as log_warn:
        raw = "".join(
            [
                event
                async for event in stream_web_server_tool_response(
                    request,
                    input_tokens=1,
                    web_fetch_egress=_STRICT_EGRESS,
                    verbose_client_errors=False,
                )
            ]
        )

    assert secret not in raw
    assert "ValueError" not in raw
    assert "Web tool request failed." in raw
    err_events = parse_sse_text(raw)
    assert_anthropic_stream_contract(err_events)
    assert any(
        e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
        for e in err_events
    )
    cli_err_text: list[str] = []
    for ev in err_events:
        cli_err_text.extend(
            str(p.get("text", ""))
            for p in parse_cli_event(ev.data)
            if p.get("type") == "text_delta"
        )
    assert "Web tool request failed." in "".join(cli_err_text)
    log_blob = " ".join(str(a) for c in log_warn.call_args_list for a in c.args)
    assert secret not in log_blob
    assert "example.com" in log_blob
    assert "/sensitive-path" not in log_blob


@pytest.mark.asyncio
async def test_streams_web_fetch_error_summary_verbose_includes_exception_class(
    monkeypatch,
):
    async def boom(_url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        raise OSError(5, "oops")

    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", boom)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="Fetch https://example.com/x")],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request,
                input_tokens=1,
                web_fetch_egress=_STRICT_EGRESS,
                verbose_client_errors=True,
            )
        ]
    )
    assert "OSError" in raw


@pytest.mark.asyncio
async def test_read_response_body_capped_truncates_single_oversized_chunk():
    cap = 500

    async def aiter_bytes(chunk_size=None):
        yield b"z" * (cap * 20)

    response = MagicMock()
    response.aiter_bytes = aiter_bytes

    out = await _read_response_body_capped(response, cap)
    assert len(out) == cap
    assert out == b"z" * cap


@pytest.mark.asyncio
async def test_drain_response_body_capped_stops_after_first_chunk_when_oversized():
    cap = 300
    chunk_calls = {"n": 0}

    async def aiter_bytes(chunk_size=None):
        chunk_calls["n"] += 1
        yield b"y" * (cap * 10)

    response = MagicMock()
    response.aiter_bytes = aiter_bytes

    await _drain_response_body_capped(response, cap)
    assert chunk_calls["n"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
async def test_service_rejects_listed_server_tools_for_every_provider(
    provider_id: str,
) -> None:
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": False})
    service = MessagesHandler(
        settings,
        provider_resolver=lambda _: MagicMock(),
        model_router=FixedProviderModelRouter(settings, provider_id),
    )
    request = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[Message(role="user", content="q")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )
    with pytest.raises(InvalidRequestError, match="ENABLE_WEB_SERVER_TOOLS=false"):
        await service.create(request)
