import asyncio
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.responses import JSONResponse, StreamingResponse

import free_claude_code.runtime.web_tools.constants as web_tool_constants
from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import (
    ModelRouter,
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
)
from free_claude_code.application.web_tools.ports import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
)
from free_claude_code.application.web_tools.request import (
    HIDDEN_WEB_SEARCH_NAME,
    is_web_server_tool_request,
    plan_automatic_web_search,
    unsupported_server_tool_error,
)
from free_claude_code.application.web_tools.service import WebToolService
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.conversion import AnthropicToOpenAIConverter
from free_claude_code.core.anthropic.models import (
    ContentBlockServerToolUse,
    Message,
    MessagesRequest,
    Tool,
)
from free_claude_code.core.anthropic.native import (
    NativeMessagesOptions,
    build_native_messages_request,
)
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.version import package_version
from free_claude_code.core.web_tools import WebFetchResult, WebSearchResult
from free_claude_code.messaging.event_parser import parse_cli_event
from free_claude_code.runtime.web_tools import egress as web_egress
from free_claude_code.runtime.web_tools.client import (
    HTTPWebToolsClient,
    _drain_response_body_capped,
    _read_response_body_capped,
)
from free_claude_code.runtime.web_tools.egress import enforce_web_fetch_egress
from tests.web_tools_support import StubWebToolsClient

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


def _local_tool_body(
    request: MessagesRequest,
    input_tokens: int,
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
) -> AsyncGenerator[str]:
    settings = Settings().model_copy(
        update={
            "enable_web_server_tools": True,
            "web_fetch_allow_private_networks": web_fetch_egress.allow_private_network_targets,
            "web_fetch_allowed_schemes": ",".join(
                sorted(web_fetch_egress.allowed_schemes)
            ),
            "log_api_error_tracebacks": verbose_client_errors,
        }
    )

    async def unexpected_provider(_provider_id: str):
        pytest.fail("Forced local tools must not resolve a provider")

    service = WebToolService(
        settings=settings,
        client=StubWebToolsClient(),
        executor=ProviderExecutor(unexpected_provider, progress_timeout_seconds=30),
        token_counter=lambda *_: input_tokens,
    )
    routed = FixedProviderModelRouter(
        settings, _PROVIDER_IDS[0]
    ).resolve_messages_request(request)
    body = service.try_stream_messages(routed, request_id="req_local_tool")
    assert isinstance(body, AsyncGenerator)
    return body


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
        model_info: ProviderModelInfo | None = None,
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
    additional_inputs: list[dict[str, object]] | None = None,
) -> list[str]:
    calls = [(name, arguments if arguments is not None else {"query": "selected"})]
    calls.extend(
        (name, {"query": f"extra-{index}"}) for index in range(additional_calls)
    )
    calls.extend((name, item) for item in additional_inputs or [])
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
        provider_resolver=AsyncMock(side_effect=lambda _: provider),
        model_router=FixedProviderModelRouter(
            effective_settings,
            _PROVIDER_IDS[0],
            provider_model="upstream-model",
        ),
        web_tools=StubWebToolsClient(),
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
        provider_resolver=AsyncMock(side_effect=lambda _: MagicMock()),
        model_router=FixedProviderModelRouter(settings, provider_id),
        web_tools=StubWebToolsClient(),
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
    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.search", search)
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

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        seen_queries.append(query)
        return [
            WebSearchResult(title="Allowed", url="https://docs.example.com/page"),
            WebSearchResult(title="Filtered", url="https://unrelated.test/page"),
        ]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
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

    async def fake_search(self, _query: str) -> list[WebSearchResult]:
        return [WebSearchResult(title="JSON result", url="https://example.com/json")]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
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
async def test_automatic_web_search_runs_multiple_calls_in_one_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(_provider_tool_events(additional_calls=1))
    seen_queries: list[str] = []

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        seen_queries.append(query)
        return [
            WebSearchResult(title=f"{query} first", url="https://example.com/1"),
            WebSearchResult(title=f"{query} second", url="https://example.com/2"),
        ]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    service = _automatic_search_service(provider)

    response = await service.create(_automatic_search_request(stream=False))

    assert isinstance(response, JSONResponse)
    body = _json_body(response)
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    first_use, first_result, second_use, second_result, _ = body["content"]
    assert [first_use["input"]["query"], second_use["input"]["query"]] == [
        "selected",
        "extra-0",
    ]
    assert first_use["id"] != second_use["id"]
    assert first_result["tool_use_id"] == first_use["id"]
    assert second_result["tool_use_id"] == second_use["id"]
    assert len(first_result["content"]) == len(second_result["content"]) == 2
    assert set(seen_queries) == {"selected", "extra-0"}
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 2}


@pytest.mark.asyncio
async def test_automatic_web_search_streams_ordered_multiple_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(_provider_tool_events(additional_calls=1))

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        return [WebSearchResult(title=query, url=f"https://example.com/{query}")]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    response = await _automatic_search_service(provider).create(
        _automatic_search_request()
    )

    assert isinstance(response, StreamingResponse)
    events = parse_sse_text(await _streaming_body_text(response))
    assert_anthropic_stream_contract(events)
    starts = [event for event in events if event.event == "content_block_start"]
    assert [event.data["index"] for event in starts] == list(range(5))
    assert [event.data["content_block"]["type"] for event in starts] == [
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert sum(event.event == "message_start" for event in events) == 1
    assert sum(event.event == "message_stop" for event in events) == 1
    assert next(
        event.data["usage"]["server_tool_use"]
        for event in events
        if event.event == "message_delta"
    ) == {"web_search_requests": 2}


@pytest.mark.asyncio
async def test_automatic_web_search_max_uses_limits_current_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(_provider_tool_events(additional_calls=1))
    search = AsyncMock(
        return_value=[WebSearchResult(title="found", url="https://example.com/")]
    )
    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.search", search)
    request = _automatic_search_request(
        stream=False,
        tool=Tool(name="web_search", type="web_search_20250305", max_uses=1),
    )

    response = await _automatic_search_service(provider).create(request)

    assert isinstance(response, JSONResponse)
    body = _json_body(response)
    assert body["content"][1]["content"][0]["title"] == "found"
    assert body["content"][3]["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "max_uses_exceeded",
    }
    search.assert_awaited_once_with("selected")
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}


@pytest.mark.asyncio
async def test_automatic_web_search_keeps_other_results_when_one_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(_provider_tool_events(additional_calls=1))

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        if query == "extra-0":
            raise RuntimeError("search unavailable")
        return [WebSearchResult(title="found", url="https://example.com/")]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    response = await _automatic_search_service(provider).create(
        _automatic_search_request(stream=False)
    )

    assert isinstance(response, JSONResponse)
    body = _json_body(response)
    assert body["content"][1]["content"][0]["title"] == "found"
    assert body["content"][3]["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "unavailable",
    }
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 2}


@pytest.mark.asyncio
async def test_automatic_web_search_validates_all_calls_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(
        _provider_tool_events(additional_inputs=[{"query": ""}])
    )
    search = AsyncMock()
    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.search", search)

    response = await _automatic_search_service(provider).create(
        _automatic_search_request(stream=False)
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_automatic_web_search_bounds_parallel_local_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(_provider_tool_events(additional_calls=4))
    four_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0
    started: list[str] = []

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.append(query)
        if active == 4:
            four_started.set()
        try:
            await release.wait()
        finally:
            active -= 1
        return []

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    pending = asyncio.create_task(
        _automatic_search_service(provider).create(
            _automatic_search_request(stream=False)
        )
    )
    try:
        await asyncio.wait_for(four_started.wait(), 1)
        assert len(started) == 4
        assert peak == 4
        release.set()
        response = await asyncio.wait_for(pending, 1)
        assert isinstance(response, JSONResponse)
        assert _json_body(response)["usage"]["server_tool_use"] == {
            "web_search_requests": 5
        }
        assert len(started) == 5
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_automatic_web_search_cancellation_stops_parallel_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(_provider_tool_events(additional_calls=1))
    both_started = asyncio.Event()
    started: set[str] = set()
    cancelled: set[str] = set()

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        started.add(query)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(query)
        return []

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    response = await _automatic_search_service(provider).create(
        _automatic_search_request()
    )
    assert isinstance(response, StreamingResponse)
    drain = asyncio.create_task(_streaming_body_text(response))
    try:
        await asyncio.wait_for(both_started.wait(), 1)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert cancelled == {"selected", "extra-0"}
        assert provider.close_count == 1
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_automatic_web_search_can_search_again_after_completed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(
        _provider_tool_events(arguments={"query": "first"})
    )

    async def fake_search(self, query: str) -> list[WebSearchResult]:
        return [WebSearchResult(title=query, url=f"https://example.com/{query}")]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    service = _automatic_search_service(provider)
    first = await service.create(_automatic_search_request(stream=False))
    assert isinstance(first, JSONResponse)
    first_content = _json_body(first)["content"]

    provider.events = _provider_tool_events(arguments={"query": "second"})
    second_request = _automatic_search_request(stream=False)
    second_request.messages.extend(
        [
            Message.model_validate({"role": "assistant", "content": first_content}),
            Message(role="user", content="Search for a second thing"),
        ]
    )
    second = await service.create(second_request)

    assert isinstance(second, JSONResponse)
    assert second.status_code == 200
    assert _json_body(second)["content"][1]["content"][0]["title"] == "second"
    assert len(provider.requests) == 2
    assert all(
        getattr(block, "type", None)
        not in {"server_tool_use", "web_search_tool_result"}
        for block in provider.requests[1].messages[1].content
    )
    assert "https://example.com/first" in str(provider.requests[1].messages[1].content)
    assert getattr(second_request.messages[1].content[0], "type", None) == (
        "server_tool_use"
    )


def test_automatic_web_search_history_projects_for_each_provider_transport() -> None:
    request = _automatic_search_request()
    request.messages.extend(
        [
            Message.model_validate(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_prior",
                            "name": "web_search",
                            "input": {"query": "prior"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_prior",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "title": "Prior hit",
                                    "url": "https://example.com/prior",
                                    "encrypted_content": "opaque",
                                }
                            ],
                        },
                    ],
                }
            ),
            Message(role="user", content="Search again"),
        ]
    )

    plan = plan_automatic_web_search(request, web_tools_enabled=True)

    assert plan is not None
    chat = AnthropicToOpenAIConverter.convert_messages(plan.request.messages)
    responses = build_responses_provider_request(
        plan.request, reasoning=ReasoningPolicy.off()
    )
    native = build_native_messages_request(
        plan.request,
        options=NativeMessagesOptions(model=plan.request.model, max_tokens=100),
    ).body
    for wire in (chat, responses, native):
        serialized = json.dumps(wire)
        assert "https://example.com/prior" in serialized
        assert '"opaque"' not in serialized
        assert "server_tool_use" not in serialized
    assert getattr(request.messages[1].content[0], "type", None) == ("server_tool_use")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        _provider_tool_events(arguments={"query": ""}),
        _provider_tool_events(name="unexpected_tool"),
    ],
    ids=["blank-query", "wrong-tool"],
)
async def test_automatic_web_search_rejects_malformed_provider_selection(
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedSelectionProvider(events)
    search = AsyncMock()
    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.search", search)
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
    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.search", search)
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
    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.search", search)
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


@pytest.mark.asyncio
async def test_web_fetch_pins_validated_addresses_for_every_redirect(monkeypatch):
    addresses = iter(["8.8.8.8", "1.1.1.1"])
    resolved_hosts = []

    def resolve(host, port, **kwargs):
        resolved_hosts.append(host)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (next(addresses), port),
            )
        ]

    monkeypatch.setattr(web_egress.socket, "getaddrinfo", resolve)
    redirect = _aiohttp_response(
        302, url="https://first.example/", location="https://next.example/", body=b""
    )
    final = _aiohttp_response(200, url="https://next.example/", body=b"done")
    client_cm, _ = _aiohttp_client_session_patch(redirect, final)
    connectors = []

    def connector(**kwargs):
        value = MagicMock()
        value.close = AsyncMock()
        connectors.append((kwargs["resolver"], value))
        return value

    with (
        patch(
            "free_claude_code.runtime.web_tools.client.ClientSession",
            return_value=client_cm,
        ),
        patch(
            "free_claude_code.runtime.web_tools.client.TCPConnector",
            side_effect=connector,
        ),
    ):
        result = await HTTPWebToolsClient().fetch(
            "https://first.example/", egress=_STRICT_EGRESS
        )

    assert result.data == "done"
    assert resolved_hosts == ["first.example", "next.example"]
    for (resolver, value), expected in zip(
        connectors, ["8.8.8.8", "1.1.1.1"], strict=True
    ):
        pinned = await resolver.resolve("ignored.example", 443)
        assert [row["host"] for row in pinned] == [expected]
        value.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_web_fetch_follows_redirect_when_each_hop_is_allowed():
    res_redirect = _aiohttp_response(
        302, url="http://8.8.8.8/start", location="/final", body=b""
    )
    res_ok = _aiohttp_response(200, url="http://8.8.8.8/final", body=b"hello world")
    client_cm, session = _aiohttp_client_session_patch(res_redirect, res_ok)
    with patch(
        "free_claude_code.runtime.web_tools.client.ClientSession",
        return_value=client_cm,
    ):
        out = await HTTPWebToolsClient().fetch(
            "http://8.8.8.8/start", egress=_STRICT_EGRESS
        )

    assert out.data == "hello world"
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_run_web_fetch_truncates_large_body_to_byte_cap(monkeypatch):
    huge = b"x" * 5000
    res_ok = _aiohttp_response(200, url="http://8.8.8.8/big", body=huge)
    client_cm, _ = _aiohttp_client_session_patch(res_ok)
    monkeypatch.setattr(web_tool_constants, "_MAX_WEB_FETCH_RESPONSE_BYTES", 100)
    with patch(
        "free_claude_code.runtime.web_tools.client.ClientSession",
        return_value=client_cm,
    ):
        out = await HTTPWebToolsClient().fetch(
            "http://8.8.8.8/big", egress=_STRICT_EGRESS
        )

    assert len(out.data) <= 100
    assert out.data == "x" * 100


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
            "free_claude_code.runtime.web_tools.client.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation),
    ):
        await HTTPWebToolsClient().fetch("http://8.8.8.8/start", egress=_STRICT_EGRESS)

    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_run_web_fetch_redirect_without_location_raises():
    res_bad = _aiohttp_response(302, url="http://8.8.8.8/here", body=b"")
    client_cm, _ = _aiohttp_client_session_patch(res_bad)
    with (
        patch(
            "free_claude_code.runtime.web_tools.client.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation, match="missing Location"),
    ):
        await HTTPWebToolsClient().fetch("http://8.8.8.8/here", egress=_STRICT_EGRESS)


@pytest.mark.asyncio
async def test_run_web_fetch_excess_redirects_raises():
    res1 = _aiohttp_response(302, url="http://8.8.8.8/a", location="/b", body=b"")
    res2 = _aiohttp_response(302, url="http://8.8.8.8/b", location="/c", body=b"")
    client_cm, _ = _aiohttp_client_session_patch(res1, res2)
    with (
        patch(
            "free_claude_code.runtime.web_tools.constants._MAX_WEB_FETCH_REDIRECTS", 1
        ),
        patch(
            "free_claude_code.runtime.web_tools.client.ClientSession",
            return_value=client_cm,
        ),
        pytest.raises(WebFetchEgressViolation, match="exceeded maximum redirects"),
    ):
        await HTTPWebToolsClient().fetch("http://8.8.8.8/a", egress=_STRICT_EGRESS)


@pytest.mark.asyncio
async def test_streams_web_search_server_tool_result(monkeypatch):
    async def fake_search(self, query: str) -> list[WebSearchResult]:
        assert query == "DeepSeek V4 model release 2026"
        return [
            WebSearchResult(title="DeepSeek V4 Released", url="https://example.com/v4")
        ]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
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
            async for event in _local_tool_body(
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
    async def fake_search(self, _query: str) -> list[WebSearchResult]:
        return [
            WebSearchResult(title="DeepSeek V4 Released", url="https://example.com/v4")
        ]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = AsyncMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(
            settings,
            _PROVIDER_IDS[0],
            provider_model="upstream-model",
        ),
        web_tools=StubWebToolsClient(),
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
    async def fake_search(self, _query: str) -> list[WebSearchResult]:
        return [
            WebSearchResult(title="DeepSeek V4 Released", url="https://example.com/v4")
        ]

    monkeypatch.setattr(
        "tests.web_tools_support.StubWebToolsClient.search", fake_search
    )
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = AsyncMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(settings, _PROVIDER_IDS[0]),
        web_tools=StubWebToolsClient(),
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

    async def fake_fetch(
        self, url: str, *, egress: WebFetchEgressPolicy
    ) -> WebFetchResult:
        assert url == target
        return WebFetchResult(url=url, title="T", media_type="text/plain", data="x")

    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.fetch", fake_fetch)
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
            async for event in _local_tool_body(
                request, input_tokens=1, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    assert target in raw


@pytest.mark.asyncio
async def test_service_aggregates_forced_web_fetch_when_stream_false(monkeypatch):
    async def fake_fetch(
        self, url: str, *, egress: WebFetchEgressPolicy
    ) -> WebFetchResult:
        return WebFetchResult(
            url=url,
            title="Example Article",
            media_type="text/plain",
            data="Article body",
        )

    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.fetch", fake_fetch)
    settings = Settings.model_validate({"ENABLE_WEB_SERVER_TOOLS": True})
    provider_resolver = AsyncMock()
    service = MessagesHandler(
        settings,
        provider_resolver=provider_resolver,
        model_router=FixedProviderModelRouter(settings, _PROVIDER_IDS[0]),
        web_tools=StubWebToolsClient(),
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
    async def fake_fetch(
        self, url: str, *, egress: WebFetchEgressPolicy
    ) -> WebFetchResult:
        assert url == "https://example.com/article"
        return WebFetchResult(
            url=url,
            title="Example Article",
            media_type="text/plain",
            data="Article body",
        )

    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.fetch", fake_fetch)
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
            async for event in _local_tool_body(
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

    async def boom(self, _url: str, *, egress: WebFetchEgressPolicy) -> WebFetchResult:
        raise ValueError(secret)

    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.fetch", boom)
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

    with patch(
        "free_claude_code.application.web_tools.service.logger.warning"
    ) as log_warn:
        raw = "".join(
            [
                event
                async for event in _local_tool_body(
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
    async def boom(self, _url: str, *, egress: WebFetchEgressPolicy) -> WebFetchResult:
        raise OSError(5, "oops")

    monkeypatch.setattr("tests.web_tools_support.StubWebToolsClient.fetch", boom)
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
            async for event in _local_tool_body(
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
        provider_resolver=AsyncMock(side_effect=lambda _: MagicMock()),
        model_router=FixedProviderModelRouter(settings, provider_id),
        web_tools=StubWebToolsClient(),
    )
    request = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[Message(role="user", content="q")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )
    with pytest.raises(InvalidRequestError, match="ENABLE_WEB_SERVER_TOOLS=false"):
        await service.create(request)


@pytest.mark.asyncio
async def test_automatic_selection_stays_private_until_provider_finishes() -> None:
    release = asyncio.Event()
    provider = ScriptedSelectionProvider(
        _provider_text_events("declined"), wait_for=release
    )
    task = asyncio.create_task(
        _automatic_search_service(provider).create(_automatic_search_request())
    )
    try:
        await asyncio.wait_for(provider.started.wait(), 1)
        assert not task.done()
        release.set()
        response = await asyncio.wait_for(task, 1)
        assert isinstance(response, StreamingResponse)
        assert await _streaming_body_text(response) == "".join(provider.events)
        assert provider.close_count == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["web_search", "web_fetch"])
@pytest.mark.parametrize("close_before_outbound", [False, True])
async def test_local_tool_start_precedes_outbound_and_cancellation_closes_operation(
    monkeypatch, tool_name: str, close_before_outbound: bool
) -> None:
    entered, closed = asyncio.Event(), asyncio.Event()

    async def blocked(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(
        f"tests.web_tools_support.StubWebToolsClient.{tool_name.removeprefix('web_')}",
        blocked,
    )
    tool_type = (
        "web_search_20250305" if tool_name == "web_search" else "web_fetch_20250910"
    )
    request = MessagesRequest(
        model="public-model",
        messages=[Message(role="user", content="https://example.com/")],
        tools=[Tool(name=tool_name, type=tool_type)],
        tool_choice={"type": "tool", "name": tool_name},
    )
    body = _local_tool_body(request, 7, web_fetch_egress=_STRICT_EGRESS)
    pending = None
    try:
        prefix = [await anext(body) for _ in range(3)]
        assert [event.event for event in parse_sse_text("".join(prefix))] == [
            "message_start",
            "content_block_start",
            "content_block_stop",
        ]
        assert not entered.is_set()
        if close_before_outbound:
            await body.aclose()
            assert not entered.is_set()
            return
        pending = asyncio.create_task(anext(body))
        await asyncio.wait_for(entered.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert closed.is_set()
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await body.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "automatic"),
    [("web_search", False), ("web_fetch", False), ("web_search", True)],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_local_failures_remain_completed_tool_results(
    monkeypatch, tool_name: str, automatic: bool, stream: bool
) -> None:
    async def fail(*_args, **_kwargs):
        raise OSError("private upstream detail")

    monkeypatch.setattr(
        f"tests.web_tools_support.StubWebToolsClient.{tool_name.removeprefix('web_')}",
        fail,
    )
    provider = ScriptedSelectionProvider(
        _provider_tool_events(arguments={"query": "query"})
    )
    handler = _automatic_search_service(provider)
    request = _automatic_search_request().model_copy(update={"stream": stream})
    if not automatic:
        tool_type = (
            "web_search_20250305" if tool_name == "web_search" else "web_fetch_20250910"
        )
        request = request.model_copy(
            update={
                "tools": [Tool(name=tool_name, type=tool_type)],
                "tool_choice": {"type": "tool", "name": tool_name},
            }
        )
    response = await handler.create(request)
    if stream:
        assert isinstance(response, StreamingResponse)
        raw = await _streaming_body_text(response)
        events = parse_sse_text(raw)
        assert_anthropic_stream_contract(events)
        assert not any(event.event == "error" for event in events)
        result = next(
            event.data["content_block"]
            for event in events
            if event.event == "content_block_start" and event.data["index"] == 1
        )
        usage = next(
            event.data["usage"] for event in events if event.event == "message_delta"
        )
        assert events[-1].event == "message_stop"
    else:
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        message = _json_body(response)
        result, usage = message["content"][1], message["usage"]
        assert message["stop_reason"] == "end_turn"
    assert result["type"] == f"{tool_name}_tool_result"
    expected_error = (
        "web_search_tool_result_error"
        if tool_name == "web_search"
        else "web_fetch_tool_error"
    )
    assert result["content"] == {"type": expected_error, "error_code": "unavailable"}
    assert usage["server_tool_use"] == {f"{tool_name}_requests": 1}
    assert len(provider.requests) == int(automatic)


@pytest.mark.parametrize("tool_name", ["web_search", "web_fetch"])
def test_forced_tools_keep_permissive_matching(tool_name: str) -> None:
    tool_type = (
        "web_search_20250305" if tool_name == "web_search" else "web_fetch_20250910"
    )
    request = MessagesRequest(
        model="model",
        messages=[
            Message(
                role="assistant",
                content=[
                    ContentBlockServerToolUse(
                        type="server_tool_use", id="old", name=tool_name, input={}
                    )
                ],
            )
        ],
        tools=[
            Tool(name=tool_name, type=tool_type, max_uses=0, unsupported_option=True),
            Tool(name="client_tool", input_schema={"type": "object"}),
        ],
        tool_choice={"type": "tool", "name": tool_name, "extra": True},
    )
    assert is_web_server_tool_request(request)
    assert plan_automatic_web_search(request, web_tools_enabled=True) is None
    assert unsupported_server_tool_error(request, web_tools_enabled=True) is None
