"""Model-selected local web-tool regressions; all network and models are mocked."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.api.web_tools import execution
from free_claude_code.api.web_tools.request import unsupported_server_tool_error
from free_claude_code.application.execution import ProviderExecutor, WireApi
from free_claude_code.application.routing import RoutedMessagesRequest
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    ContentBlockToolResult,
    Message,
    MessagesRequest,
    Tool,
)
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.json_types import JsonObject


class ScriptedExecutor(ProviderExecutor):
    def __init__(self, turns: list[list[JsonObject]], *, stop: str = "end_turn"):
        super().__init__(MagicMock())
        self.turns = turns
        self.requests: list[MessagesRequest] = []
        self.routes: list[RoutedMessagesRequest] = []
        self.stop = stop

    def stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: object,
        request_id: str,
    ) -> AsyncIterator[str]:
        self.requests.append(routed.request.model_copy(deep=True))
        self.routes.append(routed)
        blocks = self.turns.pop(0)

        async def body() -> AsyncIterator[str]:
            yield format_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_internal",
                        "type": "message",
                        "role": "assistant",
                        "model": routed.resolved.original_model,
                        "content": [],
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            )
            for index, block in enumerate(blocks):
                yield format_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": block,
                    },
                )
                yield format_sse_event(
                    "content_block_stop",
                    {
                        "type": "content_block_stop",
                        "index": index,
                    },
                )
            yield format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self.stop
                        if not any(block["type"] == "tool_use" for block in blocks)
                        else "tool_use",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": 5},
                },
            )
            yield format_sse_event("message_stop", {"type": "message_stop"})

        return body()


def search_call(
    name: str = "web_search", *, call_id: str = "toolu_search"
) -> JsonObject:
    key = "query" if name == "web_search" else "url"
    value = "official API docs" if name == "web_search" else "https://example.com/docs"
    return {"type": "tool_use", "id": call_id, "name": name, "input": {key: value}}


def web_request(name: str = "web_search", **updates: object) -> MessagesRequest:
    values: dict[str, object] = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 100,
        "messages": [Message(role="user", content="Find official documentation")],
        "tools": [
            Tool(
                name=name,
                type=(
                    "web_search_20250305"
                    if name == "web_search"
                    else "web_fetch_20250910"
                ),
            )
        ],
    }
    values.update(updates)
    return MessagesRequest.model_validate(values)


def handler(executor: ProviderExecutor) -> MessagesHandler:
    settings = Settings.model_validate(
        {
            "MODEL": "azure_openai/test-model",
            "ENABLE_WEB_SERVER_TOOLS": True,
        }
    )
    return MessagesHandler(settings, MagicMock(), provider_executor=executor)


@pytest.fixture
def local_http(monkeypatch):
    search = AsyncMock(
        return_value=[{"title": "Official docs", "url": "https://example.com/docs"}]
    )
    fetch = AsyncMock(
        return_value={
            "title": "Official docs",
            "url": "https://example.com/docs",
            "media_type": "text/plain",
            "data": "Verified documentation",
        }
    )
    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", fetch)
    return search, fetch


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["web_search", "web_fetch"])
@pytest.mark.parametrize("stream", [False, True])
async def test_listed_web_tools_execute_locally_and_synthesize(
    local_http, name, stream
):
    executor = ScriptedExecutor(
        [[search_call(name)], [{"type": "text", "text": "https://example.com/docs"}]]
    )
    request = web_request(name, stream=stream)
    response = await handler(executor).create(request)
    if stream:
        assert isinstance(response, StreamingResponse)
        raw = "".join([str(part) async for part in response.body_iterator])
        events = parse_sse_text(raw)
        assert_anthropic_stream_contract(events)
        blocks = [
            event.data["content_block"]
            for event in events
            if event.event == "content_block_start"
        ]
        body = next(
            event.data["message"] for event in events if event.event == "message_start"
        )
        usage = next(
            event.data["usage"] for event in events if event.event == "message_delta"
        )
        tool_input = next(
            event.data["delta"]["partial_json"]
            for event in events
            if event.event == "content_block_delta"
            and event.data["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(tool_input) == search_call(name)["input"]
    else:
        assert isinstance(response, JSONResponse)
        body = json.loads(bytes(response.body))
        blocks = body["content"]
        usage = body["usage"]
        assert blocks[0]["input"] == search_call(name)["input"]
    assert [block["type"] for block in blocks] == [
        "server_tool_use",
        f"{name}_tool_result",
        "text",
    ]
    assert blocks[0]["id"] == blocks[1]["tool_use_id"]
    assert body["model"] == request.model
    assert usage["output_tokens"] == 10
    assert usage["server_tool_use"][f"{name}_requests"] == 1
    assert len(executor.requests) == 2
    assert executor.requests[1].max_tokens == 95
    internal_tools = executor.requests[0].tools
    assert internal_tools is not None
    assert internal_tools[0].type is None
    continuation = executor.requests[1].messages[-1].content
    assert isinstance(continuation, list)
    assert isinstance(continuation[0], ContentBlockToolResult)
    assert executor.routes[0].resolved == executor.routes[1].resolved
    assert executor.routes[0].reasoning == executor.routes[1].reasoning
    (local_http[0] if name == "web_search" else local_http[1]).assert_awaited_once()
    assert request.tools is not None
    assert request.tools[0].type is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [None, {"type": "auto"}, {"type": "none"}])
async def test_model_can_answer_without_web_calls(local_http, choice):
    executor = ScriptedExecutor([[{"type": "text", "text": "No search required."}]])
    response = await handler(executor).create(web_request(tool_choice=choice))
    assert isinstance(response, JSONResponse)
    assert (
        json.loads(bytes(response.body))["content"][0]["text"] == "No search required."
    )
    local_http[0].assert_not_awaited()
    assert executor.requests[0].tool_choice == choice


@pytest.mark.asyncio
async def test_none_rejects_unexpected_model_tool_call_without_network(local_http):
    response = await handler(ScriptedExecutor([[search_call()]])).create(
        web_request(tool_choice={"type": "none"})
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    local_http[0].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_value", [{}, {"query": ""}, {"query": 5}, {"query": "q", "command": "run"}]
)
async def test_invalid_arguments_have_no_web_side_effects(local_http, input_value):
    call = search_call()
    call["input"] = input_value
    response = await handler(ScriptedExecutor([[call]])).create(web_request())
    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    local_http[0].assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_calls_are_rejected_before_any_network(local_http):
    response = await handler(ScriptedExecutor([[search_call(), search_call()]])).create(
        web_request()
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    local_http[0].assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_call_is_rejected_before_network(local_http):
    call = search_call()
    call["name"] = "shell"
    response = await handler(ScriptedExecutor([[search_call(), call]])).create(
        web_request()
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    local_http[0].assert_not_awaited()


@pytest.mark.asyncio
async def test_local_failure_is_safe_evidence_not_success(monkeypatch):
    secret = "private-secret"
    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search",
        AsyncMock(side_effect=OSError(secret)),
    )
    executor = ScriptedExecutor(
        [[search_call()], [{"type": "text", "text": "Retrieval failed."}]]
    )
    response = await handler(executor).create(web_request())
    assert isinstance(response, JSONResponse)
    assert secret not in bytes(response.body).decode()
    continuation = executor.requests[-1].messages[-1].content
    assert isinstance(continuation, list)
    assert isinstance(continuation[0], ContentBlockToolResult)
    assert continuation[0].model_extra is not None
    assert continuation[0].model_extra["is_error"] is True


@pytest.mark.asyncio
async def test_domain_filters_remove_unapproved_search_links(local_http):
    local_http[0].return_value = [
        {"title": "Allowed", "url": "https://docs.example.com/page"},
        {"title": "Lookalike", "url": "https://example.com.attacker.net/page"},
    ]
    executor = ScriptedExecutor(
        [[search_call()], [{"type": "text", "text": "https://docs.example.com/page"}]]
    )
    response = await handler(executor).create(
        web_request(
            tools=[
                Tool(
                    name="web_search",
                    type="web_search_20250305",
                    allowed_domains=["example.com"],
                )
            ]
        )
    )
    assert isinstance(response, JSONResponse)
    content = json.loads(bytes(response.body))["content"][1]["content"]
    assert len(content) == 1
    assert content[0]["url"] == "https://docs.example.com/page"


@pytest.mark.parametrize(
    "choice", [None, {"type": "auto"}, {"type": "any"}, {"type": "none"}]
)
def test_enabled_web_only_choices_supported(choice):
    assert (
        unsupported_server_tool_error(
            web_request(tool_choice=choice), web_tools_enabled=True
        )
        is None
    )


@pytest.mark.parametrize(
    "tool",
    [
        Tool(name="other", type="web_search_20250305"),
        Tool(name="web_search", type="web_search_20260209"),
        Tool(
            name="web_search",
            type="web_search_20250305",
            user_location={"country": "IN"},
        ),
        Tool(name="web_search", type="web_search_20250305", max_uses=True),
        Tool(
            name="web_search",
            type="web_search_20250305",
            allowed_domains=["https://example.com"],
        ),
    ],
)
def test_unsupported_capabilities_are_not_silently_dropped(tool):
    error = unsupported_server_tool_error(
        web_request(tools=[tool]), web_tools_enabled=True
    )
    assert error is not None


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "auto", "disable_parallel_tool_use": "true"},
        {"type": "auto", "disable_parallel_tool_use": 1},
        {"type": "none", "name": "web_search"},
    ],
)
def test_invalid_tool_choice_fields_rejected(choice):
    assert (
        unsupported_server_tool_error(
            web_request(tool_choice=choice), web_tools_enabled=True
        )
        is not None
    )


def test_mixed_tools_still_rejected():
    request = web_request(
        tools=[
            Tool(name="web_search", type="web_search_20250305"),
            Tool(name="Read", input_schema={}),
        ]
    )
    assert unsupported_server_tool_error(request, web_tools_enabled=True) is not None


def test_forced_missing_definition_rejected():
    request = web_request(tools=[], tool_choice={"type": "tool", "name": "web_search"})
    assert unsupported_server_tool_error(request, web_tools_enabled=True) is not None


@pytest.mark.asyncio
async def test_timeout_closes_internal_stream(monkeypatch):
    closed: list[bool] = []

    class HangingExecutor(ScriptedExecutor):
        def stream(self, routed, **kwargs):
            async def body():
                try:
                    await asyncio.Event().wait()
                    yield ""
                finally:
                    closed.append(True)

            return body()

    monkeypatch.setattr(execution, "_EXECUTION_TIMEOUT_SECONDS", 0.01)
    response = await handler(HangingExecutor([])).create(web_request())
    assert isinstance(response, JSONResponse)
    assert response.status_code == 504
    assert closed == [True]
