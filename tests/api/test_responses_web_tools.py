"""Codex web actions execute through the real translated transports."""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.responses_execution import ResponsesBinding
from free_claude_code.application.web_tools.responses import ResponsesWebTools
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.sse import parse_sse_text
from free_claude_code.core.web_tools import WebFetchResult, WebSearchResult
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.api.support import create_test_app
from tests.application.test_execution import _routed_responses_request
from tests.application.test_responses_web_workflow import WebClient
from tests.application.test_responses_web_workflow import request as web_request
from tests.core.openai_responses.test_client_tool_discovery import SEARCH
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_anthropic_messages_transport import (
    Endpoint,
    Wire,
    _events,
    _sse,
    _transport,
)


async def _web_events(provider, web):
    async def resolve(_):
        return provider

    original = web_request(
        tools=[
            {"type": "web_search"},
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
        ]
    )
    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=60,
        responses_token_counter=lambda _: 1,
        responses_web_tools=ResponsesWebTools(web),
    )
    stream = executor.stream_responses(
        replace(
            _routed_responses_request(),
            request=original,
            reasoning=ReasoningPolicy.provider_default(),
        ),
        raw_log_payload={},
        request_id="real-failure",
    )
    try:
        events = parse_sse_text("".join([frame async for frame in stream]))
    finally:
        assert isinstance(stream, AsyncCloseable)
        await stream.aclose()
    return events


async def _web_failure_events(provider):
    web = WebClient()
    events = await _web_events(provider, web)
    assert events[-1].event == "response.failed"
    assert not web.searches
    done = [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ]
    assert done == events[-1].data["response"]["output"]
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "incomplete", "failed"])
async def test_real_messages_terminal_survives_timeout_closing_provider_stream(
    monkeypatch, outcome
):
    timeouts = []
    releasing = asyncio.Event()
    closes = []

    def controlled_timeout(deadline):
        timeout = asyncio.timeout(None)
        timeouts.append(timeout)
        return timeout

    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )

    class StalledClose(Wire):
        async def aclose(self):
            closes.append("body")
            releasing.set()
            try:
                await asyncio.Event().wait()
            finally:
                await super().aclose()

    text = "x" * 70_000 if outcome == "failed" else "Finished answer"
    chunks: list[bytes | Exception]
    if outcome == "failed":
        chunks = [
            _sse(*_events(text)[:3]),
            httpx.RemoteProtocolError("original provider failure"),
        ]
    else:
        chunks = [
            _sse(
                *_events(text, "max_tokens" if outcome == "incomplete" else "end_turn")
            )
        ]
    body = StalledClose(chunks)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    web = WebClient()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        consuming = asyncio.create_task(
            _web_events(MessagesProvider(_transport(client)), web)
        )
        try:
            await asyncio.wait_for(releasing.wait(), 5)
            timeouts[-1].reschedule(asyncio.get_running_loop().time())
            events = await asyncio.wait_for(consuming, 5)
        finally:
            consuming.cancel()
            await asyncio.gather(consuming, return_exceptions=True)
    terminals = [
        event
        for event in events
        if event.event
        in ("response.completed", "response.failed", "response.incomplete")
    ]
    assert [event.event for event in terminals] == [f"response.{outcome}"]
    final = terminals[0].data["response"]
    assert final["output"][0]["content"][0]["text"] == text
    assert [
        event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    ] == final["output"]
    if outcome == "failed":
        assert final["error"]["type"] == "api_error"
        assert "original provider failure" in final["error"]["message"]
        assert final["usage"] is None
    else:
        assert final["usage"]["input_tokens"] == 3
        assert final["usage"]["output_tokens"] == 2
        if outcome == "incomplete":
            assert final["incomplete_details"] == {"reason": "max_output_tokens"}
    assert not web.searches
    assert len(calls) == 1
    assert body.closed and closes == ["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["text", "reasoning", "ordinary-tool", "web-tool"])
async def test_real_messages_failure_preserves_and_closes_partial_output(partial):
    text = "x" * 70_000
    events: list[JsonObject] = _events(text)[:3]
    if partial == "reasoning":
        events[1]["content_block"] = {"type": "thinking", "thinking": ""}
        events[2]["delta"] = {"type": "thinking_delta", "thinking": text}
    elif partial.endswith("tool"):
        events = [
            *_events(text)[:4],
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_partial",
                    "name": "read_file" if partial == "ordinary-tool" else "fcc_web",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":',
                },
            },
        ]
    body = Wire([_sse(*events), httpx.ReadTimeout("stalled")])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        public = await _web_failure_events(MessagesProvider(_transport(client)))
    assert body.closed and len(calls) == 1
    final = public[-1].data["response"]
    assert final["usage"] is None
    assert final["output"][-1]["status"] == "incomplete"
    assert final["output"][0]["content"][0]["text"] == text
    assert all(
        event.event != "response.function_call_arguments.done" for event in public
    )
    if partial == "ordinary-tool":
        assert final["output"][-1]["arguments"] == ""
        assert final["output"][-1]["call_id"] == "call_partial"
    elif partial == "web-tool":
        assert final["output"][-1]["type"] == "web_search_call"


@pytest.mark.asyncio
async def test_real_chat_failure_closes_call_omitted_from_terminal_snapshot():
    def handler(request):
        data = chat_wire(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_partial",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":'},
                    }
                ],
            },
            "tool_calls",
        )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=data
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://chat.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        provider = OpenAIChatProvider(
            make_provider_config("test", "https://chat.invalid/v1"),
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    provider_name="TEST",
                    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
                ),
                NO_REASONING,
            ),
            admission=immediate_admission(),
            client=client,
        )
        events = await _web_failure_events(provider)
    final = events[-1].data["response"]
    assert len(final["output"]) == 1
    assert final["output"][0]["status"] == "incomplete"
    assert final["output"][0]["call_id"] == "call_partial"
    assert all(
        event.event != "response.function_call_arguments.done" for event in events
    )


class MessagesProvider:
    def __init__(self, transport):
        self.transport = transport

    @asynccontextmanager
    async def bind_responses(self, request, **kwargs):
        yield ResponsesBinding("messages", partial(self.stream_responses, **kwargs))

    def stream_responses(self, request, **kwargs):
        kwargs.pop("input_tokens", None)
        kwargs.pop("request_headers", None)
        return self.transport.stream_responses(
            request, endpoint_context=Endpoint(), **kwargs
        )


def chat_wire(message, finish):
    chunks = [
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "delta": message, "finish_reason": None}],
        },
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
    ]
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )


@pytest.mark.parametrize("egress", ["chat", "messages"])
@pytest.mark.parametrize("full", [False, True])
def test_codex_search_reaches_model_and_returns_one_response(egress, full):
    bodies = []
    actions = [{"action": "search", "query": "python docs"}]
    if full:
        actions += [
            {"action": "open_page", "url": "https://docs.python.org/3/"},
            {
                "action": "find_in_page",
                "url": "https://docs.python.org/3/",
                "pattern": "needle",
            },
        ]

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert "client_metadata" not in body
        declarations = [tool.get("function", tool) for tool in body.get("tools", [])]
        web = next(
            (
                tool
                for tool in declarations
                if tool.get("name", "").startswith("fcc_web")
            ),
            None,
        )
        if web is not None and len(bodies) <= len(actions):
            if egress == "chat":
                data = chat_wire(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_web",
                                "type": "function",
                                "function": {
                                    "name": web["name"],
                                    "arguments": json.dumps(actions[len(bodies) - 1]),
                                },
                            }
                        ],
                    },
                    "tool_calls",
                )
            else:
                events = _events()
                events[1] = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "call_web",
                        "name": web["name"],
                        "input": {},
                    },
                }
                events[2] = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(actions[len(bodies) - 1]),
                    },
                }
                events[-2] = {**events[-2], "delta": {"stop_reason": "tool_use"}}
                data = _sse(*events)
        else:
            answer = "Read [Python docs](https://docs.python.org/3/)."
            data = (
                chat_wire({"role": "assistant", "content": answer}, "stop")
                if egress == "chat"
                else _sse(*_events(answer))
            )
        if egress == "chat":
            assert isinstance(data, str)
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, text=data
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Wire([data])
        )

    if egress == "chat":
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://chat.invalid/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        provider = OpenAIChatProvider(
            make_provider_config("test", "https://chat.invalid/v1"),
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    provider_name="TEST",
                    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
                ),
                NO_REASONING,
            ),
            admission=immediate_admission(),
            client=client,
        )
    else:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = MessagesProvider(_transport(client))
    app = create_test_app()
    search = AsyncMock(
        return_value=[WebSearchResult("Python docs", "https://docs.python.org/3/")]
    )
    fetch = AsyncMock(
        return_value=WebFetchResult(
            "https://docs.python.org/3/",
            "Python docs",
            "text/plain",
            "A page with needle.",
        )
    )
    try:
        with (
            patch(
                "free_claude_code.api.routes.resolve_provider", return_value=provider
            ),
            patch.object(app.state.services.web_tools, "search", search),
            patch.object(app.state.services.web_tools, "fetch", fetch),
            TestClient(app) as api,
        ):
            response = api.post(
                "/v1/responses",
                json={
                    "model": "nvidia_nim/test",
                    "input": "Find Python docs",
                    "tools": [
                        {
                            "type": "function",
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                        {
                            "type": "web_search",
                            "external_web_access": full,
                            "search_content_types": ["text", "image"],
                        },
                    ],
                    "tool_choice": "auto",
                    "include": ["reasoning.encrypted_content"],
                    "client_metadata": {
                        "session_id": "native-session",
                        "x-codex-turn-metadata": '{"turn_id":"test"}',
                    },
                },
            )
        assert response.status_code == 200, response.text
        search.assert_awaited_once_with("python docs")
        assert len(bodies) == len(actions) + 1, response.text[-3000:]
        assert "https://docs.python.org/3/" in json.dumps(bodies[1])
        events = parse_sse_text(response.text)
        assert sum(event.event == "response.created" for event in events) == 1
        assert sum(event.event == "response.completed" for event in events) == 1
        result = events[-1].data["response"]
        assert any(item["type"] == "web_search_call" for item in result["output"])
        assert "fcc_web" not in response.text
        assert any(item["type"] == "message" for item in result["output"])
        assert [
            item["action"]["type"]
            for item in result["output"]
            if item["type"] == "web_search_call"
        ] == [item["action"] for item in actions]
        assert fetch.await_count == int(full)
        if full:
            assert "needle" in json.dumps(bodies[-1])
    finally:
        asyncio.run(
            client.close() if isinstance(client, AsyncOpenAI) else client.aclose()
        )


def test_messages_web_search_hands_custom_and_discovery_tools_back_to_codex():
    bodies = []
    patch_text = "*** Begin Patch\n*** End Patch"

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            definitions = body["tools"]
            web_name = next(
                tool["name"]
                for tool in definitions
                if "Search the web" in tool.get("description", "")
            )
            patch_name = next(
                tool["name"]
                for tool in definitions
                if "input" in tool["input_schema"].get("properties", {})
            )
            search_name = next(
                tool["name"]
                for tool in definitions
                if tool.get("description") == SEARCH["description"]
            )
            calls = [
                (web_name, {"action": "search", "query": "python docs"}),
                (patch_name, {"input": patch_text}),
                (search_name, {"query": "read file"}),
            ]
            events = _events()[:1]
            for index, (name, args) in enumerate(calls):
                events += [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": f"call_{index}",
                            "name": name,
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(args),
                        },
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            events += [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 10},
                },
                {"type": "message_stop"},
            ]
        else:
            text = json.dumps(body)
            assert "https://docs.python.org/3/" in text
            assert "Patch applied" in text
            assert any(tool["name"] == "fcc_web" for tool in body["tools"])
            assert any(tool["name"] == "fcc_web_1" for tool in body["tools"])
            results = body["messages"][-1]["content"]
            assert [part["type"] for part in results] == [
                "tool_result",
                "tool_result",
                "text",
            ]
            events = _events("Done.")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*events)]),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = MessagesProvider(_transport(client))
    app = create_test_app()
    search = AsyncMock(
        return_value=[WebSearchResult("Python docs", "https://docs.python.org/3/")]
    )
    payload: dict[str, Any] = {
        "model": "nvidia_nim/test",
        "input": [{"role": "user", "content": "Research and edit"}],
        "tools": [
            {
                "type": "namespace",
                "name": "editor",
                "tools": [
                    {
                        "type": "custom",
                        "name": "patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": "start: /.+/",
                        },
                    }
                ],
            },
            SEARCH,
            {"type": "web_search", "external_web_access": False},
        ],
        "tool_choice": "auto",
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "test-session",
    }
    try:
        with (
            patch(
                "free_claude_code.api.routes.resolve_provider", return_value=provider
            ),
            patch.object(app.state.services.web_tools, "search", search),
            TestClient(app) as api,
        ):
            first = api.post("/v1/responses", json=payload)
            assert first.status_code == 200
            response = parse_sse_text(first.text)[-1].data["response"]
            assert response["status"] == "completed", response
            assert len(bodies) == 1
            retained = deepcopy(response["output"])
            custom = next(
                item for item in retained if item["type"] == "custom_tool_call"
            )
            discovery = next(
                item for item in retained if item["type"] == "tool_search_call"
            )
            assert (custom["name"], custom["namespace"], custom["input"]) == (
                "patch",
                "editor",
                patch_text,
            )
            assert discovery["arguments"] == {"query": "read file"}
            # Codex retains typed action fields and the opaque carrier, not sources.
            for item in retained:
                if item["type"] == "web_search_call":
                    item["action"].pop("sources", None)
            payload["input"] = [
                *payload["input"],
                *retained,
                {
                    "type": "custom_tool_call_output",
                    "call_id": custom["call_id"],
                    "output": "Patch applied",
                },
                {
                    "type": "tool_search_output",
                    "call_id": discovery["call_id"],
                    "tools": [
                        {
                            "type": "function",
                            "name": "fcc_web",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            ]
            second = api.post("/v1/responses", json=payload)
            assert second.status_code == 200
            final = parse_sse_text(second.text)[-1].data["response"]
            assert final["status"] == "completed", final
            assert len(bodies) == 2
            search.assert_awaited_once_with("python docs")
    finally:
        asyncio.run(client.aclose())
