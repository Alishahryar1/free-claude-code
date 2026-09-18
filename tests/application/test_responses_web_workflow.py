"""Model/tool-loop contracts independent of a particular provider SDK."""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.responses_execution import ResponsesBinding
from free_claude_code.application.web_tools.responses import ResponsesWebTools
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.items import message_item
from free_claude_code.core.openai_responses.streaming.event_builders import (
    ResponseEventBuilder,
)
from free_claude_code.core.openai_responses.web_history import (
    prepare_web_history,
    replay_item,
)
from free_claude_code.core.sse import parse_sse_text
from free_claude_code.core.web_tools import WebFetchResult, WebSearchResult
from tests.application.test_execution import _routed_responses_request, _target
from tests.core.openai_responses.test_client_tool_discovery import SEARCH

URL = "https://docs.example.org/page"


class WebClient:
    def __init__(self):
        self.searches = []
        self.fetches = []

    async def search(self, query):
        self.searches.append(query)
        return [
            WebSearchResult("Documentation", URL, "An indexed documentation snippet.")
        ]

    async def fetch(self, url, *, egress):
        self.fetches.append((url, egress))
        return WebFetchResult(
            URL, "Documentation", "text/plain", "A page with the needle to find."
        )


def request(**extra):
    values = {
        "model": "test",
        "input": [{"role": "user", "content": "Research docs"}],
        "tools": [{"type": "web_search", "external_web_access": True}],
        "include": ["reasoning.encrypted_content", "web_search_call.action.sources"],
    }
    values.update(extra)
    return OpenAIResponsesRequest.model_validate(values)


def web_call(name, action, number=0):
    return {
        "type": "function_call",
        "id": f"fc_private_{number}",
        "call_id": f"call_{number}",
        "name": name,
        "arguments": json.dumps(action),
        "status": "completed",
    }


def wire(output, *, number=0, status="completed", usage=True):
    events = ResponseEventBuilder()
    response = {
        "id": f"resp_private_{number}",
        "object": "response",
        "created_at": 1,
        "model": "public",
        "status": "in_progress",
        "output": [],
    }
    yield events.response_created(response)
    for index, item in enumerate(output):
        yield events.output_item_added(index, {**item, "status": "in_progress"})
        if item["type"] == "message":
            yield events.content_part_added(item["id"], index)
            yield events.emit(
                "response.output_text.delta",
                {
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": item["content"][0]["text"],
                },
            )
            yield events.emit(
                "response.content_part.done",
                {
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": deepcopy(item["content"][0]),
                },
            )
        yield events.output_item_done(index, item)
    yield events.emit(
        f"response.{status}",
        {
            "response": {
                **response,
                "status": status,
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
                if usage
                else None,
            }
        },
    )


async def run(service, original, choose, *, egress="chat"):
    requests = []

    async def stream(turn, *, input_tokens):
        requests.append(turn.model_copy(deep=True))
        output = choose(turn, len(requests))
        if isinstance(output, Exception):
            raise output
        for chunk in wire(output, number=len(requests)):
            yield chunk

    class Provider:
        @asynccontextmanager
        async def bind_responses(self, turn, **kwargs):
            yield ResponsesBinding(egress, stream)

    async def resolve(_):
        return Provider()

    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=60,
        responses_web_tools=service,
        responses_token_counter=lambda turn: len(json.dumps(turn.model_dump())),
    )
    routed = replace(_routed_responses_request(), request=original)
    chunks = [
        chunk
        async for chunk in executor.stream_responses(
            routed, raw_log_payload={}, request_id="workflow-test"
        )
    ]
    return parse_sse_text("".join(chunks)), requests


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", ["chat", "messages"])
async def test_search_open_find_preserves_one_stream_sources_usage_and_request(egress):
    client = WebClient()
    original = request()
    before = original.model_dump()
    actions = [
        {"action": "search", "query": "docs"},
        {"action": "open_page", "url": URL},
        {"action": "find_in_page", "url": URL, "pattern": "needle"},
    ]

    def choose(turn, number):
        if number <= 3:
            return [web_call(turn.tools[0]["name"], actions[number - 1], number)]
        return [message_item("msg_answer", f"See [documentation]({URL}).", "completed")]

    events, turns = await run(
        ResponsesWebTools(client), original, choose, egress=egress
    )
    assert original.model_dump() == before
    assert client.searches == ["docs"] and len(client.fetches) == 1
    assert len(turns) == 4
    assert "needle" in json.dumps(turns[-1].input)
    assert [event.data["sequence_number"] for event in events] == list(
        range(len(events))
    )
    assert sum(event.event == "response.created" for event in events) == 1
    final = events[-1].data["response"]
    assert final["usage"] == {
        "input_tokens": 40,
        "output_tokens": 12,
        "total_tokens": 52,
    }
    assert final["tools"] == original.tools
    done = {
        event.data["output_index"]: event.data["item"]
        for event in events
        if event.event == "response.output_item.done"
    }
    assert [done[index] for index in sorted(done)] == final["output"]
    calls = [item for item in final["output"] if item["type"] == "web_search_call"]
    assert [item["action"]["type"] for item in calls] == [
        "search",
        "open_page",
        "find_in_page",
    ]
    assert calls[0]["action"]["sources"][0]["url"] == URL
    text = next(item for item in final["output"] if item["type"] == "message")[
        "content"
    ][0]
    annotation = text["annotations"][0]
    assert text["text"][annotation["start_index"] : annotation["end_index"]] == URL
    assert (
        len(
            [
                event
                for event in events
                if event.event == "response.output_text.annotation.added"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_cached_mode_never_fetches_fresh_pages():
    client = WebClient()
    original = request(tools=[{"type": "web_search", "external_web_access": False}])

    def choose(turn, number):
        if number == 1:
            return [
                web_call(turn.tools[0]["name"], {"action": "open_page", "url": URL})
            ]
        assert "cached_page_unavailable" in json.dumps(turn.input)
        return [message_item("msg_answer", "No cached page.", "completed")]

    events, _ = await run(ResponsesWebTools(client), original, choose)
    assert client.fetches == []
    assert events[-1].data["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_mixed_client_calls_handoff_and_resume_without_repeating_search():
    client = WebClient()
    original = request()
    ordinary = {
        "type": "function_call",
        "id": "fc_client",
        "call_id": "client_call",
        "name": "read_file",
        "arguments": "{}",
        "status": "completed",
    }
    original.tools.append(
        {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
    )

    def choose(turn, number):
        assert number == 1
        return [
            web_call(turn.tools[-1]["name"], {"action": "search", "query": "docs"}),
            ordinary,
        ]

    events, turns = await run(ResponsesWebTools(client), original, choose)
    assert len(turns) == 1
    # Mirror Codex serde: arbitrary result/source fields are not retained.
    retained = deepcopy(events[-1].data["response"]["output"])
    for item in retained:
        if item["type"] == "web_search_call":
            item["action"] = {
                key: value
                for key, value in item["action"].items()
                if key in {"type", "query", "queries"}
            }
    next_request = original.model_copy(
        update={
            "input": [
                *original.input,
                *retained,
                {
                    "type": "function_call_output",
                    "call_id": "client_call",
                    "output": "file contents",
                },
            ]
        },
        deep=True,
    )

    def answer(turn, number):
        assert "An indexed documentation snippet" in json.dumps(turn.input)
        assert "file contents" in json.dumps(turn.input)
        return [message_item("msg_answer", "Done.", "completed")]

    await run(ResponsesWebTools(client), next_request, answer)
    assert client.searches == ["docs"]
    assert prepare_web_history(next_request).records


@pytest.mark.asyncio
async def test_provider_failure_after_search_retains_completed_actions():
    client = WebClient()

    def choose(turn, number):
        if number == 1:
            return [
                web_call(turn.tools[0]["name"], {"action": "search", "query": "docs"})
            ]
        return ExecutionFailure(FailureKind.UPSTREAM, 502, "Provider failed", False)

    events, _ = await run(ResponsesWebTools(client), request(), choose)
    final = events[-1].data["response"]
    assert final["status"] == "failed"
    assert final["error"]["message"] == "Provider failed"
    assert any(item["type"] == "web_search_call" for item in final["output"])
    assert any(
        item.get("encrypted_content", "").startswith("fcc:web:")
        for item in final["output"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_forced_ordinary_tool_does_not_require_local_web(enabled):
    ordinary = {
        "type": "function",
        "name": "read_file",
        "parameters": {"type": "object"},
    }
    original = request(
        tools=[{"type": "web_search"}, ordinary],
        tool_choice={"type": "function", "name": "read_file"},
    )

    def choose(turn, number):
        assert turn.tool_choice == {"type": "function", "name": "read_file"}
        return [
            {
                "type": "function_call",
                "id": "fc_read",
                "call_id": "client_call",
                "name": "read_file",
                "arguments": "{}",
                "status": "completed",
            }
        ]

    events, turns = await run(ResponsesWebTools(enabled=enabled), original, choose)
    assert len(turns) == 1
    assert events[-1].data["response"]["output"][0]["call_id"] == "client_call"


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["auto", "required", {"type": "web_search"}])
async def test_selectable_web_still_requires_local_service(choice):
    with pytest.raises(InvalidRequestError, match="disabled or unavailable"):
        await run(ResponsesWebTools(), request(tool_choice=choice), lambda *_: [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,choice",
    [
        ({"type": "custom", "name": "edit"}, {"type": "custom", "name": "edit"}),
        (
            {
                "type": "namespace",
                "name": "editor",
                "tools": [
                    {
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            {"type": "function", "namespace": "editor", "name": "read"},
        ),
        (SEARCH, {"type": "tool_search", "execution": "client"}),
        (
            {"type": "function", "name": "read", "parameters": {"type": "object"}},
            "none",
        ),
    ],
)
async def test_unavailable_web_does_not_override_other_tool_choices(tool, choice):
    original = request(tools=[{"type": "web_search"}, tool], tool_choice=choice)

    def choose(turn, number):
        assert turn.tool_choice == choice
        return [
            message_item("msg_no_web", "Ordinary tool request accepted.", "completed")
        ]

    events, turns = await run(ResponsesWebTools(enabled=False), original, choose)
    assert len(turns) == 1
    assert events[-1].event == "response.completed"


@pytest.mark.asyncio
async def test_provider_cannot_execute_web_against_forced_ordinary_choice():
    client = WebClient()
    original = request(
        tools=[
            {"type": "web_search"},
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
        ],
        tool_choice={"type": "function", "name": "read_file"},
    )
    events, turns = await run(
        ResponsesWebTools(client, enabled=False),
        original,
        lambda turn, _: [
            web_call(turn.tools[-1]["name"], {"action": "search", "query": "forbidden"})
        ],
    )
    assert len(turns) == 1 and client.searches == []
    assert events[-1].event == "response.failed"
    assert events[-1].data["response"]["output"][0]["status"] == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [False, True])
async def test_negative_find_preserves_page_completeness_through_replay(truncated):
    class PageClient(WebClient):
        async def fetch(self, url, *, egress):
            self.fetches.append(url)
            return WebFetchResult(
                url, "Page", "text/plain", "Start", truncated=truncated
            )

    client = PageClient()

    def open_page(turn, number):
        if number == 1:
            return [
                web_call(turn.tools[0]["name"], {"action": "open_page", "url": URL})
            ]
        return [message_item("msg_open", "Read page.", "completed")]

    first, _ = await run(ResponsesWebTools(client), request(), open_page)

    def find_page(turn, number):
        if number == 1:
            return [
                web_call(
                    turn.tools[0]["name"],
                    {"action": "find_in_page", "url": URL, "pattern": "needle"},
                )
            ]
        result = json.loads(turn.input[-1]["output"])
        assert result["matches"] == []
        assert result["searched_complete_page"] is not truncated
        return [message_item("msg_find", "No match.", "completed")]

    await run(
        ResponsesWebTools(client),
        request(
            input=first[-1].data["response"]["output"],
            tools=[{"type": "web_search", "external_web_access": False}],
        ),
        find_page,
    )
    assert client.fetches == [URL]


@pytest.mark.asyncio
async def test_action_budget_finishes_without_resetting_forced_choice():
    client = WebClient()

    def choose(turn, number):
        if turn.tools:
            assert turn.tool_choice == (
                {"type": "function", "name": turn.tools[0]["name"]}
                if number == 1
                else "auto"
            )
            return [
                web_call(
                    turn.tools[0]["name"], {"action": "search", "query": "docs"}, number
                )
            ]
        return [message_item("msg_end", "Finished.", "completed")]

    events, turns = await run(
        ResponsesWebTools(client),
        request(max_tool_calls=2, tool_choice={"type": "web_search"}),
        choose,
    )
    assert len(client.searches) == 2 and len(turns) == 3
    assert events[-1].data["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_cancellation_closes_inflight_web_operation():
    entered, closed = asyncio.Event(), asyncio.Event()

    class BlockingWeb(WebClient):
        async def search(self, query):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

    task = asyncio.create_task(
        run(
            ResponsesWebTools(BlockingWeb()),
            request(),
            lambda turn, number: [
                web_call(turn.tools[0]["name"], {"action": "search", "query": "docs"})
            ],
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


@pytest.mark.asyncio
async def test_cached_redirect_cannot_bypass_current_domain_filter():
    client = WebClient()
    previous: JsonObject = {
        "id": "ws_previous",
        "action": {"type": "open_page", "url": URL},
        "result": {"text": "old result"},
        "page": {
            "url": "https://elsewhere.example.com/redirected",
            "title": "Redirected",
            "text": "outside the new domain filter",
            "truncated": False,
        },
    }

    def choose(turn, number):
        if number == 1:
            return [
                web_call(turn.tools[0]["name"], {"action": "open_page", "url": URL})
            ]
        latest = json.loads(turn.input[-1]["output"])
        assert latest["error"]["code"] == "blocked_domain"
        return [message_item("msg_end", "Domain blocked.", "completed")]

    await run(
        ResponsesWebTools(client),
        request(
            tools=[
                {
                    "type": "web_search",
                    "external_web_access": False,
                    "filters": {"allowed_domains": ["docs.example.org"]},
                }
            ],
            input=[
                {"role": "user", "content": "Read the page"},
                replay_item([previous]),
            ],
        ),
        choose,
    )
    assert not client.fetches


@pytest.mark.asyncio
async def test_malformed_search_url_does_not_discard_valid_results():
    class BadLink(WebClient):
        async def search(self, query):
            return [
                WebSearchResult("Bad URL", "https://[invalid"),
                WebSearchResult("Valid", URL),
            ]

    def choose(turn, number):
        if number == 1:
            return [
                web_call(turn.tools[0]["name"], {"action": "search", "query": "docs"})
            ]
        latest = json.loads(turn.input[-1]["output"])
        assert [item["url"] for item in latest["results"]] == [URL]
        return [message_item("msg_end", "Done.", "completed")]

    await run(ResponsesWebTools(BadLink()), request(), choose)


@pytest.mark.asyncio
async def test_executor_private_progress_keeps_timeout_alive_without_committing_fallback(
    monkeypatch,
):
    selected = []
    closed = []
    deadlines = []
    heartbeats = []

    def controlled_timeout(deadline):
        deadlines.append(deadline)
        return asyncio.timeout(None)

    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )

    class Provider:
        @asynccontextmanager
        async def bind_responses(self, turn, **kwargs):
            async def stream(request, *, input_tokens):
                try:
                    if turn.model == "provider-model":
                        for _ in range(4):
                            heartbeats.append(asyncio.get_running_loop().time())
                            yield ": private heartbeat\n\n"
                        raise ExecutionFailure(
                            FailureKind.OVERLOADED, 529, "busy", True
                        )
                    for chunk in wire(
                        [message_item("msg_answer", "Fallback answer", "completed")]
                    ):
                        yield chunk
                finally:
                    closed.append(turn.model)

            yield ResponsesBinding("chat", stream)

    async def resolve(provider_id):
        selected.append(provider_id)
        return Provider()

    routed = _routed_responses_request(_target("fallback", "fallback-model"))
    routed.request.tools = [{"type": "web_search"}]
    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=60,
        responses_token_counter=lambda _: 1,
        responses_web_tools=ResponsesWebTools(WebClient()),
    )
    result = "".join(
        [
            chunk
            async for chunk in executor.stream_responses(
                routed, raw_log_payload={}, request_id="progress"
            )
        ]
    )
    assert "Fallback answer" in result and "private heartbeat" not in result
    assert selected == ["provider", "fallback"]
    assert all(
        deadline >= issued + 60
        for deadline, issued in zip(deadlines[1:5], heartbeats, strict=True)
    )
    assert closed == ["provider-model", "fallback-model"]
    assert (
        sum(event.event == "response.created" for event in parse_sse_text(result)) == 1
    )


@pytest.mark.asyncio
async def test_executor_does_not_spend_provider_timeout_on_local_web_work(monkeypatch):
    deadlines = []

    def controlled_timeout(deadline):
        deadlines.append(deadline)
        return asyncio.timeout(None)

    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )

    class SlowWeb(WebClient):
        async def search(self, query):
            assert deadlines[-1] is None
            return await super().search(query)

    turns = []
    releases = []

    class Provider:
        @asynccontextmanager
        async def bind_responses(self, original, **kwargs):
            async def stream(turn, *, input_tokens):
                turns.append(turn)
                items = (
                    [
                        web_call(
                            turn.tools[0]["name"], {"action": "search", "query": "docs"}
                        )
                    ]
                    if len(turns) == 1
                    else [message_item("msg_answer", "Done.", "completed")]
                )
                for chunk in wire(items):
                    yield chunk

            try:
                yield ResponsesBinding("chat", stream)
            finally:
                releases.append(True)

    async def resolve(provider_id):
        return Provider()

    routed = _routed_responses_request()
    routed.request.tools = [{"type": "web_search"}]
    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=60,
        responses_token_counter=lambda _: 1,
        responses_web_tools=ResponsesWebTools(SlowWeb()),
    )
    result = "".join(
        [
            chunk
            async for chunk in executor.stream_responses(
                routed, raw_log_payload={}, request_id="local"
            )
        ]
    )
    assert parse_sse_text(result)[-1].event == "response.completed"
    assert len(turns) == 2 and releases == [True]


@pytest.mark.asyncio
async def test_local_action_timeout_is_a_result_and_does_not_retry(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.application.web_tools.responses._ACTION_TIMEOUT", 0.02
    )
    cancelled = []

    class StalledWeb(WebClient):
        async def search(self, query):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    def choose(turn, number):
        if number == 1:
            return [
                web_call(turn.tools[0]["name"], {"action": "search", "query": "docs"})
            ]
        assert json.loads(turn.input[-1]["output"])["error"]["code"] == "timeout"
        return [message_item("msg_answer", "Search timed out.", "completed")]

    result, turns = await run(ResponsesWebTools(StalledWeb()), request(), choose)
    assert result[-1].event == "response.completed"
    assert len(turns) == 2 and cancelled == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "incomplete"])
async def test_unsuccessful_model_turn_never_executes_its_web_calls(status):
    client = WebClient()

    async def stream(turn, *, input_tokens):
        for chunk in wire(
            [web_call(turn.tools[0]["name"], {"action": "search", "query": "docs"})],
            status=status,
        ):
            yield chunk

    chunks = [
        chunk
        async for chunk in ResponsesWebTools(client)
        .operation(request(), token_counter=lambda turn: 1)
        .stream(ResponsesBinding("chat", stream))
        if isinstance(chunk, str)
    ]
    result = parse_sse_text("".join(chunks))
    assert result[-1].event == f"response.{status}"
    assert not client.searches
    assert result[-1].data["response"]["output"][0]["status"] == "incomplete"
