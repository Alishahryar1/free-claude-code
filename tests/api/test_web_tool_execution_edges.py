"""Bounded local web-bridge failure and lifecycle regressions without network."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import JSONResponse

from free_claude_code.api.web_tools.egress import WebFetchEgressPolicy
from free_claude_code.api.web_tools.execution import stream_local_web_tool_response
from free_claude_code.application.execution import WireApi
from free_claude_code.application.routing import ModelRouter, RoutedMessagesRequest
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest, Tool
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject, JsonValue
from tests.api.test_web_tool_execution import (
    ScriptedExecutor,
    handler,
    search_call,
    web_request,
)


class RawExecutor(ScriptedExecutor):
    """Supply exact SSE frames, including deliberately malformed wire data."""

    def __init__(self, streams: list[list[str]]) -> None:
        super().__init__([])
        self.streams = streams
        self.closed = 0

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
        frames = self.streams.pop(0)

        async def body() -> AsyncIterator[str]:
            try:
                for frame in frames:
                    yield frame
            finally:
                self.closed += 1

        return body()


@pytest.fixture
def local_http(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, AsyncMock]:
    search = AsyncMock(
        return_value=[{"title": "Docs", "url": "https://example.com/docs"}]
    )
    fetch = AsyncMock(
        return_value={
            "title": "Docs",
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


def _payloads(
    blocks: list[JsonObject],
    *,
    stop: str | None = None,
    stop_sequence: JsonValue = None,
    initial_usage: JsonObject | None = None,
    terminal_usage: JsonObject | None = None,
    partial_json: str | None = None,
) -> list[JsonObject]:
    payloads: list[JsonObject] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_raw",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 0}
                if initial_usage is None
                else initial_usage,
            },
        }
    ]
    for index, block in enumerate(blocks):
        payloads.append(
            {"type": "content_block_start", "index": index, "content_block": block}
        )
        if partial_json is not None:
            payloads.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                }
            )
        payloads.append({"type": "content_block_stop", "index": index})
    reason = stop or (
        "tool_use"
        if any(block["type"] == "tool_use" for block in blocks)
        else "end_turn"
    )
    payloads.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": 5}
                if terminal_usage is None
                else terminal_usage,
            },
            {"type": "message_stop"},
        ]
    )
    return payloads


def _frames(payloads: list[JsonObject]) -> list[str]:
    return [
        format_sse_event(cast(str, payload["type"]), payload) for payload in payloads
    ]


async def _response(
    executor: ScriptedExecutor, request: MessagesRequest | None = None
) -> JSONResponse:
    response = await handler(executor).create(
        web_request() if request is None else request
    )
    assert isinstance(response, JSONResponse)
    return response


def _body(response: JSONResponse) -> JsonObject:
    return cast(JsonObject, json.loads(bytes(response.body)))


def _no_network(local_http: tuple[AsyncMock, AsyncMock]) -> None:
    for mock in local_http:
        mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partial",
    [
        '{"query":"unterminated',
        '{"query":"first","query":"second"}',
        '{"query":NaN}',
        '{"query":Infinity}',
        '{"query":-Infinity}',
        "[]",
        "null",
        '{"query":"valid"} trailing',
        "",
    ],
)
async def test_invalid_tool_json_never_falls_back_to_initial_input(
    local_http, partial: str
):
    executor = RawExecutor([_frames(_payloads([search_call()], partial_json=partial))])
    response = await _response(executor)
    assert response.status_code == 502
    _no_network(local_http)
    assert executor.closed == 1


@pytest.mark.asyncio
async def test_valid_streamed_tool_json_overrides_initial_input(local_http):
    executor = RawExecutor(
        [
            _frames(
                _payloads(
                    [search_call()], partial_json='{"query":"actual streamed query"}'
                )
            ),
            _frames(_payloads([{"type": "text", "text": "Verified answer"}])),
        ]
    )
    response = await _response(executor)
    assert response.status_code == 200
    local_http[0].assert_awaited_once_with("actual streamed query")
    assert executor.closed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        "event: message_start\ndata: {\n\n",
        'event: message_start\ndata: {"type":"message_start","type":"message_stop"}\n\n',
        'event: message_start\ndata: {"type":"message_start","unknown":NaN}\n\n',
        "event: message_start\ndata: []\n\n",
        'event: message_start\ndata: {"type":"message_stop"}\n\n',
    ],
)
async def test_invalid_sse_json_is_not_a_successful_completion(local_http, frame: str):
    executor = RawExecutor([[frame]])
    assert (await _response(executor)).status_code == 502
    _no_network(local_http)
    assert executor.closed == 1


@pytest.mark.asyncio
async def test_fragmented_crlf_stream_and_comment_keepalives(local_http):
    wire = ": keepalive\r\n\r\n" + "".join(_frames(_payloads([search_call()]))).replace(
        "\n", "\r\n"
    )
    executor = RawExecutor(
        [
            list(wire),
            list(
                "".join(
                    _frames(_payloads([{"type": "text", "text": "Answer"}]))
                ).replace("\n", "\r\n")
            ),
        ]
    )
    response = await _response(executor)
    assert response.status_code == 200
    assert _body(response)["usage"] == {
        "input_tokens": 20,
        "output_tokens": 10,
        "server_tool_use": {"web_search_requests": 1},
    }
    local_http[0].assert_awaited_once()
    assert executor.closed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial,terminal",
    [
        ({}, {"output_tokens": 5}),
        ({"input_tokens": True}, {"output_tokens": 5}),
        ({"input_tokens": -1}, {"output_tokens": 5}),
        ({"input_tokens": "10"}, {"output_tokens": 5}),
        ({"input_tokens": 10}, {}),
        ({"input_tokens": 10}, {"output_tokens": True}),
        ({"input_tokens": 10}, {"output_tokens": -1}),
        ({"input_tokens": 10}, {"output_tokens": "5"}),
        ({"input_tokens": 10}, {"input_tokens": False, "output_tokens": 5}),
    ],
)
async def test_invalid_or_missing_usage_blocks_network(
    local_http, initial: JsonObject, terminal: JsonObject
):
    executor = RawExecutor(
        [
            _frames(
                _payloads(
                    [search_call()], initial_usage=initial, terminal_usage=terminal
                )
            )
        ]
    )
    assert (await _response(executor)).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
async def test_decreasing_cumulative_output_usage_is_rejected(local_http):
    payloads = _payloads([search_call()])
    payloads.insert(
        -2, {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 6}}
    )
    assert (await _response(RawExecutor([_frames(payloads)]))).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [{"type": "none"}, {"type": "any"}])
async def test_first_choice_preserved_and_post_tool_choice_is_auto(
    local_http, choice: JsonObject
):
    turns: list[list[JsonObject]] = (
        [[{"type": "text", "text": "No tool"}]]
        if choice["type"] == "none"
        else [[search_call()], [{"type": "text", "text": "Answer"}]]
    )
    executor = ScriptedExecutor(turns)
    response = await _response(executor, web_request(tool_choice=choice))
    assert response.status_code == 200
    assert executor.requests[0].tool_choice == choice
    if choice["type"] == "none":
        _no_network(local_http)
    else:
        assert executor.requests[1].tool_choice == {"type": "auto"}
        local_http[0].assert_awaited_once()


@pytest.mark.asyncio
async def test_any_requires_tool_selection_before_normal_final_answer(local_http):
    executor = ScriptedExecutor([[{"type": "text", "text": "Skipped required tool"}]])
    assert (
        await _response(executor, web_request(tool_choice={"type": "any"}))
    ).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {
            "type": "tool_use",
            "id": "toolu_bad",
            "name": "shell",
            "input": {"query": "q"},
        },
        {"type": "tool_use", "id": "toolu_bad", "name": "web_search", "input": {}},
        {
            "type": "tool_use",
            "id": "bad id",
            "name": "web_search",
            "input": {"query": "q"},
        },
        {
            "type": "tool_use",
            "id": "toolu_search",
            "name": "web_search",
            "input": {"query": "q"},
        },
        {
            "type": "tool_use",
            "id": "toolu_bad",
            "name": "web_fetch",
            "input": {"url": "https://example.com:bad"},
        },
    ],
)
async def test_entire_batch_is_validated_before_first_network_operation(
    local_http, bad: JsonObject
):
    request = web_request(
        tools=[
            Tool(name="web_search", type="web_search_20250305"),
            Tool(name="web_fetch", type="web_fetch_20250910"),
        ]
    )
    response = await _response(ScriptedExecutor([[search_call(), bad]]), request)
    assert response.status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
@pytest.mark.parametrize("count,max_uses", [(5, 20), (2, 1)])
async def test_batch_and_per_tool_limits_reject_before_network(
    local_http, count: int, max_uses: int
):
    calls = [search_call(call_id=f"toolu_{index}") for index in range(count)]
    request = web_request(
        tools=[Tool(name="web_search", type="web_search_20250305", max_uses=max_uses)]
    )
    assert (await _response(ScriptedExecutor([calls]), request)).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
async def test_four_call_batch_forces_next_synthesis_without_tools(local_http):
    executor = ScriptedExecutor(
        [
            [search_call(call_id=f"toolu_{index}") for index in range(4)],
            [{"type": "text", "text": "Synthesized evidence"}],
        ]
    )
    response = await _response(executor)
    assert response.status_code == 200
    assert local_http[0].await_count == 4
    assert executor.requests[-1].tools is None
    assert executor.requests[-1].tool_choice == {"type": "none"}


@pytest.mark.asyncio
async def test_per_tool_exhaustion_disables_synthesis_tools(local_http):
    executor = ScriptedExecutor([[search_call()], [{"type": "text", "text": "Answer"}]])
    response = await _response(
        executor,
        web_request(
            tools=[Tool(name="web_search", type="web_search_20250305", max_uses=1)]
        ),
    )
    assert response.status_code == 200
    assert executor.requests[-1].tools is None
    local_http[0].assert_awaited_once()


@pytest.mark.asyncio
async def test_three_selection_rounds_then_disabled_synthesis(local_http):
    executor = ScriptedExecutor(
        [
            *[[search_call(call_id=f"toolu_{index}")] for index in range(3)],
            [{"type": "text", "text": "Final evidence-based answer"}],
        ]
    )
    response = await _response(executor)
    assert response.status_code == 200
    assert len(executor.requests) == 4
    assert local_http[0].await_count == 3
    assert [request.max_tokens for request in executor.requests] == [100, 95, 90, 85]
    assert executor.requests[-1].tools is None
    assert executor.requests[-1].tool_choice == {"type": "none"}
    assert _body(response)["usage"] == {
        "input_tokens": 40,
        "output_tokens": 20,
        "server_tool_use": {"web_search_requests": 3},
    }


@pytest.mark.asyncio
async def test_unexpected_final_tool_call_is_not_executed(local_http):
    executor = ScriptedExecutor(
        [[search_call(call_id=f"toolu_{index}")] for index in range(4)]
    )
    assert (await _response(executor)).status_code == 502
    assert local_http[0].await_count == 3
    assert executor.requests[-1].tool_choice == {"type": "none"}


@pytest.mark.asyncio
async def test_token_depletion_does_not_execute_unsynthesizable_selection(local_http):
    executor = ScriptedExecutor([[search_call()]])
    response = await _response(executor, web_request(max_tokens=5))
    assert response.status_code == 200
    body = _body(response)
    assert body["stop_reason"] == "max_tokens"
    assert body["content"] == []
    _no_network(local_http)
    assert len(executor.requests) == 1


@pytest.mark.asyncio
async def test_provider_cannot_exceed_remaining_output_budget(local_http):
    executor = RawExecutor(
        [
            _frames(_payloads([search_call()])),
            _frames(
                _payloads(
                    [{"type": "text", "text": "Over budget"}],
                    terminal_usage={"output_tokens": 6},
                )
            ),
        ]
    )
    assert (await _response(executor, web_request(max_tokens=10))).status_code == 502
    assert executor.requests[-1].max_tokens == 5
    local_http[0].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
async def test_final_refusal_and_truncation_preserved(local_http, reason: str):
    executor = RawExecutor(
        [
            _frames(_payloads([search_call()])),
            _frames(
                _payloads(
                    [{"type": "text", "text": "Incomplete or declined"}], stop=reason
                )
            ),
        ]
    )
    response = await _response(executor)
    assert response.status_code == 200
    assert _body(response)["stop_reason"] == reason
    local_http[0].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
async def test_refused_or_truncated_tool_selection_has_no_side_effects(
    local_http, reason: str
):
    executor = RawExecutor([_frames(_payloads([search_call()], stop=reason))])
    assert (await _response(executor)).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing", ["message_stop", "message_delta", "content_block_stop"]
)
async def test_incomplete_selection_is_rejected_before_network(
    local_http, missing: str
):
    payloads = [
        payload for payload in _payloads([search_call()]) if payload["type"] != missing
    ]
    assert (await _response(RawExecutor([_frames(payloads)]))).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,status",
    [
        ("authentication_error", 401),
        ("billing_error", 402),
        ("permission_error", 403),
        ("not_found_error", 404),
        ("request_too_large", 413),
        ("rate_limit_error", 429),
        ("api_error", 500),
        ("timeout_error", 504),
        ("overloaded_error", 529),
    ],
)
async def test_provider_sse_error_type_and_status_preserved(
    local_http, error_type: str, status: int
):
    executor = RawExecutor(
        [
            _frames(
                [
                    {
                        "type": "error",
                        "error": {"type": error_type, "message": "Provider failure"},
                    }
                ]
            )
        ]
    )
    response = await _response(executor)
    assert response.status_code == status
    assert cast(JsonObject, _body(response)["error"])["type"] == error_type
    _no_network(local_http)
    assert executor.closed == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_internal_provider(local_http):
    started = asyncio.Event()
    closed: list[bool] = []

    class HangingExecutor(ScriptedExecutor):
        def stream(
            self,
            routed: RoutedMessagesRequest,
            *,
            wire_api: WireApi,
            raw_log_label: str,
            raw_log_payload: object,
            request_id: str,
        ) -> AsyncIterator[str]:
            async def body() -> AsyncIterator[str]:
                try:
                    started.set()
                    await asyncio.Event().wait()
                    yield ""
                finally:
                    closed.append(True)

            return body()

    task = asyncio.create_task(_response(HangingExecutor([])))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]
    _no_network(local_http)


@pytest.mark.asyncio
async def test_canonical_provider_failure_is_not_reclassified(local_http):
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="Rate limited",
        retryable=False,
    )

    class FailedExecutor(ScriptedExecutor):
        def stream(
            self,
            routed: RoutedMessagesRequest,
            *,
            wire_api: WireApi,
            raw_log_label: str,
            raw_log_payload: object,
            request_id: str,
        ) -> AsyncIterator[str]:
            raise failure

    response = await _response(FailedExecutor([]))
    assert response.status_code == 429
    assert cast(JsonObject, _body(response)["error"])["type"] == "rate_limit_error"
    _no_network(local_http)


@pytest.mark.asyncio
async def test_final_stop_sequence_metadata_preserved(local_http):
    executor = RawExecutor(
        [
            _frames(_payloads([search_call()])),
            _frames(
                _payloads(
                    [{"type": "text", "text": "Answer"}],
                    stop="stop_sequence",
                    stop_sequence="<END>",
                )
            ),
        ]
    )
    response = await _response(executor, web_request(stop_sequences=["<END>"]))
    assert response.status_code == 200
    body = _body(response)
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == "<END>"
    local_http[0].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,sequence",
    [("stop_sequence", None), ("stop_sequence", False), ("end_turn", "<END>")],
)
async def test_inconsistent_stop_sequence_metadata_rejected(
    local_http, reason: str, sequence: JsonValue
):
    executor = RawExecutor(
        [_frames(_payloads([search_call()], stop=reason, stop_sequence=sequence))]
    )
    assert (await _response(executor)).status_code == 502
    _no_network(local_http)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_type", ["auto", "any", "tool"])
async def test_disable_parallel_flag_survives_auto_continuation(
    local_http, initial_type: str
):
    choice: JsonObject = {"type": initial_type, "disable_parallel_tool_use": True}
    if initial_type == "tool":
        choice["name"] = "web_search"
    executor = ScriptedExecutor(
        [
            [search_call(call_id="toolu_first")],
            [search_call(call_id="toolu_second"), search_call(call_id="toolu_third")],
        ]
    )
    routed = ModelRouter(
        Settings.model_validate({"MODEL": "azure_openai/test-model"})
    ).resolve_messages_request(web_request(tool_choice=choice))
    # Forced public requests use the separate legacy fast path; exercise this
    # bridge directly to validate its first-choice/continuation contract too.
    with pytest.raises(ExecutionFailure, match="parallel calls"):
        async for _ in stream_local_web_tool_response(
            routed,
            executor,
            web_fetch_egress=WebFetchEgressPolicy(
                allow_private_network_targets=False,
                allowed_schemes=frozenset({"https", "http"}),
            ),
            request_id="req_parallel_edge",
        ):
            pass
    assert executor.requests[0].tool_choice == choice
    assert executor.requests[1].tool_choice == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    local_http[0].assert_awaited_once()
    local_http[1].assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_parallel_false_survives_auto_continuation(local_http):
    executor = ScriptedExecutor(
        [
            [search_call(call_id="toolu_first")],
            [search_call(call_id="toolu_second"), search_call(call_id="toolu_third")],
            [{"type": "text", "text": "Answer"}],
        ]
    )
    response = await _response(
        executor,
        web_request(tool_choice={"type": "auto", "disable_parallel_tool_use": False}),
    )
    assert response.status_code == 200
    assert executor.requests[1].tool_choice == {
        "type": "auto",
        "disable_parallel_tool_use": False,
    }
    assert local_http[0].await_count == 3
