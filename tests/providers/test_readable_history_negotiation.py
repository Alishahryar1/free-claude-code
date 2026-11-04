"""Acceptance-based history correction through the real provider transports."""

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import (
    HistoryMode,
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.anthropic_messages.passthrough import (
    stream_native_messages,
)
from free_claude_code.providers.deepseek import DeepSeekProvider
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    HistoryNegotiation,
    resolve_history_overrides,
)
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_anthropic_messages_transport import (
    _events,
    _sse,
    _transport,
)
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply


def readable_history():
    return [
        {"role": "user", "content": "First"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Old"},
                {"type": "text", "text": "First answer"},
            ],
        },
        {"role": "user", "content": "Next"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "New"},
                {"type": "text", "text": "Next answer"},
            ],
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_carrier", [False, True])
async def test_deepseek_tool_history_keeps_empty_reasoning_after_native_filtering(
    foreign_carrier,
):
    opaque = "cipher"
    if foreign_carrier:
        opaque = encode_replay(
            ReplayRecord(
                ReplayOrigin(
                    "OPENAI_API", "responses", "https://openai.test/v1", "", "model"
                ),
                {
                    "type": "reasoning",
                    "id": "rs_a",
                    "summary": [{"type": "summary_text", "text": "Use search"}],
                    "encrypted_content": "cipher",
                },
            )
        )
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": opaque},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "search",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "Result",
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "search", "input_schema": {"type": "object", "properties": {}}}
            ],
        }
    )
    original = request.model_dump()

    def reply(bodies):
        assistant = next(m for m in bodies[-1]["messages"] if m["role"] == "assistant")
        if "reasoning_content" not in assistant:
            return 400, {
                "message": "Missing reasoning_content in assistant tool history"
            }
        return 200, _events_for("chat")

    async with _harness(
        "chat",
        reply,
        chat_provider_factory=lambda: DeepSeekProvider(
            make_provider_config(api_key="a", base_url="https://provider.test/v1"),
            admission=immediate_admission(),
        ),
    ) as (_, bodies, provider):
        await _saved_reply(
            provider.stream_messages(
                request,
                input_tokens=0,
                request_id="test",
                response_model="public",
                reasoning=ReasoningPolicy.on(),
            ),
            "messages",
        )
    assert len(bodies) == 1
    assistant = next(m for m in bodies[0]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == ""
    assert "reasoning_details" not in assistant
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }
    ]
    assert request.model_dump() == original


@pytest.mark.asyncio
async def test_http_400_tries_all_current_none_without_changing_source():
    def reply(bodies):
        if len(bodies) < 3:
            return 400, {"type": "invalid_request_error", "message": "Rejected"}
        return 200, _events_for("chat")

    history = readable_history()
    original = deepcopy(history)
    async with _harness("chat", reply) as (send, bodies, _):
        await _saved_reply(send("messages", history), "messages")
    assert len(bodies) == 3
    fields = [
        [
            m.get("reasoning_content")
            for m in body["messages"]
            if m["role"] == "assistant"
        ]
        for body in bodies
    ]
    assert fields == [["Old", "New"], [None, "New"], [None, None]]
    assert [
        [m["content"] for m in body["messages"] if m["role"] == "assistant"]
        for body in bodies
    ] == [["First answer", "Next answer"]] * 3
    assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_modes_are_shared_across_transports_and_cached(protocol):
    def reply(bodies):
        if len(bodies) < 3:
            return 400, {"type": "invalid_request_error", "message": "Rejected"}
        return 200, _events_for(protocol)

    async with _harness(protocol, reply) as (send, bodies, _):
        await _saved_reply(send("messages", readable_history()), "messages")
        await _saved_reply(send("messages", readable_history()), "messages")
        assert len(bodies) == 4
        assert bodies[2] == bodies[3] or protocol == "responses"
        for index, expected in enumerate([("Old", "New"), ("New",), (), ()]):
            serialized = json.dumps(bodies[index])
            for thought in ("Old", "New"):
                assert (thought in serialized) == (thought in expected)
            assert "First answer" in serialized and "Next answer" in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_cached_none_rejection_restores_all(protocol):
    def reply(bodies):
        if len(bodies) in {1, 2, 4}:
            return 400, {"message": "Rejected"}
        return 200, _events_for(protocol)

    async with _harness(protocol, reply) as (send, bodies, _):
        await _saved_reply(send("messages", readable_history()), "messages")
        await _saved_reply(send("messages", readable_history()), "messages")
        assert len(bodies) == 5
        assert "Old" not in json.dumps(bodies[3])
        assert "Old" in json.dumps(bodies[4])
        assert "New" in json.dumps(bodies[4])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
@pytest.mark.parametrize("status", [401, 403, 404, 422, 429, 500])
async def test_non_400_does_not_change_mode(protocol, status):
    def reply(bodies):
        return status, {"type": "invalid_request_error", "message": "Rejected"}

    async with _harness(protocol, reply, max_attempts=3) as (send, bodies, _):
        with pytest.raises(ExecutionFailure):
            await _saved_reply(send("messages", readable_history()), "messages")
        assert bodies
        assert all(
            "Old" in json.dumps(body) and "New" in json.dumps(body) for body in bodies
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_total_rejection_does_not_teach_success(protocol):
    reject = True

    def reply(bodies):
        return (
            (400, {"message": "Rejected"}) if reject else (200, _events_for(protocol))
        )

    async with _harness(protocol, reply) as (send, bodies, _):
        with pytest.raises(ExecutionFailure):
            await _saved_reply(send("messages", readable_history()), "messages")
        assert len(bodies) == 3
        reject = False
        await _saved_reply(send("messages", readable_history()), "messages")
        assert "Old" in json.dumps(bodies[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_equivalent_current_and_none_are_not_sent_twice(protocol):
    history = readable_history()[:3]
    async with _harness(protocol, lambda _: (400, {"message": "Rejected"})) as (
        send,
        bodies,
        _,
    ):
        with pytest.raises(ExecutionFailure):
            await _saved_reply(send("messages", history), "messages")
        assert len(bodies) == 2


def cache_origin():
    return ReplayOrigin(
        "provider", "chat", "https://provider.test/v1", "private-credential", "model"
    )


def cache_source():
    return {
        "model": "model",
        "messages": [
            {"role": "assistant", "content": "Answer", "reasoning_content": "Old"},
            {"role": "user", "content": "Next"},
            {"role": "assistant", "content": "Answer", "reasoning_content": "New"},
        ],
    }


@pytest.mark.parametrize("dimension", ["endpoint", "model"])
def test_destination_change_restarts_mode_order_and_seen_shapes(dimension):
    cache = HistoryModeCache()
    origin = cache_origin()
    state = HistoryNegotiation(cache, origin)
    source = cache_source()
    state.prepare(source, origin)
    assert state.rejected() is not None
    state.prepare(source, origin)
    assert state.mode is HistoryMode.CURRENT

    target = replace(
        origin,
        **{dimension: "https://new.test/v1" if dimension == "endpoint" else "new"},
    )
    target_key = (target.provider, target.endpoint, target.protocol, target.model)
    cache.update(target_key, 0, HistoryMode.NONE)
    source = {**source, "model": target.model}
    wire = state.prepare(source, target)
    assert "Old" not in json.dumps(wire) and "New" not in json.dumps(wire)
    assert state.mode is HistoryMode.NONE
    assert state.rejected() is not None
    wire = state.prepare(source, target)
    assert "Old" in json.dumps(wire) and "New" in json.dumps(wire)
    assert state.mode is HistoryMode.ALL
    state.accepted()
    assert cache.read(target_key).mode is HistoryMode.ALL


def test_credential_refresh_at_same_destination_keeps_negotiation_progress():
    state = HistoryNegotiation(HistoryModeCache(), cache_origin())
    state.prepare(cache_source(), cache_origin())
    assert state.rejected() is not None
    refreshed = replace(cache_origin(), connection="refreshed")
    wire = state.prepare(cache_source(), refreshed)
    assert "Old" not in json.dumps(wire) and "New" in json.dumps(wire)
    assert state.mode is HistoryMode.CURRENT
    assert state.rejected() is not None
    wire = state.prepare(cache_source(), refreshed)
    assert "New" not in json.dumps(wire)
    assert state.rejected() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("new_status", [200, 400])
async def test_auth_refresh_uses_new_endpoint_mode_with_same_attempt_budget(new_status):
    class Endpoint:
        fresh = False

        async def endpoint(self, *, force_refresh=False):
            self.fresh |= force_refresh
            return HttpEndpoint(
                "https://new.test/v1" if self.fresh else "https://old.test/v1",
                {},
                "token",
            )

    bodies = []

    def reply(request):
        bodies.append((request.url.host, json.loads(request.content)))
        if request.url.host == "old.test":
            return httpx.Response(
                401,
                json={"error": {"type": "authentication_error", "message": "expired"}},
            )
        if new_status == 400:
            return httpx.Response(400, json={"error": {"message": "Rejected"}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(*_events()),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        adapter = _transport(client)
        for endpoint, mode in [("old", HistoryMode.NONE), ("new", HistoryMode.ALL)]:
            adapter._history_cache.update(
                ("test/messages", f"https://{endpoint}.test/v1", "messages", "route"),
                0,
                mode,
            )
        stream = adapter.stream_messages(
            MessagesRequest.model_validate(
                {"model": "route", "messages": readable_history()}
            ),
            endpoint_context=Endpoint(),
            reasoning=ReasoningPolicy.provider_default(),
        )
        if new_status == 400:
            with pytest.raises(ExecutionFailure):
                await _saved_reply(stream, "messages")
        else:
            await _saved_reply(stream, "messages")
            assert (
                adapter._history_cache.read(
                    ("test/messages", "https://new.test/v1", "messages", "route")
                ).mode
                is HistoryMode.ALL
            )
    assert [host for host, _ in bodies] == ["old.test", "new.test"]
    assert "Old" not in json.dumps(bodies[0][1]) and "New" not in json.dumps(
        bodies[0][1]
    )
    assert "Old" in json.dumps(bodies[1][1]) and "New" in json.dumps(bodies[1][1])


def test_late_cold_completion_cannot_replace_newer_suggestion():
    cache = HistoryModeCache()
    origin = cache_origin()
    early = HistoryNegotiation(cache, origin)
    late = HistoryNegotiation(cache, origin)
    for state in (early, late):
        state.prepare(cache_source(), origin)
    assert early.rejected() is not None
    early.prepare(cache_source(), origin)
    early.accepted()
    late.accepted()
    assert HistoryNegotiation(cache, origin).mode is HistoryMode.CURRENT


def test_cached_invalidation_keeps_update_token_but_cannot_replace_newer_acceptance():
    cache = HistoryModeCache()
    origin = cache_origin()
    seed = HistoryNegotiation(cache, origin)
    seed.prepare(cache_source(), origin)
    seed.accepted()
    stale = HistoryNegotiation(cache, origin)
    stale.prepare(cache_source(), origin)
    assert stale.rejected() is not None
    fresh = HistoryNegotiation(cache, origin)
    fresh.prepare(cache_source(), origin)
    fresh.accepted()
    stale.prepare(cache_source(), origin)
    stale.accepted()
    assert HistoryNegotiation(cache, origin).mode is HistoryMode.ALL
    assert "private-credential" not in repr(cache._entries)
    assert "Answer" not in repr(cache._entries)


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "other"),
        ("endpoint", "https://provider.test/v2"),
        ("protocol", "responses"),
        ("model", "other"),
    ],
)
def test_cache_dimensions(field, value):
    cache = HistoryModeCache()
    origin = cache_origin()
    state = HistoryNegotiation(cache, origin)
    state.prepare(cache_source(), origin)
    state.rejected()
    state.prepare(cache_source(), origin)
    state.accepted()
    assert (
        HistoryNegotiation(cache, replace(origin, **{field: value})).mode
        is HistoryMode.ALL
    )
    assert (
        HistoryNegotiation(cache, replace(origin, connection="another-account")).mode
        is HistoryMode.CURRENT
    )
    cache.clear()
    assert HistoryNegotiation(cache, origin).mode is HistoryMode.ALL


def test_sdk_history_and_model_override_project_without_conflicting_overlay():
    original = {
        "model": "wrong",
        "messages": [],
        "extra_body": {**cache_source(), "temperature": 0.4},
    }
    resolved = resolve_history_overrides(original)
    assert resolved["model"] == "model"
    state = HistoryNegotiation(HistoryModeCache(), cache_origin())
    all_body = state.prepare(resolved, cache_origin())
    assert "Old" in json.dumps(all_body)
    state.rejected()
    state.prepare(resolved, cache_origin())
    state.rejected()
    none_body = state.prepare(resolved, cache_origin())
    assert "Old" not in json.dumps(none_body)
    assert "New" not in json.dumps(none_body)
    assert none_body["extra_body"] == {"temperature": 0.4}
    assert original["messages"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_native_messages_negotiates_stream_and_json_and_reuses_cache(streaming):
    bodies = []
    cache = HistoryModeCache()

    def respond(request):
        bodies.append(json.loads(request.content))
        if len(bodies) < 3:
            return httpx.Response(
                400,
                json={
                    "error": {"type": "invalid_request_error", "message": "Rejected"}
                },
            )
        if not streaming:
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "model",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        events = _events_for("messages")
        raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=raw
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        for _ in range(2):
            result = [
                event
                async for event in stream_native_messages(
                    client,
                    immediate_admission(),
                    base_url="https://provider.test/v1",
                    headers={},
                    body={
                        "model": "model",
                        "messages": readable_history(),
                        "max_tokens": 100,
                        "stream": streaming,
                    },
                    public_model="public",
                    provider_name="native",
                    read_timeout_s=30,
                    request_id="native-history",
                    history_cache=cache,
                )
            ]
            assert result
    assert len(bodies) == 4
    assert "Old" in json.dumps(bodies[0])
    assert "Old" not in json.dumps(bodies[1]) and "New" in json.dumps(bodies[1])
    assert "New" not in json.dumps(bodies[2])
    assert bodies[2] == bodies[3]


def test_precise_native_repair_keeps_other_records_in_same_carrier():
    from free_claude_code.core.history_replay import ReplayRecord, encode_replay
    from free_claude_code.providers.request_recovery import RequestCorrections

    origin = cache_origin()
    first = {"type": "reasoning.text", "text": "First", "signature": "first-sig"}
    second = {"type": "reasoning.text", "text": "Second", "signature": "second-sig"}
    carrier = encode_replay(
        ReplayRecord(origin, {"reasoning_details": [first, second]})
    )
    source: dict[str, Any] = {
        "model": "model",
        "messages": [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": carrier}],
            }
        ],
    }
    corrections = RequestCorrections("chat")
    sent = corrections.prepare_history(source, origin, HistoryModeCache())
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://provider.test"),
        json={
            "error": {
                "code": "invalid_signature",
                "message": "invalid signature",
                "param": "messages[0].reasoning_details[0]",
            }
        },
    )
    error = httpx.HTTPStatusError(
        "rejected", request=response.request, response=response
    )
    corrected = corrections.next_body(
        error, source, sent_body=sent, reasoning_error=error
    )
    assert corrected is not None
    retried = cast(
        dict[str, Any],
        corrections.prepare_history(corrected, origin, HistoryModeCache()),
    )
    assert retried["messages"][0]["reasoning_details"] == [second]
    assert "first-sig" not in json.dumps(retried)
    assert source["messages"][0]["reasoning_details"][0]["data"] == carrier


@pytest.mark.asyncio
async def test_scoped_native_correction_then_all_restores_only_readable_source():
    history = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "Old"}],
            "summary": [],
        },
        {
            "type": "reasoning",
            "id": "rs_signed",
            "encrypted_content": "cipher",
            "summary": [],
        },
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Next"},
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "New"}],
            "summary": [],
        },
        {"role": "assistant", "content": "Next answer"},
    ]
    original = deepcopy(history)

    def reply(bodies):
        if len(bodies) in {1, 2, 5}:
            return 400, {"message": "Rejected"}
        if len(bodies) == 4:
            assert bodies[-1]["input"][0]["id"] == "rs_signed"
            return 400, {
                "code": "invalid_encrypted_content",
                "message": "Invalid cipher",
                "param": "input[0]",
            }
        return 200, _events_for("responses")

    async with _harness("responses", reply) as (send, bodies, _):
        await _saved_reply(send("responses", history), "responses")
        await _saved_reply(send("responses", history), "responses")
        assert len(bodies) == 6
        assert "Old" in json.dumps(bodies[-1]) and "New" in json.dumps(bodies[-1])
        assert "cipher" not in json.dumps(bodies[-1])
        assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_first_in_band_error_neither_changes_mode_nor_teaches_acceptance(
    protocol,
):
    reject = True

    def reply(bodies):
        if reject:
            event = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "400 bad request",
                    "code": "invalid_request_error",
                },
            }
            return 200, [event]
        return 200, _events_for(protocol)

    async with _harness(protocol, reply, max_attempts=2) as (send, bodies, _):
        with pytest.raises(ExecutionFailure):
            await _saved_reply(send("messages", readable_history()), "messages")
        assert all("Old" in json.dumps(body) for body in bodies)
        reject = False
        await _saved_reply(send("messages", readable_history()), "messages")
        assert "Old" in json.dumps(bodies[-1])


@pytest.mark.asyncio
async def test_non_history_correction_survives_all_three_modes():
    def reply(bodies):
        if len(bodies) == 1:
            return 400, {
                "message": "Unsupported parameter: stream_options.include_usage"
            }
        if len(bodies) < 4:
            return 400, {"message": "Rejected"}
        return 200, _events_for("chat")

    async with _harness("chat", reply) as (send, bodies, _):
        await _saved_reply(send("messages", readable_history()), "messages")
        assert len(bodies) == 4
        assert "Old" in json.dumps(bodies[1])
        assert "Old" not in json.dumps(bodies[2]) and "New" in json.dumps(bodies[2])
        assert "New" not in json.dumps(bodies[3])
        assert all(
            not b.get("stream_options", {}).get("include_usage") for b in bodies[1:]
        )
