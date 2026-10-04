"""Public API routes preserve parallel Chat calls with imperfect indexes."""

import json

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import create_test_app
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_chat_tool_identity import call
from tests.providers.test_history_transports import _harness, _saved_reply
from tests.providers.test_native_tool_arguments import tool_events


def chat_chunk(delta, finish=None):
    return {
        "id": "chat_identity",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@pytest.mark.parametrize(
    "indexes", [(0, 1), (0, 0), (None, None), ("missing", "missing")]
)
@pytest.mark.parametrize(
    "wire,streaming", [("messages", True), ("messages", False), ("responses", True)]
)
def test_gemini_parallel_calls_reach_public_api(monkeypatch, indexes, wire, streaming):
    sent = []
    events = [
        chat_chunk(
            {
                "tool_calls": [
                    call(indexes[0], "call_a", '{"path":"a"}'),
                    call(indexes[1], "call_b", '{"path":"b"}'),
                ]
            }
        ),
        chat_chunk({}, "tool_calls"),
    ]
    upstream = (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        + "data: [DONE]\n\n"
    )

    def reply(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=upstream
        )

    sdk = AsyncOpenAI(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    monkeypatch.setattr(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", lambda **_: sdk
    )
    provider = GeminiProvider(
        make_provider_config(api_key="fixture", base_url="https://fixture.invalid/v1"),
        admission=immediate_admission(max_attempts=2),
    )
    app = create_test_app(
        Settings(MODEL="gemini/test-model", ENABLE_WEB_SERVER_TOOLS=False),
        providers={"gemini": provider},
    )
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    if wire == "messages":
        body = {
            "model": "gemini/test-model",
            "max_tokens": 100,
            "stream": streaming,
            "messages": [{"role": "user", "content": "read both"}],
            "tools": [{"name": "read", "input_schema": schema}],
        }
    else:
        body = {
            "model": "gemini/test-model",
            "stream": True,
            "input": "read both",
            "tools": [{"type": "function", "name": "read", "parameters": schema}],
        }
    with TestClient(app) as client:
        response = client.post(f"/v1/{wire}", json=body)
    assert response.status_code == 200, response.text
    assert len(sent) == 1

    if not streaming:
        values = [
            (item["id"], item["name"], item["input"])
            for item in response.json()["content"]
            if item["type"] == "tool_use"
        ]
    elif wire == "messages":
        calls = {}
        events = parse_sse_text(response.text)
        assert events[-1].event == "message_stop"
        for event in events:
            data = event.data
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                calls[data["index"]] = {
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": "",
                }
            if data.get("delta", {}).get("type") == "input_json_delta":
                calls[data["index"]]["arguments"] += data["delta"]["partial_json"]
        values = [
            (item["id"], item["name"], json.loads(item["arguments"]))
            for item in calls.values()
        ]
    else:
        events = parse_sse_text(response.text)
        assert events[-1].event == "response.completed"
        values = [
            (item["call_id"], item["name"], json.loads(item["arguments"]))
            for item in events[-1].data["response"]["output"]
            if item["type"] == "function_call"
        ]
    assert values == [
        ("call_a", "read", {"path": "a"}),
        ("call_b", "read", {"path": "b"}),
    ]


def completed_arguments(events, wire):
    if wire == "messages":
        return "".join(
            event.data["delta"]["partial_json"]
            for event in events
            if event.data.get("delta", {}).get("type") == "input_json_delta"
        )
    return "".join(
        event.data["item"]["arguments"]
        for event in events
        if event.event == "response.output_item.done"
        and event.data["item"]["type"] == "function_call"
    )


def ambiguous_events(stop, *, incomplete=False):
    events = [
        chat_chunk(
            {
                "tool_calls": [
                    call(
                        0,
                        "call_abandoned_a",
                        '{"path":' if incomplete else '{"path":"a"}',
                    ),
                    call(
                        0,
                        "call_abandoned_b",
                        '{"path":' if incomplete else '{"path":"b"}',
                    ),
                ]
            }
        )
    ]
    if stop == "before":
        events.append(chat_chunk({}, "tool_calls"))
    events.append(
        chat_chunk(
            {"tool_calls": [call(0, None, "uncertain", name=None)]},
            "tool_calls" if stop == "same" else None,
        )
    )
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("prefix", [None, "content", "reasoning_content"])
@pytest.mark.parametrize("stop", [None, "same", "before"])
@pytest.mark.parametrize("max_attempts", [1, 2])
async def test_public_identity_failure_respects_delivery_stop_and_budget(
    monkeypatch, wire, prefix, stop, max_attempts
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    first = ambiguous_events(stop)
    if prefix:
        first.insert(0, chat_chunk({prefix: "already visible"}))
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else tool_events("chat", '{"path":"winning"}'),
        ),
        max_attempts=max_attempts,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )

    # OpenRouter's Messages thinking includes a Chat replay signature, which
    # the existing public continuation policy treats as opaque.
    opaque = wire == "messages" and prefix == "reasoning_content"
    retry = stop is None and max_attempts > 1 and not opaque
    assert len(bodies) == (2 if retry else 1)
    assert "call_abandoned" not in result
    events = parse_sse_text(result)
    if prefix:
        assert "already visible" in result
    if retry:
        assert completed_arguments(events, wire) == '{"path":"winning"}'
        assert events[-1].event == (
            "message_stop" if wire == "messages" else "response.completed"
        )
        assert bodies[1]["tools"] == bodies[0]["tools"]
        if prefix:
            assert "already visible" in str(bodies[1])
        else:
            assert bodies[0] == bodies[1]
    else:
        assert events[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )
        assert "winning" not in result
        assert not any(
            event.event
            in {"response.function_call_arguments.done", "response.output_item.done"}
            and (
                event.event != "response.output_item.done"
                or event.data["item"]["type"] == "function_call"
            )
            for event in events
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize(
    "committed,stop,max_attempts",
    [
        (False, None, 2),
        (False, None, 1),
        (True, None, 2),
        (False, "same", 2),
        (False, "before", 2),
        (True, "same", 2),
    ],
)
async def test_private_identity_failure_never_salvages_or_repairs(
    monkeypatch, wire, incomplete, committed, stop, max_attempts
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(
            holdback_seconds=60, max_bytes=1 if committed else 1_000_000
        ),
    )
    first = ambiguous_events(stop, incomplete=incomplete)
    emitted = []
    failure = None
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else tool_events("chat", '{"path":"winning"}'),
        ),
        max_attempts=max_attempts,
    ) as (send, bodies, _):
        try:
            async for event in send(
                wire, [{"role": "user", "content": "read"}], tools=_tools(wire)
            ):
                # Retain the prefix even when iteration raises before completion.
                emitted.extend((event,))
        except ExecutionFailure as error:
            failure = error

    retry = not committed and stop is None and max_attempts > 1
    assert len(bodies) == (2 if retry else 1)
    result = "".join(emitted)
    events = parse_sse_text(result)
    if retry:
        assert failure is None
        assert bodies[0] == bodies[1]
        assert "call_abandoned" not in result
        assert completed_arguments(events, wire) == '{"path":"winning"}'
        assert events[-1].event == (
            "message_stop" if wire == "messages" else "response.completed"
        )
    else:
        assert failure is not None or any(
            event.event == "response.failed" for event in events
        )
        assert not any(
            event.event
            in {
                "message_stop",
                "response.completed",
                "response.function_call_arguments.done",
            }
            for event in events
        )
        started_tools = {
            event.data["index"]
            for event in events
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "tool_use"
        }
        assert not any(
            event.event == "content_block_stop" and event.data["index"] in started_tools
            for event in events
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "indexes,ids",
    [
        ((0, 0), ("call_a", "call_b")),
        ((None, None), ("call_a", "call_b")),
        ((0, 1), ("reused", "reused")),
    ],
)
async def test_gemini_metadata_replays_under_each_public_id(wire, indexes, ids):
    first = [
        chat_chunk(
            {
                "tool_calls": [
                    call(
                        indexes[0],
                        ids[0],
                        '{"path":"a"}',
                        metadata={"google": {"thought_signature": "sig-a"}},
                    ),
                    call(
                        indexes[1],
                        ids[1],
                        '{"path":"b"}',
                        metadata={"google": {"thought_signature": "sig-b"}},
                    ),
                ]
            }
        ),
        chat_chunk({}, "tool_calls"),
    ]

    def gemini():
        return GeminiProvider(
            make_provider_config("fixture", "https://provider.invalid/v1"),
            admission=immediate_admission(max_attempts=2),
        )

    async with _harness(
        "chat",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else [chat_chunk({"content": "done"}, "stop")],
        ),
        chat_provider_factory=gemini,
    ) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "read both"}], tools=_tools(wire)),
            wire,
        )
        if wire == "messages":
            calls = [item for item in saved[0]["content"] if item["type"] == "tool_use"]
            public_ids = [item["id"] for item in calls]
            followup = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": item["id"],
                            "name": item["name"],
                            "input": item["input"],
                        }
                        for item in calls
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}
                        for tool_id in public_ids
                    ],
                },
            ]
        else:
            calls = [item for item in saved if item["type"] == "function_call"]
            public_ids = [item["call_id"] for item in calls]
            followup = [
                {
                    "type": "function_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
                for item in calls
            ] + [
                {"type": "function_call_output", "call_id": tool_id, "output": "ok"}
                for tool_id in public_ids
            ]
        assert len(set(public_ids)) == 2
        await _saved_reply(send(wire, followup, tools=_tools(wire)), wire)
        replayed = next(
            row["tool_calls"] for row in bodies[1]["messages"] if row.get("tool_calls")
        )
    assert [(item["id"], item["extra_content"]) for item in replayed] == [
        (public_ids[0], {"google": {"thought_signature": "sig-a"}}),
        (public_ids[1], {"google": {"thought_signature": "sig-b"}}),
    ]
