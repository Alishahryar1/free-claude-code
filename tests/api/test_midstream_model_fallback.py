"""Model fallback continues the public response through real provider transports."""

from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    after_text,
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import partial_tool
from tests.api.test_response_streams import _serve
from tests.api.test_tool_call_buffer import _response
from tests.api.test_tool_call_buffer_transports import _tools
from tests.application.test_execution import _routed_request, _target
from tests.providers.support import make_provider_config
from tests.providers.test_history_transports import _events_for, _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def release_provider_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


async def delivered_candidates(senders, wire, *, tools=None, progress_timeout=10):
    history = [{"role": "user", "content": "greet"}]

    async def open_candidate(index, target, continuation=None):
        return senders[index](
            wire,
            history,
            tools=tools,
            model=target.provider_model,
            continuation=continuation,
        )

    routed = _routed_request(
        *(
            _target(f"fallback-{index}", f"model-{index}")
            for index in range(1, len(senders))
        )
    )
    body = ProviderExecutor(
        AsyncMock(), progress_timeout_seconds=progress_timeout
    )._stream_candidates(
        resolved=routed.resolved,
        reasoning=ReasoningPolicy.provider_default(),
        wire_api=wire,
        raw_log_label="MIDSTREAM_FALLBACK",
        raw_log_payload=dict,
        request_snapshot=dict,
        ingress_count_name="message_count",
        ingress_count=1,
        request_id="midstream-fallback",
        open_candidate=open_candidate,
    )
    response = await _response(wire, body)
    messages = await _serve(response)
    return b"".join(message.get("body", b"") for message in messages).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["chat", "messages", "responses"])
@pytest.mark.parametrize("target", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_exhausted_model_continues_on_next_candidate(source, target, wire):
    async with (
        _harness(
            source,
            lambda bodies: (
                200,
                text_events(
                    source, ["Hello ", "there. "][len(bodies) - 1], complete=False
                ),
            ),
            max_attempts=2,
        ) as (first, first_bodies, _),
        _harness(
            target, lambda _: (200, text_events(target, "Hello there. Done."))
        ) as (second, second_bodies, _),
    ):
        raw = await delivered_candidates([first, second], wire, tools=_tools(wire))

    assert len(first_bodies) == 2
    assert len(second_bodies) == 1
    events = parse_sse_text(raw)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello there. Done."
    body = second_bodies[0]
    history = body.get("messages", body.get("input"))
    assert len(history) == 3
    assert "Hello there. " in str(history[-2])
    assert body["tools"]
    if wire == "responses":
        assert (
            "".join(
                part.get("text", "")
                for item in events[-1].data["response"]["output"]
                for part in item.get("content", [])
            )
            == "Hello there. Done."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_normal_completion_never_opens_another_candidate(protocol, wire):
    async with (
        _harness(protocol, lambda _: (200, text_events(protocol, "Done."))) as (
            first,
            first_bodies,
            _,
        ),
        _harness("chat", lambda _: (200, text_events("chat", "Unexpected"))) as (
            second,
            second_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, second], wire)

    assert len(first_bodies) == 1
    assert not second_bodies
    events = parse_sse_text(raw)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Done."


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_each_model_receives_the_latest_prefix_and_its_own_budget(wire):
    async with (
        _harness(
            "chat",
            lambda _: (200, text_events("chat", "One ", complete=False)),
            max_attempts=1,
        ) as (first, first_bodies, _),
        _harness(
            "messages",
            lambda bodies: (
                200,
                text_events(
                    "messages", ["two ", "three "][len(bodies) - 1], complete=False
                ),
            ),
            max_attempts=2,
        ) as (second, second_bodies, _),
        _harness(
            "responses",
            lambda _: (200, text_events("responses", "One two three four.")),
            max_attempts=1,
        ) as (third, third_bodies, _),
    ):
        raw = await delivered_candidates([first, second, third], wire)
    assert [len(first_bodies), len(second_bodies), len(third_bodies)] == [1, 2, 1]
    for body, expected in [
        (second_bodies[0], "One "),
        (second_bodies[1], "One two "),
        (third_bodies[0], "One two three "),
    ]:
        history = body.get("messages", body.get("input"))
        assert len(history) == 3
        assert expected in str(history[-2])
    events = parse_sse_text(raw)
    assert_completed(events, wire)
    assert public_text(events, wire) == "One two three four."


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "status,error",
    [
        (401, {"type": "authentication_error", "message": "Bad credentials"}),
        (
            400,
            {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "Context full",
            },
        ),
    ],
)
async def test_permanent_rejection_advances_without_erasing_incoming_prefix(
    wire, status, error
):
    async with (
        _harness(
            "chat",
            lambda _: (200, text_events("chat", "Hello ", complete=False)),
            max_attempts=1,
        ) as (first, first_bodies, _),
        _harness("responses", lambda _: (status, error), max_attempts=5) as (
            second,
            second_bodies,
            _,
        ),
        _harness(
            "messages",
            lambda _: (200, text_events("messages", "world.")),
            max_attempts=1,
        ) as (third, third_bodies, _),
    ):
        raw = await delivered_candidates([first, second, third], wire)
    assert [len(first_bodies), len(second_bodies), len(third_bodies)] == [1, 1, 1]
    assert "Hello " in str(third_bodies[0]["messages"][-2])
    assert_completed(parse_sse_text(raw), wire)
    assert public_text(parse_sse_text(raw), wire) == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("visible", [False, True])
async def test_abandoned_calls_never_cross_model_boundary(protocol, wire, visible):
    failed_events = partial_tool(protocol)
    if visible:
        failed_events = after_text(protocol, failed_events)
    async with (
        _harness(protocol, lambda _: (200, failed_events), max_attempts=1) as (
            first,
            first_bodies,
            _,
        ),
        _harness(
            "messages",
            lambda _: (200, tool_events("messages", '{"path":"winning"}')),
            max_attempts=1,
        ) as (second, second_bodies, _),
    ):
        raw = await delivered_candidates([first, second], wire, tools=_tools(wire))
    assert len(first_bodies) == len(second_bodies) == 1
    assert "call_abandoned" not in raw and "abandoned" not in str(second_bodies[0])
    assert len(second_bodies[0]["messages"]) == (3 if visible else 1)
    events = parse_sse_text(raw)
    assert_completed(events, wire)
    assert public_text(events, wire) == ("Before. " if visible else "")
    if wire == "messages":
        assert (
            "".join(
                event.data.get("delta", {}).get("partial_json", "") for event in events
            )
            == '{"path":"winning"}'
        )
    else:
        calls = [
            item
            for item in events[-1].data["response"]["output"]
            if item["type"] == "function_call"
        ]
        assert len(calls) == 1 and calls[0]["arguments"] == '{"path":"winning"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_first_fallback_call_rejects_native_state_without_trying_next_model(
    protocol, wire
):
    async with (
        _harness(
            "chat",
            lambda _: (200, text_events("chat", "Hello ", complete=False)),
            max_attempts=1,
        ) as (first, _, _),
        _harness(protocol, lambda _: (200, _events_for(protocol)), max_attempts=3) as (
            second,
            second_bodies,
            _,
        ),
        _harness("chat", lambda _: (200, text_events("chat", "Unexpected"))) as (
            third,
            third_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, second, third], wire)
    assert len(second_bodies) == 1 and not third_bodies
    events = parse_sse_text(raw)
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")
    assert public_text(events, wire) == "Hello "
    assert "opaque-original" not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("text", ["", "Hello "])
async def test_first_fallback_call_cannot_complete_without_new_output(
    protocol, wire, text
):
    async with (
        _harness(
            "chat",
            lambda _: (200, text_events("chat", "Hello ", complete=False)),
            max_attempts=1,
        ) as (first, _, _),
        _harness(
            protocol, lambda _: (200, text_events(protocol, text)), max_attempts=3
        ) as (second, second_bodies, _),
        _harness("chat", lambda _: (200, text_events("chat", "Unexpected"))) as (
            third,
            third_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, second, third], wire)
    assert len(second_bodies) == 1 and not third_bodies
    events = parse_sse_text(raw)
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")
    assert public_text(events, wire) == "Hello "


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("permanent", [False, True])
async def test_released_calls_handoff_only_for_retryable_failure(
    protocol, wire, permanent
):
    completed = tool_events(protocol, '{"path":"kept"}')
    failed_events = completed[:-2] if protocol == "messages" else completed[:-1]
    if permanent:
        error = {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "Context full",
        }
        failed_events.append(
            {"type": "error", "error": error}
            if protocol == "messages"
            else {
                **completed[-1],
                "type": "response.failed",
                "response": {
                    **completed[-1]["response"],
                    "status": "failed",
                    "error": error,
                },
            }
        )
    async with (
        _harness(protocol, lambda _: (200, failed_events), max_attempts=1) as (
            first,
            first_bodies,
            _,
        ),
        _harness("chat", lambda _: (200, text_events("chat", "Unexpected"))) as (
            second,
            second_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, second], wire, tools=_tools(wire))
    assert len(first_bodies) == 1 and not second_bodies
    events = parse_sse_text(raw)
    if permanent:
        assert events[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )
    else:
        assert_completed(events, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_source_native_state_prevents_model_fallback(protocol, wire):
    async with (
        _harness(
            protocol, lambda _: (200, _events_for(protocol)[:-1]), max_attempts=1
        ) as (first, first_bodies, _),
        _harness("chat", lambda _: (200, text_events("chat", "Unexpected"))) as (
            second,
            second_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, second], wire)
    assert len(first_bodies) == 1 and not second_bodies
    assert parse_sse_text(raw)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("later_failure", [False, True])
async def test_hidden_hosted_activity_remains_a_model_boundary_after_local_retry(
    wire, later_failure
):
    hidden = [
        *partial_tool("responses"),
        {
            "type": "response.output_item.added",
            "sequence_number": 3,
            "output_index": 1,
            "item": {
                "type": "web_search_call",
                "id": "ws_hidden",
                "status": "in_progress",
            },
        },
    ]

    def reply(bodies):
        if len(bodies) == 1:
            return 200, hidden
        if later_failure:
            return 401, {
                "type": "authentication_error",
                "message": "Credentials expired",
            }
        return 200, text_events("responses", "Done.")

    async with (
        _harness("responses", reply, max_attempts=2) as (first, first_bodies, _),
        _harness("chat", lambda _: (200, text_events("chat", "Unexpected"))) as (
            second,
            second_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, second], wire, tools=_tools(wire))
    assert len(first_bodies) == 2
    assert not second_bodies
    events = parse_sse_text(raw)
    if later_failure:
        assert events[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )
    else:
        assert_completed(events, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_shared_cooldown_skips_candidate_with_zero_calls_and_keeps_seed(
    monkeypatch, wire
):
    shared = ProviderAdmissionController(
        provider_name="shared",
        rate_limit=1000,
        max_attempts=2,
        base_delay=60,
        max_delay=60,
        jitter=0,
    )
    executions = []
    start = shared.start_execution

    def capture(**kwargs):
        execution = start(**kwargs)
        executions.append(execution)
        return execution

    monkeypatch.setattr(shared, "start_execution", capture)

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("chat", "Hello ", complete=False)
        return 503, {"type": "server_error", "message": "Unavailable"}

    async with (
        _harness(
            "chat",
            reply,
            chat_provider_factory=lambda: OpenRouterProvider(
                make_provider_config("fixture", "https://provider.invalid/v1"),
                admission=shared,
            ),
        ) as (first, shared_bodies, _),
        _harness("messages", lambda _: (200, text_events("messages", "world."))) as (
            third,
            third_bodies,
            _,
        ),
    ):
        raw = await delivered_candidates([first, first, third], wire)
    assert [execution.attempts_started for execution in executions] == [2, 0]
    assert len(shared_bodies) == 2 and len(third_bodies) == 1
    assert "Hello " in str(third_bodies[0]["messages"][-2])
    assert_completed(parse_sse_text(raw), wire)
    assert public_text(parse_sse_text(raw), wire) == "Hello world."
