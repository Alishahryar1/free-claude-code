"""Physical usage remains distinct from logical response estimates."""

from unittest.mock import patch

import pytest

from tests.providers.test_history_transports import (
    _chat_reasoning_events,
    _events_for,
    _harness,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
async def test_each_attempt_reports_counts_or_explicitly_missing_usage(protocol):
    def reply(bodies):
        return 200, [] if len(bodies) == 1 else _events_for(protocol)

    with patch("free_claude_code.providers.stream_candidate.trace_event") as trace:
        async with _harness(protocol, reply) as (send, bodies, _):
            output = [
                event
                async for event in send(
                    "messages", [{"role": "user", "content": "hello"}]
                )
            ]
        assert output and len(bodies) == 2
    rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.attempt.usage"
    ]
    assert len(rows) == 2
    assert rows[0]["usage_status"] == "missing"
    assert rows[0]["completed"] is False
    assert rows[1]["completed"] is True
    if protocol != "chat":
        assert rows[1]["usage_status"] == "reported"
        assert rows[1]["reported_output_tokens"] >= 0
    assert "opaque-original" not in repr(rows)


@pytest.mark.asyncio
async def test_usage_from_a_failed_attempt_is_recorded_without_prompt_or_arguments():
    events = _chat_reasoning_events([{"content": "first"}])[:-1]
    events[0]["usage"] = {
        "prompt_tokens": 111,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 100},
    }

    def reply(bodies):
        return 200, events if len(bodies) == 1 else _chat_reasoning_events(
            [{"content": "second"}]
        )

    with patch("free_claude_code.providers.stream_candidate.trace_event") as trace:
        async with _harness("chat", reply) as (send, _, _):
            assert [
                event
                async for event in send(
                    "messages", [{"role": "user", "content": "private-prompt"}]
                )
            ]
    rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.attempt.usage"
    ]
    assert rows[0]["completed"] is False
    assert rows[0]["reported_input_tokens"] == 111
    assert rows[0]["reported_output_tokens"] == 7
    assert rows[0]["reported_cache_read_tokens"] == 100
    assert "private-prompt" not in repr(rows)
