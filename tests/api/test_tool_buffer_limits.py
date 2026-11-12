"""Pending limits fail closed across transport retry and model fallback."""

from functools import partial

import pytest

from free_claude_code.core import tool_call_buffer
from free_claude_code.core.openai_responses import tool_calls as responses_tools
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_midstream_model_fallback import delivered_candidates
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("limit", ["bytes", "records"])
async def test_pending_limit_does_not_release_tools_retry_or_fallback(
    monkeypatch, protocol, limit
):
    budget = partial(
        tool_call_buffer.RetentionBudget,
        **({"max_bytes": 4096} if limit == "bytes" else {"max_records": 3}),
    )
    monkeypatch.setattr(tool_call_buffer, "RetentionBudget", budget)
    monkeypatch.setattr(responses_tools, "RetentionBudget", budget)
    events = tool_events(protocol, '{"path":"' + "x" * 5000 + '"}')
    async with (
        _harness(protocol, lambda _: (200, events), max_attempts=3) as (
            send,
            bodies,
            _,
        ),
        _harness(
            "chat", lambda _: (200, text_events("chat", "fallback")), max_attempts=3
        ) as (fallback, fallback_bodies, _),
    ):
        raw = await delivered_candidates([send, fallback], "messages")
    assert len(bodies) == 1
    assert fallback_bodies == []
    assert "pending output limit" in raw
    assert "call_probe" not in raw
    assert "input_json_delta" not in raw
