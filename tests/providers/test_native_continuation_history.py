"""Recovery reuses client-native delivered output through ordinary history policy."""

from copy import deepcopy
from typing import Any

import pytest

from free_claude_code.core.history_replay import HistoryMode, ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.continuation import ContinuationRequest
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    HistoryNegotiation,
)


def native_seed(wire):
    items: list[JsonObject]
    if wire == "messages":
        items = [
            {"type": "thinking", "thinking": "Partial thought"},
            {"type": "text", "text": "Partial answer"},
        ]
    elif wire == "responses":
        items = [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Partial thought"}],
                "summary": [],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Partial answer"}],
            },
        ]
    else:
        items = [
            {"thought": True, "text": "Partial thought"},
            {"text": "Partial answer"},
        ]
    return ContinuationSeed(wire_api=wire, items=tuple(items))


@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("mode", list(HistoryMode))
def test_partial_thinking_is_governed_by_history_mode(wire, protocol, mode):
    field = "input" if protocol == "responses" else "messages"
    body = {"model": "model", field: [{"role": "user", "content": "Question"}]}
    original = deepcopy(body)
    seed = native_seed(wire)
    source = ContinuationRequest(protocol).build(body, seed)
    origin = ReplayOrigin("provider", protocol, "https://test.invalid", "key", "model")
    cache = HistoryModeCache()
    cache.update(
        (origin.provider, origin.endpoint, origin.protocol, origin.model), 0, mode
    )
    wire_body = HistoryNegotiation(cache, origin).prepare(
        source, origin, continuation=True
    )
    rows = wire_body[field]
    assert isinstance(rows, list)
    assert "Partial thought" not in str(rows[-1])
    assert "Partial answer" in str(rows)
    assert ("Partial thought" in str(rows)) is (mode is not HistoryMode.NONE)
    assert body == original
    assert "Partial thought" in str(seed.items)


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_current_keeps_current_history_and_prefix_but_not_old_thinking(protocol):
    field = "input" if protocol == "responses" else "messages"
    request = ContinuationRequest(protocol)
    first = request.build({"model": "model", field: []}, native_seed("messages"))
    # A real user message after this old assistant separates the current turn.
    first[field][-1] = {"role": "user", "content": "New question"}
    source = ContinuationRequest(protocol).build(first, native_seed("messages"))
    origin = ReplayOrigin("provider", protocol, "https://test.invalid", "key", "model")
    cache = HistoryModeCache()
    cache.update(
        (origin.provider, origin.endpoint, origin.protocol, origin.model),
        0,
        HistoryMode.CURRENT,
    )
    projected = HistoryNegotiation(cache, origin).prepare(
        source, origin, continuation=True
    )
    assert str(projected[field]).count("Partial thought") == 1


def test_native_suffix_preserves_response_summary_meaning_and_input():
    item: dict[str, Any] = {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "Full"}],
        "summary": [{"type": "summary_text", "text": "Summary"}],
    }
    seed = ContinuationSeed(wire_api="responses", items=(item,))
    item["summary"][0]["text"] = "Changed externally"
    source = ContinuationRequest("responses").build({"input": []}, seed)
    assert source["input"][0]["summary"][0]["text"] == "Summary"
    assert source["input"][0]["content"][0]["text"] == "Full"
    assert "Summary" not in str(source["input"][-1])


def test_response_continuation_preserves_delivered_part_text():
    item: JsonObject = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "pri"},
            {"type": "output_text", "text": "nt("},
        ],
    }
    before = deepcopy(item)
    seed = ContinuationSeed("responses", (item,))
    sent = ContinuationRequest("chat").build(
        {"model": "model", "messages": [{"role": "user", "content": "Code"}]},
        seed,
    )
    assert sent["messages"][-2]["content"] == seed.readable_text("text") == "print("
    assert item == before


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_fallback_prefix_follows_effective_sdk_history_override(protocol):
    from free_claude_code.core.anthropic.models import MessagesRequest
    from free_claude_code.core.openai_responses import OpenAIResponsesRequest
    from free_claude_code.core.reasoning import ReasoningPolicy
    from free_claude_code.providers.deepseek import DeepSeekProvider
    from tests.api.test_delivered_stream_recovery import text_events
    from tests.providers.support import immediate_admission, make_provider_config
    from tests.providers.test_history_transports import _harness, _saved_reply

    field = "messages" if protocol == "chat" else "input"
    supplied = {
        "model": "original-model",
        field: [{"role": "user", "content": "Overridden question"}],
        "extra_body": {
            "model": "effective-model",
            field: [{"role": "user", "content": "Effective question"}],
        },
    }
    request = (
        MessagesRequest.model_validate(supplied)
        if protocol == "chat"
        else OpenAIResponsesRequest.model_validate(supplied)
    )
    before = request.model_dump()
    wire = "messages" if protocol == "chat" else "responses"
    async with _harness(
        protocol,
        lambda _: (200, text_events(protocol, "Finished.")),
        chat_provider_factory=lambda: DeepSeekProvider(
            make_provider_config(api_key="a", base_url="https://provider.invalid/v1"),
            admission=immediate_admission(),
        ),
    ) as (_, bodies, provider):
        await _saved_reply(
            getattr(provider, f"stream_{wire}")(
                request,
                input_tokens=0,
                request_id="override-continuation",
                response_model="public",
                reasoning=ReasoningPolicy.on(),
                continuation=native_seed(wire),
            ),
            wire,
        )
    assert len(bodies) == 1
    assert bodies[0]["model"] == "effective-model"
    assert "Effective question" in str(bodies[0][field])
    assert "Overridden question" not in str(bodies[0][field])
    assert "Partial thought" in str(bodies[0][field])
    assert "Partial answer" in str(bodies[0][field])
    assert request.model_dump() == before
