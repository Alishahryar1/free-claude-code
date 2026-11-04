"""Ordinary thinking uses client history without a second replay representation."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import responses_reasoning_blocks
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.continuation import SourceRecoveryState
from tests.providers.test_anthropic_messages_transport import Endpoint
from tests.providers.test_gemini_transport import collect, gemini_request
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply


def unsigned_events(protocol):
    events = deepcopy(_events_for(protocol))
    if protocol == "chat":
        events[0]["choices"][0]["delta"] = {
            "reasoning_content": "Find 17.",
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": "Find 17.", "index": 0}
            ],
        }
    elif protocol == "messages":
        events[1]["content_block"].pop("signature")
    else:
        for event in events:
            items = [event["item"]] if "item" in event else []
            items += event.get("response", {}).get("output", [])
            for item in items:
                item.pop("encrypted_content", None)
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
async def test_unsigned_thinking_has_only_client_native_history(protocol, wire):
    async with _harness(protocol, lambda _: (200, unsigned_events(protocol))) as (
        send,
        _,
        provider,
    ):
        if wire == "gemini":
            options = {"reasoning": ReasoningPolicy.on()}
            if protocol == "messages":
                options["endpoint_context"] = Endpoint()
                options["reasoning"] = ReasoningPolicy.provider_default()
            elif protocol == "responses":
                options.update(
                    input_tokens=0, request_id="unsigned", response_model="public"
                )
            value, chunks = await collect(
                provider.stream_gemini(
                    gemini_request(
                        generationConfig={"thinkingConfig": {"includeThoughts": True}}
                    ),
                    **options,
                )
            )
            parts = value["candidates"][0]["content"]["parts"]
            assert (
                "".join(p.get("text", "") for p in parts if p.get("thought"))
                == "Find 17."
            )
            assert all("thoughtSignature" not in part for part in parts)
        else:
            chunks = [
                chunk
                async for chunk in send(wire, [{"role": "user", "content": "hello"}])
            ]

            async def replay():
                for chunk in chunks:
                    yield chunk

            saved = await _saved_reply(replay(), wire)
            if wire == "messages":
                blocks = saved[0]["content"]
                assert "".join(b.get("thinking", "") for b in blocks) == "Find 17."
                assert all(
                    not b.get("signature") and b["type"] != "redacted_thinking"
                    for b in blocks
                )
            else:
                assert all(not item.get("encrypted_content") for item in saved)
        assert all(
            event.data.get("type") != "error"
            for event in parse_sse_text("".join(chunks))
        )


def test_unsigned_summary_becomes_ordinary_messages_thinking():
    assert responses_reasoning_blocks(
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan."}]}
    ) == [{"type": "thinking", "thinking": "Plan."}]


def test_standalone_google_signature_uses_protected_recovery_rule():
    source = SourceRecoveryState(Mock())
    payload = {
        "choices": [
            {"delta": {"extra_content": {"google": {"thought_signature": "signed"}}}}
        ]
    }
    source.observe("chat", payload)
    assert source.unsafe_reason
    assert source.can_handoff
    with pytest.raises(ExecutionFailure, match="native state"):
        SourceRecoveryState(Mock()).observe("chat", payload, continuing=True)
