"""Ordinary thinking uses client history without a second replay representation."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import Mock

import httpx2
import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.gemini.replay import decode_bundle
from free_claude_code.core.history_replay import (
    responses_reasoning_blocks,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.continuation import SourceRecoveryState
from tests.providers.test_anthropic_messages_transport import Endpoint
from tests.providers.test_gemini_transport import collect, gemini_request
from tests.providers.test_history_transports import (
    _chat_reasoning_events,
    _events_for,
    _harness,
    _saved_reply,
)


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


def test_gemini_continuation_accepts_signature_but_keeps_failure_protected():
    source = SourceRecoveryState(Mock())
    source.observe(
        "gemini",
        {"candidates": [{"content": {"parts": [{"text": "Plan", "thought": True}]}}]},
        continuing=True,
    )
    assert source.unsafe_reason is None
    source.observe(
        "gemini",
        {"candidates": [{"content": {"parts": [{"thoughtSignature": "signed"}]}}]},
        continuing=True,
    )
    assert (
        source.snapshot(
            body={}, protocol="gemini", normal_stop_seen=False
        ).unsafe_reason
        == "signed_thinking"
    )
    assert not source.can_handoff


def test_gemini_signature_cannot_clear_unsupported_native_evidence():
    source = SourceRecoveryState(Mock())
    source.observe(
        "gemini",
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": "AA==", "mimeType": "image/png"}}
                        ]
                    }
                }
            ]
        },
    )
    with pytest.raises(ExecutionFailure):
        source.observe(
            "gemini",
            {"candidates": [{"content": {"parts": [{"thoughtSignature": "signed"}]}}]},
            continuing=True,
        )
    assert source.unsafe_reason == "native_content"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
@pytest.mark.parametrize("scalar_first", [False, True])
@pytest.mark.parametrize("parsed_first", [False, True])
@pytest.mark.parametrize("parsed", ["Independent parsed.", "Native 🧠é."])
async def test_parsed_thinking_is_not_owned_by_native_signature(
    wire, scalar_first, parsed_first, parsed
):
    native = "Native 🧠é."
    detail = {
        "type": "reasoning.text",
        "text": native,
        "signature": "signed-native",
        "index": 0,
    }
    typed = {"reasoning_details": [detail]}
    parsed_delta = {"content": f"<think>{parsed}</think>"}
    deltas = [{"reasoning_content": native} if scalar_first else typed]
    deltas.insert(0 if parsed_first else 1, parsed_delta)
    if scalar_first:
        deltas.append(typed)
    events = _chat_reasoning_events(deltas)

    async with _harness("chat", lambda _: (200, events)) as (send, bodies, provider):
        if wire == "gemini":
            value, _ = await collect(
                provider.stream_gemini(
                    gemini_request(
                        generationConfig={"thinkingConfig": {"includeThoughts": True}}
                    ),
                    reasoning=ReasoningPolicy.on(),
                )
            )
            content = value["candidates"][0]["content"]
            visible = "".join(
                part.get("text", "") for part in content["parts"] if part.get("thought")
            )
            assert visible == (parsed + native if parsed_first else native + parsed)
            await collect(
                provider.stream_gemini(
                    gemini_request(
                        contents=[
                            content,
                            {"role": "user", "parts": [{"text": "next"}]},
                        ]
                    ),
                    reasoning=ReasoningPolicy.on(),
                )
            )
        else:
            saved = await _saved_reply(
                send(wire, [{"role": "user", "content": "hello"}]), wire
            )
            await _saved_reply(
                send(wire, [*saved, {"role": "user", "content": "next"}]), wire
            )
    assert len(bodies) == 2
    messages = bodies[-1]["messages"]
    signed = [
        item
        for message in messages
        for item in message.get("reasoning_details", [])
        if item.get("signature") == "signed-native"
    ]
    assert [(item["text"], item["signature"]) for item in signed] == [
        (native, "signed-native")
    ]
    assert signed == [detail]
    assert (
        "".join(message.get("reasoning_content", "") for message in messages) == parsed
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
async def test_owned_thinking_and_answer_stream_before_late_signature(wire):
    prefix = _chat_reasoning_events(
        [
            {"reasoning_content": "Native thought."},
            {"content": "<think>Parsed thought.</think>Answer."},
        ]
    )[:-1]
    suffix = _chat_reasoning_events(
        [
            {
                "reasoning_details": [
                    {
                        "type": "reasoning.text",
                        "text": "Native thought.",
                        "signature": "late",
                        "index": 0,
                    }
                ]
            }
        ]
    )
    chunks = []
    delivered = asyncio.Event()

    class Gated(httpx2.AsyncByteStream):
        async def __aiter__(self):
            for event in prefix:
                yield f"data: {json.dumps(event)}\n\n".encode()
            await asyncio.wait_for(delivered.wait(), 2)
            visible = "".join(chunks)
            assert all(
                text in visible
                for text in ("Native thought.", "Parsed thought.", "Answer.")
            ), visible
            assert "fcc:history:" not in visible
            for event in suffix:
                yield f"data: {json.dumps(event)}\n\n".encode()

    async with _harness("chat", lambda _: (200, Gated())) as (send, _, provider):
        stream = (
            provider.stream_gemini(
                gemini_request(
                    generationConfig={"thinkingConfig": {"includeThoughts": True}}
                ),
                reasoning=ReasoningPolicy.on(),
            )
            if wire == "gemini"
            else send(wire, [{"role": "user", "content": "hello"}])
        )
        async for chunk in stream:
            chunks.append(chunk)
            if all(
                text in "".join(chunks)
                for text in ("Native thought.", "Parsed thought.", "Answer.")
            ):
                delivered.set()
    assert ("thoughtSignature" if wire == "gemini" else "fcc:history:") in "".join(
        chunks
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", ["streamed", "merged", "cli"])
async def test_gemini_protected_owners_survive_reused_indices_and_client_echo(echo):
    first = {
        "type": "reasoning.text",
        "text": "First.",
        "signature": "first",
        "index": 0,
    }
    second = {
        "type": "reasoning.text",
        "text": "Second.",
        "signature": "second",
        "index": 0,
    }
    events = _chat_reasoning_events(
        [
            {"reasoning_content": "Unsigned."},
            {"content": "start"},
            {"reasoning_content": "First."},
            {"content": "<think>Ordinary.</think>middle"},
            {"reasoning_details": [first]},
            {"reasoning_content": "Second."},
            {"reasoning_details": [second]},
            {"content": "<think>Tail."},
        ]
    )
    # Leave the parser in thinking mode so its final flush also has no owner.
    events[-1]["choices"][0]["delta"] = {}
    async with _harness("chat", lambda _: (200, events)) as (_, bodies, provider):
        value, chunks = await collect(
            provider.stream_gemini(
                gemini_request(
                    generationConfig={"thinkingConfig": {"includeThoughts": True}}
                ),
                reasoning=ReasoningPolicy.on(),
            )
        )
        content = value["candidates"][0]["content"]
        if echo == "streamed":
            content["parts"] = [
                part
                for frame in parse_sse_text("".join(chunks))
                for candidate in frame.data.get("candidates", [])
                for part in candidate.get("content", {}).get("parts", [])
            ]
        elif echo == "cli":
            # Gemini CLI 0.29.5 processStreamResponse drops all thought Parts.
            content["parts"] = [
                part for part in content["parts"] if not part.get("thought")
            ]
        bundle = decode_bundle(
            next(
                p["thoughtSignature"]
                for p in content["parts"]
                if p.get("thoughtSignature")
            )
        )
        assert bundle is not None
        assert [record.native for record in bundle.records] == [
            {"reasoning_content": "First.", "reasoning_details": [first]},
            {"reasoning_content": "Second.", "reasoning_details": [second]},
        ]
        await collect(
            provider.stream_gemini(
                gemini_request(
                    contents=[content, {"role": "user", "parts": [{"text": "next"}]}]
                ),
                reasoning=ReasoningPolicy.on(),
            )
        )
    assistant = bodies[-1]["messages"][0]
    assert assistant["reasoning_details"] == [first, second]
    assert assistant.get("reasoning_content", "") == (
        "" if echo == "cli" else "Unsigned.Ordinary.Tail."
    )
