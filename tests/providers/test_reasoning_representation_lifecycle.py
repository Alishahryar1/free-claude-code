"""Native reasoning item and part ownership survives completion and replay."""

import json
from copy import deepcopy

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.kilo import KiloProvider
from free_claude_code.providers.openai_chat.reasoning_details import (
    StructuredReasoningStream,
)
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
    ChatStreamUsage,
)
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply
from tests.providers.test_native_reasoning_completion import _chunks


def _responses_events(
    texts, *, field="summary", streamed=True, late=False, signature=None
):
    source = deepcopy(_events_for("responses"))
    reasoning = {**source[3]["item"], "summary": [], "content": []}
    reasoning.pop("encrypted_content", None)
    part_type = "summary_text" if field == "summary" else "reasoning_text"
    reasoning[field] = [{"type": part_type, "text": text} for text in texts]
    if signature is not None:
        reasoning["encrypted_content"] = signature
    message = {
        "id": "msg_answer",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Answer.", "annotations": []}],
    }
    events = [
        source[0],
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                **reasoning,
                "status": "in_progress",
                "summary": [],
                "content": [],
                "encrypted_content": None,
            },
        },
    ]
    if streamed:
        for index, text in enumerate(texts):
            events.append(
                {
                    "type": "response.reasoning_summary_text.delta"
                    if field == "summary"
                    else "response.reasoning_text.delta",
                    "item_id": reasoning["id"],
                    "output_index": 0,
                    "summary_index" if field == "summary" else "content_index": index,
                    "delta": text,
                }
            )
    done = {"type": "response.output_item.done", "output_index": 0, "item": reasoning}
    if not late:
        events.append(done)
    events += [
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": message["id"],
            "output_index": 1,
            "content_index": 0,
            "delta": "Answer.",
        },
        {"type": "response.output_item.done", "output_index": 1, "item": message},
    ]
    if late:
        events.append(done)
    events.append(
        {
            "type": "response.completed",
            "response": {**source[-1]["response"], "output": [reasoning, message]},
        }
    )
    for sequence, event in enumerate(events):
        event["sequence_number"] = sequence
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["summary", "content"])
async def test_native_reasoning_parts_are_saved_and_replayed_once(field):
    upstream = _responses_events(["First part.", "Second part."], field=field)
    async with _harness("responses", lambda _: (200, upstream)) as (send, bodies, _):
        saved = await _saved_reply(
            send("messages", [{"role": "user", "content": "hello"}]), "messages"
        )
        await _saved_reply(
            send("messages", [*saved, {"role": "user", "content": "next"}]), "messages"
        )
    for text in ("First part.", "Second part."):
        assert json.dumps(saved).count(text) == 1
        assert json.dumps(bodies[-1]).count(text) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["summary", "content"])
async def test_completed_only_reasoning_keeps_final_answer_open(field):
    upstream = _responses_events(
        ["Completed thought."],
        field=field,
        streamed=False,
        late=True,
        signature="cipher",
    )
    async with _harness("responses", lambda _: (200, upstream)) as (send, _, _):
        frames = [
            frame
            async for frame in send("messages", [{"role": "user", "content": "hello"}])
        ]
    events = parse_sse_text("".join(frames))
    starts = {
        event.data["index"]: event.data["content_block"]["type"]
        for event in events
        if event.event == "content_block_start"
    }
    stops = [
        event.data["index"] for event in events if event.event == "content_block_stop"
    ]
    assert starts[stops[-1]] == "text"
    assert "Answer." in "".join(frames)
    assert "Completed thought." in "".join(frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_chat_secondary_reasoning_prefix_is_not_repeated(wire):
    upstream = _chunks(
        [
            (
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "summary": "Short summary.",
                            "index": 0,
                        }
                    ]
                },
                None,
            ),
            ({"reasoning_content": "Full "}, None),
            ({"content": "Answer."}, None),
            ({"reasoning_content": "thought."}, "stop"),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    assert json.dumps(saved).count("Full ") == 1
    assert "Full thought." in json.dumps(saved)
    assert json.dumps(bodies[-1]).count("Full ") == 1


@pytest.mark.asyncio
async def test_independent_native_items_keep_their_own_text_and_signature():
    upstream = _responses_events(
        ["Alpha."], field="content", signature="cipher-a", late=True
    )
    second = {
        "type": "reasoning",
        "id": "rs_second",
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Beta."}],
        "encrypted_content": "cipher-b",
    }
    upstream[3:3] = [
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                **second,
                "status": "in_progress",
                "content": [],
                "encrypted_content": None,
            },
        },
        {
            "type": "response.reasoning_text.delta",
            "item_id": second["id"],
            "output_index": 2,
            "content_index": 0,
            "delta": "Beta.",
        },
    ]
    upstream.insert(
        -1, {"type": "response.output_item.done", "output_index": 2, "item": second}
    )
    upstream[-1]["response"]["output"].append(second)
    for sequence, event in enumerate(upstream):
        event["sequence_number"] = sequence
    async with _harness("responses", lambda _: (200, upstream)) as (send, _, _):
        saved = await _saved_reply(
            send("messages", [{"role": "user", "content": "hello"}]), "messages"
        )
    thinking = [
        (block["thinking"], block.get("signature"))
        for block in saved[0]["content"]
        if block["type"] == "thinking"
    ]
    assert thinking == [("Alpha.", "cipher-a"), ("Beta.", "cipher-b")]


@pytest.mark.asyncio
async def test_native_signed_empty_thinking_keeps_its_block_kind():
    upstream = _responses_events(
        [""], field="content", streamed=False, signature="empty-signed"
    )
    async with _harness("responses", lambda _: (200, upstream)) as (send, _, _):
        saved = await _saved_reply(
            send("messages", [{"role": "user", "content": "hello"}]), "messages"
        )
    assert saved[0]["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "empty-signed",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("native_alias", [False, True])
async def test_chat_signed_details_keep_their_own_text_on_replay(
    wire, fragmented, native_alias
):
    deltas: list[tuple[JsonObject, str | None]]
    if fragmented:
        deltas = [
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "Al", "index": 0}
                    ]
                },
                None,
            ),
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "pha.", "index": 0},
                        {"type": "reasoning.text", "text": "Beta.", "index": 1},
                    ]
                },
                None,
            ),
            ({"content": "Answer."}, None),
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.text", "signature": "sig-", "index": 0}
                    ]
                },
                None,
            ),
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.text", "signature": "alpha", "index": 0},
                        {"type": "reasoning.text", "signature": "sig-beta", "index": 1},
                    ]
                },
                "stop",
            ),
        ]
        aliases = ["Al", "pha.Beta."]
    else:
        deltas = [
            (
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "Alpha.",
                            "signature": "sig-alpha",
                            "index": 0,
                        },
                        {
                            "type": "reasoning.text",
                            "text": "Beta.",
                            "signature": "sig-beta",
                            "index": 1,
                        },
                    ]
                },
                None,
            ),
            ({"content": "Answer."}, "stop"),
        ]
        aliases = ["Alpha.Beta."]
    if native_alias:
        for (delta, _), text in zip(deltas, aliases, strict=False):
            delta["reasoning_content"] = text
    async with _harness("chat", lambda _: (200, _chunks(deltas))) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    if wire == "messages":
        pairs = [
            (block["thinking"], block.get("signature"))
            for block in saved[0]["content"]
            if block["type"] == "thinking"
        ]
    else:
        pairs = [
            (
                "".join(part["text"] for part in item.get("content", [])),
                item.get("encrypted_content"),
            )
            for item in saved
            if item["type"] == "reasoning"
        ]
    assert pairs == [("Alpha.", "sig-alpha"), ("Beta.", "sig-beta")]
    details = bodies[-1]["messages"][0]["reasoning_details"]
    assert [(detail["text"], detail.get("signature")) for detail in details] == pairs
    assert json.dumps(bodies[-1]).count("Alpha.") == 1
    assert json.dumps(bodies[-1]).count("Beta.") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_chat_explicit_signed_empty_text_retains_its_kind(wire):
    upstream = _chunks(
        [
            (
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "",
                            "signature": "empty-signature",
                            "index": 0,
                        }
                    ]
                },
                None,
            ),
            ({"content": "Answer."}, "stop"),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    if wire == "messages":
        assert {
            "type": "thinking",
            "thinking": "",
            "signature": "empty-signature",
        } in saved[0]["content"]
    else:
        reasoning = next(item for item in saved if item["type"] == "reasoning")
        assert reasoning["content"] == [{"type": "reasoning_text", "text": ""}]
        assert reasoning["encrypted_content"] == "empty-signature"
    assert bodies[-1]["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "", "signature": "empty-signature"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("after_answer", [False, True])
@pytest.mark.parametrize("fragmented", [False, True])
async def test_native_first_alias_preserves_later_signed_details(
    wire, after_answer, fragmented
):
    details = [
        {
            "type": "reasoning.text",
            "text": "Alpha.",
            "signature": "sig-alpha",
            "index": 0,
        },
        {
            "type": "reasoning.text",
            "text": "Beta.",
            "signature": "sig-beta",
            "index": 1,
        },
    ]
    typed = (
        [
            ([{"type": "reasoning.text", "text": "Al", "index": 0}], None),
            (
                [{**details[0], "text": "pha."}, details[1]],
                "stop" if after_answer else None,
            ),
        ]
        if fragmented
        else [(details, "stop" if after_answer else None)]
    )
    deltas = [({"reasoning": "Alpha.Beta."}, None)]
    if after_answer:
        deltas.append(({"content": "Answer."}, None))
    deltas.extend(({"reasoning_details": parts}, finish) for parts, finish in typed)
    if not after_answer:
        deltas.append(({"content": "Answer."}, "stop"))
    async with _harness(
        "chat",
        lambda _: (200, _chunks(deltas)),
        chat_provider_factory=lambda: KiloProvider(
            make_provider_config(api_key="a", base_url="https://provider.invalid/v1"),
            admission=immediate_admission(),
        ),
    ) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    replayed = bodies[-1]["messages"][0]["reasoning_details"]
    assert [(part.get("text"), part.get("signature")) for part in replayed] == [
        ("Alpha.", "sig-alpha"),
        ("Beta.", "sig-beta"),
    ]
    assert json.dumps(saved).count("Alpha.") == 1
    assert json.dumps(saved).count("Beta.") == 1
    assert "Alpha.Beta." not in json.dumps(saved)
    assert json.dumps(bodies[-1]).count("Alpha.") == 1
    assert json.dumps(bodies[-1]).count("Beta.") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_identical_signed_details_keep_independent_ownership(wire):
    upstream = _chunks(
        [
            ({"reasoning_content": "Same.Same."}, None),
            ({"content": "Answer."}, None),
            (
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "Same.",
                            "signature": "a",
                            "index": 0,
                        },
                        {
                            "type": "reasoning.text",
                            "text": "Same.",
                            "signature": "b",
                            "index": 1,
                        },
                    ]
                },
                "stop",
            ),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    assert bodies[-1]["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Same.", "signature": "a"},
        {"type": "reasoning.text", "text": "Same.", "signature": "b"},
    ]


@pytest.mark.parametrize("typed", [False, True])
def test_typed_reasoning_streams_while_native_fallback_waits(typed):
    output = AnthropicChatStreamOutput(
        message_id="msg_test", model="test", input_tokens=1
    )
    reasoning = StructuredReasoningStream()
    output.reasoning_replay = reasoning
    early = list(
        reasoning.events(
            {
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "Plan.", "index": 0}
                ]
            }
            if typed
            else {},
            output,
            native_reasoning="Plan.",
        )
    )
    assert ("Plan." in "".join(early)) == typed
    answer = [*output.ensure_text_block(), output.emit_text_delta("Answer.")]
    assert "Answer." in "".join(answer)
    complete = output.finish_success(
        stop_reason="end_turn", usage=ChatStreamUsage(input_tokens=1, output_tokens=3)
    )
    events = parse_sse_text("".join(early + answer + complete))
    assert (
        sum(event.data.get("delta", {}).get("thinking") == "Plan." for event in events)
        == 1
    )
    starts = {
        event.data["index"]: event.data["content_block"]["type"]
        for event in events
        if event.event == "content_block_start"
    }
    stops = [
        event.data["index"] for event in events if event.event == "content_block_stop"
    ]
    assert starts[stops[-1]] == "text"
