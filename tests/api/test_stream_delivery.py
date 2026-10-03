"""Public envelope observation preserves ordinary frames and invisible retries."""

import asyncio
import json
from decimal import Decimal

import pytest
import simplejson
from starlette.responses import StreamingResponse

from free_claude_code.api.handlers.classifier_response import classifier_response
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.stream_delivery import current_stream_delivery
from tests.api.test_response_streams import _serve
from tests.api.test_tool_call_buffer import _call, _end, _frames, _response, _start
from tests.api.test_web_server_tools import (
    ScriptedSelectionProvider,
    _automatic_search_request,
    _automatic_search_service,
    _provider_text_events,
)


async def drained(body, wire):
    response = await _response(wire, body)
    try:
        return "".join([str(chunk) async for chunk in response.body_iterator])
    finally:
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending", ["activity", "empty", "stop", "error", "raise", "eof"]
)
async def test_messages_start_is_published_only_with_committed_activity(ending):
    start = _frames("messages", [_start("messages")])[0]
    harmless = ': keepalive\r\n\r\nevent: ping\r\ndata: {"type":"ping"}\r\n\r\n'
    activity = (
        'event: future\ndata: {"type":"future","extension":0.1234567890123456789}\n\n'
    )

    async def source():
        yield start
        yield harmless
        if ending == "activity":
            yield activity
        if ending in {"activity", "empty"}:
            yield "".join(_frames("messages", _end("messages")))
        elif ending == "stop":
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        elif ending == "error":
            yield "".join(_frames("messages", _end("messages", failed=True)))
        elif ending == "raise":
            raise RuntimeError("stream interrupted")

    response = await _response("messages", source())
    try:
        iterator = aiter(response.body_iterator)
        assert await anext(iterator) == harmless
        remaining = "".join([str(chunk) async for chunk in iterator])
        if ending in {"activity", "empty", "stop"}:
            assert remaining.startswith(start)
            assert parse_sse_text(remaining)[-1].event == "message_stop"
            if ending == "activity":
                assert activity in remaining
        else:
            assert "message_start" not in remaining
            if ending != "eof":
                assert parse_sse_text(remaining)[-1].event == "error"
    finally:
        await response.aclose()


@pytest.mark.asyncio
async def test_messages_deferral_preserves_duplicate_starts_within_one_attempt():
    raw = "".join(
        _frames("messages", [_start("messages"), _start("messages"), *_end("messages")])
    )

    async def source():
        yield raw

    assert await drained(source(), "messages") == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
@pytest.mark.parametrize("ending", ["\r", "\n", "\r\n"])
async def test_unicode_metadata_preserves_bytes_and_retry_eligibility(
    wire, separator, ending
):
    start = _start(wire)
    start[1]["message" if wire == "messages" else "response"]["extension"] = (
        "before" + separator + "after"
    )
    raw = _frames(wire, [start])[0].replace("\n", ending)
    released = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        yield raw
        released.append(state.content_released)
        yield "".join(_frames(wire, _end(wire)))

    result = await drained(source(), wire)
    assert released == [False]
    assert result.startswith(raw)
    payload = parse_sse_text(result)[0].data
    assert payload["message" if wire == "messages" else "response"]["extension"] == (
        "before" + separator + "after"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
async def test_unicode_comment_does_not_release_content(separator):
    raw = ": before" + separator + "after\r\n\r\n"
    released = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        yield raw
        released.append(state.content_released)
        yield "".join(_frames("messages", [_start("messages"), *_end("messages")]))

    assert (await drained(source(), "messages")).startswith(raw)
    assert released == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
async def test_unicode_meaningful_frame_still_releases_content(separator):
    raw = _frames("messages", [_start("messages")])[0]
    raw += 'event: future\ndata: {"type":"future","text":"a' + separator + 'b"}\n\n'
    released = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        yield raw
        released.append(state.content_released)
        yield "".join(_frames("messages", _end("messages")))

    assert (await drained(source(), "messages")).startswith(raw)
    assert released == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"type": {"future": 7}},
        {"type": ["future"]},
        {
            "type": "response.created",
            "response": {"status": {"future": 7}, "output": []},
        },
    ],
)
async def test_opaque_frame_closes_eligibility_without_changing_bytes(payload):
    raw = _frames("responses", [_start("responses")])[0]
    raw += "event: future\ndata: " + json.dumps(payload) + "\n\n"
    seen = []

    async def source():
        seen.append(current_stream_delivery())
        yield raw

    assert await drained(source(), "responses") == raw
    assert seen[0].content_released


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_retry_drops_hidden_group_tail_and_reused_call_aliases(wire):
    async def source():
        delivery = current_stream_delivery()
        assert delivery is not None
        delivery.begin_attempt()
        yield "".join(_frames(wire, [_start(wire)]))
        first, second = _call(wire, 0), _call(wire, 1)
        yield "".join(_frames(wire, [first[0], second[0], *first[1:], second[1]]))
        yield 'event: content_block_delta\ndata: {"index":1,'
        assert not delivery.content_released
        delivery.begin_attempt()
        winner = _call(wire, 0, '{"path":"winning"}')
        yield "".join(_frames(wire, [_start(wire), *winner, *_end(wire)]))

    output = await drained(source(), wire)
    assert "call_1" not in output
    assert "winning" in output
    assert ("message_stop" if wire == "messages" else "response.completed") in output
    assert (
        output.count(
            '"type": "message_start"'
            if wire == "messages"
            else '"type": "response.created"'
        )
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["", "\u2028", "\u2029", "\x85"])
async def test_retry_sequence_offset_preserves_decimal_extensions_and_call_ids(
    separator,
):
    async def source():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        yield 'event: response.created\r\ndata: {"type":"response.created","sequence_number":40,"response":{"id":"resp_old","created_at":7,"status":"in_progress","output":[]}}\r\n\r\n'
        state.begin_attempt()
        yield 'event: response.created\r\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"resp_new","created_at":9,"status":"in_progress","output":[]}}\r\n\r\n'
        yield (
            'id: native-event\r\nevent: response.completed\r\ndata: {"type":"response.completed","sequence_number":3,"response_id":"resp_new","response":{"id":"resp_new","created_at":9,"status":"completed","output":[]},"native":1.234567890123456789,"call_id":"resp_new","extension":"a'
            + separator
            + 'b"}\r\n\r\n'
        )

    output = await drained(source(), "responses")
    payload = simplejson.loads(output.split("data: ")[-1], use_decimal=True)
    assert payload["sequence_number"] == 41
    assert payload["response_id"] == payload["response"]["id"] == "resp_old"
    assert payload["response"]["created_at"] == 7
    assert payload["call_id"] == "resp_new"
    assert payload["native"] == Decimal("1.234567890123456789")
    assert payload["extension"] == "a" + separator + "b"
    assert "id: native-event\r\n" in output
    assert output.endswith("\r\n\r\n")


@pytest.mark.asyncio
async def test_classifier_resets_projection_when_hidden_attempt_is_replaced():
    async def source():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        yield "".join(_frames("messages", [_start("messages")]))
        yield 'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"hidden"}}\n\n'
        yield "event: content_block_delta\ndata: {"
        state.begin_attempt()
        yield "".join(
            _frames(
                "messages",
                [_start("messages"), *_call("messages", 0), *_end("messages")],
            )
        )

    output = await drained(classifier_response(source()), "messages")
    assert "hidden" not in output
    assert "call_0" in output and "message_stop" in output
    assert '"index": 0' in output


@pytest.mark.asyncio
async def test_prefetch_read_and_close_can_migrate_tasks_without_context_leak():
    seen = []
    closed = []

    async def source():
        state = current_stream_delivery()
        seen.append(state)
        try:
            yield "".join(_frames("messages", [_start("messages")]))
            assert current_stream_delivery() is state
            yield "".join(_frames("messages", _end("messages")))
        finally:
            closed.append(current_stream_delivery())

    response = await asyncio.create_task(_response("messages", source()))
    assert current_stream_delivery() is None
    iterator = aiter(response.body_iterator)

    async def read():
        return await anext(iterator)

    async def finish():
        return [chunk async for chunk in iterator]

    await asyncio.create_task(read())
    await asyncio.create_task(finish())
    await asyncio.create_task(response.aclose())
    assert closed == seen and current_stream_delivery() is None


@pytest.mark.asyncio
async def test_private_web_search_masks_public_observer_during_read_and_close():
    seen = []
    closed = []

    class ObservedSelection(ScriptedSelectionProvider):
        async def stream_messages(self, *args, **kwargs):
            seen.append(current_stream_delivery())
            try:
                async for frame in super().stream_messages(*args, **kwargs):
                    yield frame
            finally:
                closed.append(current_stream_delivery())

    provider = ObservedSelection(_provider_text_events("private decision"))
    response = await _automatic_search_service(provider).create(
        _automatic_search_request()
    )
    assert isinstance(response, StreamingResponse)
    messages = await _serve(response)
    output = b"".join(message.get("body", b"") for message in messages).decode()
    assert "private decision" in output
    assert seen == closed == [None]
    assert len(provider.requests) == 1 and provider.close_count == 1
    assert current_stream_delivery() is None


@pytest.mark.asyncio
async def test_concurrent_responses_do_not_share_delivery_or_call_aliases():
    states = {}
    both = asyncio.Event()
    other_done = asyncio.Event()

    async def source(label):
        state = current_stream_delivery()
        assert state is not None
        states[label] = state
        state.begin_attempt()
        yield "".join(_frames("messages", [_start("messages")]))
        call = _call("messages", 0, '{"path":"' + label + '"}')
        yield "".join(_frames("messages", call[:2]))
        if len(states) == 2:
            both.set()
        await both.wait()
        if label == "alpha":
            await other_done.wait()
            assert not state.content_released
            state.begin_attempt()
            yield "".join(
                _frames("messages", [_start("messages"), *call, *_end("messages")])
            )
        else:
            yield "".join(_frames("messages", [call[2], *_end("messages")]))
            other_done.set()

    alpha, beta = await asyncio.gather(
        drained(source("alpha"), "messages"), drained(source("beta"), "messages")
    )
    assert states["alpha"] is not states["beta"]
    assert "alpha" in alpha and "beta" not in alpha
    assert "beta" in beta and "alpha" not in beta
    assert current_stream_delivery() is None
