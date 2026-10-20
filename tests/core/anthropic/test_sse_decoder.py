import json
from unittest.mock import patch

import pytest

from free_claude_code.core.anthropic.streaming import AnthropicSSEDecoder


def test_decoder_handles_every_split_and_crlf_boundaries():
    wire = (
        'event: first\r\ndata: {"type":"first"}\r\n\r\n'
        'event: second\ndata: {"type":"second"}\n\n'
    )

    for split in range(len(wire) + 1):
        decoder = AnthropicSSEDecoder()
        events = (*decoder.feed(wire[:split]), *decoder.feed(wire[split:]))
        assert [event.event for event in events] == ["first", "second"]
        assert decoder.finish() == ()


def test_decoder_returns_one_unterminated_final_event():
    decoder = AnthropicSSEDecoder()

    assert decoder.feed('event: final\ndata: {"value": 1}') == ()
    events = decoder.finish()

    assert len(events) == 1
    assert events[0].event == "final"
    assert events[0].data == {"value": 1}
    assert decoder.finish() == ()


@pytest.mark.parametrize("ending", ["\n\n", ""])
def test_event_filter_skips_json_and_preserves_unnamed_events_at_every_split(ending):
    wire = (
        'event: delta\r\ndata: {"text":"ignored"}\r\n\r\n'
        'data: {"type":"error"}\nevent: error\n\n'
        'data: {"type":"response.failed"}\r\n\r\n'
        'event: delta\ndata: {"text":"ignored final"}' + ending
    )
    for split in range(len(wire) + 1):
        decoder = AnthropicSSEDecoder(event_names=frozenset({"error"}))
        with patch(
            "free_claude_code.core.anthropic.stream_contracts.json.loads",
            wraps=json.loads,
        ) as loads:
            events = (
                *decoder.feed(wire[:split]),
                *decoder.feed(wire[split:]),
                *decoder.finish(),
            )
        assert [(event.event, event.data["type"]) for event in events] == [
            ("error", "error"),
            ("", "response.failed"),
        ]
        assert [call.args[0] for call in loads.call_args_list] == [
            '{"type":"error"}',
            '{"type":"response.failed"}',
        ]
        assert decoder.finish() == ()


def test_event_filter_decodes_selected_unterminated_event():
    decoder = AnthropicSSEDecoder(event_names=frozenset({"error"}))
    assert decoder.feed('event: error\ndata: {"type":"error"}') == ()
    assert [event.data for event in decoder.finish()] == [{"type": "error"}]


def test_decoder_handles_many_tiny_fragments_without_losing_frames():
    wire = "".join(
        f'event: delta\ndata: {{"index":{index}}}\n\n' for index in range(250)
    )
    decoder = AnthropicSSEDecoder()

    events = tuple(event for character in wire for event in decoder.feed(character))

    assert [event.data["index"] for event in events] == list(range(250))
    assert decoder.finish() == ()
