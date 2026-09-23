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


def test_decoder_handles_many_tiny_fragments_without_losing_frames():
    wire = "".join(
        f'event: delta\ndata: {{"index":{index}}}\n\n' for index in range(250)
    )
    decoder = AnthropicSSEDecoder()

    events = tuple(event for character in wire for event in decoder.feed(character))

    assert [event.data["index"] for event in events] == list(range(250))
    assert decoder.finish() == ()
