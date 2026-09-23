"""Only SSE line terminators split frames; Unicode text stays verbatim."""

import json

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder


@pytest.mark.parametrize("newline", ["\r", "\n", "\r\n"])
@pytest.mark.parametrize("width", range(1, 8))
def test_incremental_sse_terminators_preserve_unicode_text(
    newline: str, width: int
) -> None:
    payload = {"type": "text_delta", "text": "left\u2028middle\u2029right\u0085end"}
    frame = f"event: content_block_delta{newline}data: {json.dumps(payload, ensure_ascii=False)}{newline}{newline}"
    raw = frame * 2
    assert [event.data for event in parse_sse_text(raw)] == [payload, payload]
    decoder = AnthropicSSEDecoder()
    events = []
    for start in range(0, len(raw), width):
        events.extend(decoder.feed(raw[start : start + width]))
    assert [event.data for event in events] == [payload, payload]
    assert decoder.finish() == ()
