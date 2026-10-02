"""Native boundary rules that must hold before admission and local estimation."""

from copy import deepcopy

import pytest

from free_claude_code.core.anthropic import (
    MessagesRequest,
    NativeTokenCountRequest,
    get_token_count,
)
from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.anthropic.passthrough import NativeMessagesPassthrough


@pytest.mark.parametrize(
    "block",
    [
        {"type": "not-ping"},
        {"type": "ping", "extra": float("nan")},
        {"type": "ping", "extra": "\ud800"},
    ],
)
def test_suppressed_leading_ping_still_validates_envelope(block):
    relay = NativeMessagesPassthrough("alias")
    with pytest.raises(NativeMessagesError):
        relay.feed("ping", block)
    assert not relay.started


@pytest.mark.parametrize("in_tool", [False, True])
def test_native_count_retains_existing_text_and_image_estimation(in_tool):
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "A" * 300_000},
    }
    text = {"type": "text", "text": "Screenshot captured"}

    def payload(with_image):
        blocks = [text, image] if with_image else [text]
        if in_tool:
            blocks = [{"type": "tool_result", "tool_use_id": "t1", "content": blocks}]
        return {
            "model": "test",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": blocks}],
            "system": [{"type": "text", "text": "Describe the screenshot"}],
        }

    body = payload(True)
    original = deepcopy(body)
    native = NativeTokenCountRequest.model_validate(body)
    typed = MessagesRequest.model_validate(body)
    without_image = NativeTokenCountRequest.model_validate(payload(False))
    counted = get_token_count(native.messages, native.system)
    assert counted == get_token_count(typed.messages, typed.system)
    assert (
        85
        <= counted - get_token_count(without_image.messages, without_image.system)
        < 1000
    )
    assert body == original
