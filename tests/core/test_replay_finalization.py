"""Only actual replay fields are sealed against the published response."""

import pytest

from free_claude_code.api.stream_delivery import PublicResponseStream
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.history_replay import (
    HistoryReplayError,
    decode_replay,
    encode_gemini_replay,
    encode_replay,
)
from free_claude_code.core.replay_finalization import finalize_replay
from free_claude_code.core.stream_delivery import StreamDeliveryState
from tests.core.test_gemini_continuation_records import recovered


def test_terminal_snapshot_reuses_sealed_carrier():
    draft = encode_replay(recovered(echo=False))
    delivered = DeliveredResponse("messages")
    delivered.observe(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "prefix suffix"},
        }
    )
    cache = {}
    added = {
        "type": "response.output_item.added",
        "item": {"type": "reasoning", "encrypted_content": draft},
    }
    assert finalize_replay(added, delivered.snapshot, cache)
    sealed = added["item"]["encrypted_content"]
    continuation = decode_replay(sealed).continuation
    assert continuation is not None
    assert continuation.echo == ({"text": "prefix suffix"},)
    completed = {
        "type": "response.completed",
        "response": {"output": [{"type": "reasoning", "encrypted_content": draft}]},
    }

    # A later snapshot must not re-read now-protected public output.
    def unexpected():
        raise AssertionError("Replay was already finalized")

    assert finalize_replay(completed, unexpected, cache)
    assert completed["response"]["output"][0]["encrypted_content"] == sealed


def test_carrier_looking_answer_text_is_not_interpreted():
    payload = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "fcc:history:v1:not-a-carrier"},
    }

    def unexpected():
        raise AssertionError("Ordinary text is not replay metadata")

    assert not finalize_replay(payload, unexpected, {})


def test_missing_public_prefix_prevents_draft_publication():
    payload = {
        "content_block": {
            "type": "redacted_thinking",
            "data": encode_replay(recovered(echo=False)),
        }
    }
    with pytest.raises(HistoryReplayError):
        finalize_replay(payload, DeliveredResponse("messages").snapshot, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_carrier", [False, True])
async def test_disconnected_public_stream_releases_replay_state(publish_carrier):
    import json

    closed = []

    async def source():
        try:
            yield 'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"prefix suffix"}]}}]}\n\n'
            yield (
                "data: "
                + json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {
                                            "thoughtSignature": encode_gemini_replay(
                                                recovered(echo=False)
                                            )
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
                + "\n\n"
            )
        finally:
            closed.append(True)

    stream = PublicResponseStream(
        source(),
        StreamDeliveryState(),
        wire_api="gemini",
        first_chunk="",
        terminal_frame=None,
        terminal_failure_observer=None,
    )
    await anext(stream)
    if publish_carrier:
        await anext(stream)
        assert stream._replay_carriers
    await stream.aclose()
    assert closed == [True]
    assert not stream._replay_carriers
