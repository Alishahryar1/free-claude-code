"""Public replay carriers retain the recovery restrictions of their native state."""

import pytest

from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)


def carrier(protocol, native):
    return encode_replay(
        ReplayRecord(ReplayOrigin("test", protocol, "", "", "model"), native)
    )


@pytest.mark.parametrize("surface", ["responses", "signature", "redacted"])
@pytest.mark.parametrize(
    ("value", "eligible"),
    [
        (carrier("responses", {"type": "reasoning", "summary": []}), True),
        (
            carrier(
                "responses",
                {"type": "reasoning", "summary": [], "encrypted_content": None},
            ),
            True,
        ),
        (
            carrier(
                "responses",
                {"type": "reasoning", "summary": [], "encrypted_content": ""},
            ),
            True,
        ),
        (
            carrier(
                "responses",
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Readable."}],
                    "encrypted_content": "cipher",
                },
            ),
            False,
        ),
        (
            carrier(
                "messages",
                {"type": "thinking", "thinking": "Readable.", "signature": "signed"},
            ),
            False,
        ),
        (
            carrier(
                "chat",
                {
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "Readable."}
                    ]
                },
            ),
            False,
        ),
        ("native-ciphertext", False),
        ("fcc:history:v1:malformed", False),
        ("fcc:history:v99:future", False),
    ],
    ids=[
        "absent",
        "null",
        "empty",
        "encrypted",
        "signed",
        "structured",
        "raw",
        "malformed",
        "unknown-version",
    ],
)
def test_public_carriers_follow_native_reasoning_semantics(surface, value, eligible):
    delivered = DeliveredResponse("responses" if surface == "responses" else "messages")
    if surface == "responses":
        delivered.observe(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_public",
                    "summary": [],
                    "encrypted_content": value,
                },
            }
        )
    elif surface == "signature":
        delivered.observe(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "Readable."},
            }
        )
        delivered.observe(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": value},
            }
        )
    else:
        delivered.observe(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": value},
            }
        )
    assert delivered.snapshot().eligible is eligible
