"""Format native Messages events using source completion supplied by the decoder."""

from collections.abc import Iterator

from free_claude_code.core.history_replay import ReplayRecord, encode_replay
from free_claude_code.core.stream_events import StreamEvent
from free_claude_code.core.stream_observations import (
    DecodedStreamEvent,
    MessagesObservation,
)


def native_messages_events(
    event: DecodedStreamEvent, observation: MessagesObservation, *, opaque: bool
) -> Iterator[StreamEvent]:
    kind, original = event.source.kind, event.source.payload
    body = dict(original)
    completed = observation.completed
    if not opaque:
        block = body.get("content_block")
        if kind == "content_block_start" and isinstance(block, dict):
            if block.get("type") == "thinking":
                body["content_block"] = {**block, "signature": ""}
            elif block.get("type") == "redacted_thinking":
                body["content_block"] = {
                    **block,
                    "data": encode_replay(ReplayRecord(event.origin, block)),
                }
        delta = body.get("delta")
        if (
            kind == "content_block_delta"
            and isinstance(delta, dict)
            and delta.get("type") == "signature_delta"
        ):
            return
        if (
            completed is not None
            and completed.body.get("type") == "thinking"
            and completed.body.get("signature")
        ):
            yield StreamEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": body["index"],
                    "delta": {
                        "type": "signature_delta",
                        "signature": encode_replay(
                            ReplayRecord(event.origin, completed.body)
                        ),
                    },
                },
            )
    yield StreamEvent(kind, body, completed.completion if completed else None)
