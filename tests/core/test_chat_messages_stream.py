import pytest

from free_claude_code.core.chat_completions.messages_stream import ChatMessagesStream
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.json_types import JsonObject, load_json_value
from tests.wire_payload import array_payload, object_payload


def _chunks(events: list[JsonObject]) -> list[JsonObject]:
    stream = ChatMessagesStream(
        model="public",
        origin=ReplayOrigin("provider", "messages", "endpoint", "key", "model"),
        reserved_tool_ids=frozenset(),
        include_usage=False,
        include_thoughts=True,
    )
    frames = "".join(
        frame for event in events for frame in stream.feed(str(event["type"]), event)
    )
    return [
        object_payload(load_json_value(line.removeprefix("data: ")))
        for line in frames.splitlines()
        if line.startswith("data: {")
    ]


@pytest.mark.parametrize("signature", [None, "signed"], ids=["unsigned", "signed"])
def test_completed_thinking_replays_only_signed_state(signature: str | None) -> None:
    signature_events: list[JsonObject] = (
        [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            }
        ]
        if signature
        else []
    )
    chunks = _chunks(
        [
            {
                "type": "message_start",
                "message": {"id": "msg", "role": "assistant", "content": []},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "Find 17."},
            },
            *signature_events,
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": "17"},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
    )

    choices = [
        object_payload(choice)
        for chunk in chunks
        for choice in array_payload(chunk.get("choices", []))
    ]
    deltas = [object_payload(choice.get("delta", {})) for choice in choices]
    details = [
        object_payload(detail)
        for delta in deltas
        for detail in array_payload(delta.get("reasoning_details", []))
    ]
    assert "".join(str(d.get("text", "")) for d in details) == "Find 17."
    assert "".join(str(d.get("content") or "") for d in deltas) == "17"
    assert sum(d["type"] == "reasoning.encrypted" for d in details) == int(
        signature is not None
    )
    assert choices[-1]["finish_reason"] == "stop"
