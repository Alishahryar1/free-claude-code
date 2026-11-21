"""Gemini delivery evidence drives the existing recovery decisions."""

from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_output import CompletedCall
from tests.wire_payload import array_payload, object_payload


def event(*parts: JsonObject, finish: str | None = None) -> JsonObject:
    candidate: JsonObject = {
        "index": 0,
        "content": {"role": "model", "parts": list(parts)},
    }
    if finish:
        candidate["finishReason"] = finish
    return {"responseId": "old", "modelVersion": "public", "candidates": [candidate]}


def test_delivered_gemini_text_and_complete_calls_have_native_evidence():
    delivered = DeliveredResponse("gemini")
    delivered.observe(event({"text": "answer"}, {"text": "thought", "thought": True}))
    delivered.observe(event({"functionCall": {"id": "a", "name": "read", "args": {}}}))
    delivered.complete_calls((CompletedCall("gemini", (0, 0), "a"),))
    prefix = delivered.snapshot()
    assert prefix.text == "answer"
    assert prefix.thinking == "thought"
    assert prefix.has_calls and prefix.can_handoff
    assert delivered.progress == 2
    assert delivered.closing_events() == []
    assert (
        object_payload(array_payload(delivered.handoff_events()[0]["candidates"])[0])[
            "finishReason"
        ]
        == "STOP"
    )


def test_gemini_continuation_removes_overlap_without_fabricating_usage():
    delivered = DeliveredResponse("gemini")
    delivered.observe(event({"text": "Hello world"}))
    continuation = ContinuationStream(delivered)
    continuation.begin(handoff=False)
    assert continuation.prepare(dict(event({"text": "world"}))) == []
    values = continuation.prepare(dict(event({"text": " again"})))
    for value in values:
        delivered.observe(value)
    terminal = continuation.finalize(dict(event(finish="STOP")))
    assert delivered.snapshot().text == "Hello world again"
    assert "usageMetadata" not in terminal


def test_gemini_json_aggregation_preserves_thoughts_and_late_signature():
    aggregate = GeminiResponseAccumulator()
    aggregate.feed(event({"text": "a"}))
    aggregate.feed(event({"text": "b"}))
    aggregate.feed(event({"thoughtSignature": "opaque"}, finish="STOP"))
    assert object_payload(
        object_payload(array_payload(aggregate.result()["candidates"])[0])["content"]
    )["parts"] == [
        {"text": "ab"},
        {"thoughtSignature": "opaque"},
    ]
