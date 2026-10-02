"""Recovery must preserve one public response across model changes."""

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    execution_failure,
    fallback_client,
    messages_payload,
    responses_payload,
    responses_text_stream,
    text_stream,
)


@pytest.mark.parametrize("wire_api", ["messages", "responses"])
def test_fallback_continues_committed_text_in_one_response(wire_api: str) -> None:
    prefix, suffix = "First part. ", "Remaining part."
    model = "nvidia_nim/primary-model"
    primary = ControlledFallbackProvider(
        chunks_before_failure=tuple(text_stream(prefix, model=model)[:3]),
        responses_chunks_before_failure=tuple(
            responses_text_stream(prefix, model=model)[:4]
        ),
        failure=execution_failure("upstream disconnected"),
    )
    fallback = ControlledFallbackProvider(text=suffix)
    payload = (
        messages_payload(stream=True) if wire_api == "messages" else responses_payload()
    )

    with fallback_client(primary, fallback) as client:
        response = client.post(f"/v1/{wire_api}", json=payload)

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    if wire_api == "messages":
        text = "".join(
            event.data["delta"]["text"]
            for event in events
            if event.data.get("delta", {}).get("type") == "text_delta"
        )
        assert text == prefix + suffix
        names = [event.event for event in events]
        assert names.count("message_start") == names.count("message_stop") == 1
        assert "error" not in names
    else:
        text = "".join(
            event.data["delta"]
            for event in events
            if event.event == "response.output_text.delta"
        )
        assert text == prefix + suffix
        names = [event.event for event in events]
        assert names.count("response.created") == names.count("response.completed") == 1
        assert "response.failed" not in names
        response_ids = {
            event.data["response"]["id"]
            for event in events
            if isinstance(event.data.get("response"), dict)
        }
        assert len(response_ids) == 1
    assert fallback.stream_models == ["fallback-model"]
    assert primary.close_calls == fallback.close_calls == 1


def test_interrupted_tool_input_is_not_exposed_before_recovery() -> None:
    model = "nvidia_nim/primary-model"
    prefix = tuple(text_stream("Before tool. ", model=model)[:4])
    pending_tool = (
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "unpublished_tool",
                    "name": "inspect_value",
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"value":"'},
            },
        ),
    )
    primary = ControlledFallbackProvider(
        chunks_before_failure=prefix + pending_tool,
        failure=execution_failure("upstream disconnected"),
    )
    fallback = ControlledFallbackProvider(text="Recovered.")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/messages", json=messages_payload(stream=True))

    assert "unpublished_tool" not in response.text
    events = parse_sse_text(response.text)
    text = "".join(
        event.data["delta"]["text"]
        for event in events
        if event.data.get("delta", {}).get("type") == "text_delta"
    )
    assert text == "Before tool. Recovered."
    assert fallback.stream_models == ["fallback-model"]
