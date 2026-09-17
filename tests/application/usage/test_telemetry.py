"""Unit tests for StreamTelemetryTracker and token extraction."""

import json
import time

from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.application.usage.telemetry import StreamTelemetryTracker


def test_telemetry_tracker_records_ttft():
    target = ProviderModelTarget(
        provider_id="groq",
        provider_model="llama-3.3-70b-versatile",
        provider_model_ref="groq:llama-3.3-70b-versatile",
    )
    tracker = StreamTelemetryTracker(
        request_id="req-123",
        wire_api="messages",
        gateway_model="claude-3-5-sonnet",
        target=target,
        is_primary=True,
    )
    time.sleep(0.01)
    tracker.on_chunk("event: message_start\ndata: {}\n\n")
    assert tracker._mono_first_token is not None

    record = tracker.on_success()
    assert record.request_id == "req-123"
    assert record.provider_id == "groq"
    assert record.is_primary is True
    assert record.ttft_ms is not None
    assert record.status == "success"


def test_telemetry_tracker_extracts_anthropic_sse_tokens():
    target = ProviderModelTarget(
        provider_id="anthropic",
        provider_model="claude-3-5-sonnet-20241022",
        provider_model_ref="anthropic:claude-3-5-sonnet-20241022",
    )
    tracker = StreamTelemetryTracker(
        request_id="req-456",
        wire_api="messages",
        gateway_model="claude-3-5-sonnet",
        target=target,
        is_primary=True,
    )

    start_payload = {
        "type": "message_start",
        "message": {
            "usage": {
                "input_tokens": 120,
                "output_tokens": 1,
                "cache_creation_input_tokens": 50,
            }
        },
    }
    tracker.on_chunk(f"event: message_start\ndata: {json.dumps(start_payload)}\n\n")

    delta_payload = {
        "type": "message_delta",
        "usage": {
            "output_tokens": 85,
        },
    }
    tracker.on_chunk(f"event: message_delta\ndata: {json.dumps(delta_payload)}\n\n")

    record = tracker.on_success()
    assert record.input_tokens == 120
    assert record.output_tokens == 85
    assert record.cached_tokens == 50
    assert record.total_tokens == 205
    assert record.is_estimated is False


def test_telemetry_tracker_sums_cache_token_partitions():
    target = ProviderModelTarget(
        provider_id="anthropic",
        provider_model="claude-3-5-sonnet-20241022",
        provider_model_ref="anthropic:claude-3-5-sonnet-20241022",
    )
    tracker = StreamTelemetryTracker(
        request_id="req-cache-sum",
        wire_api="messages",
        gateway_model="claude-3-5-sonnet",
        target=target,
        is_primary=True,
    )

    start_payload = {
        "type": "message_start",
        "message": {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 1,
                "cache_read_input_tokens": 17,
                "cache_creation_input_tokens": 29,
            }
        },
    }
    tracker.on_chunk(f"event: message_start\ndata: {json.dumps(start_payload)}\n\n")

    record = tracker.on_success()
    assert record.cached_tokens == 46  # 17 + 29


def test_telemetry_tracker_fallback_estimation_when_tokens_missing():
    target = ProviderModelTarget(
        provider_id="ollama",
        provider_model="qwen2.5-coder:7b",
        provider_model_ref="ollama:qwen2.5-coder:7b",
    )
    tracker = StreamTelemetryTracker(
        request_id="req-789",
        wire_api="messages",
        gateway_model="qwen",
        target=target,
        is_primary=False,
        fallback_index=1,
        fallback_from_ref="groq:llama-3.3-70b-versatile",
        fallback_reason="rate_limit",
    )

    tracker.on_chunk("chunk 1 text content ")
    tracker.on_chunk("chunk 2 more text content")

    record = tracker.on_success()
    assert record.fallback_index == 1
    assert record.fallback_from_ref == "groq:llama-3.3-70b-versatile"
    assert record.fallback_reason == "rate_limit"
    assert record.output_tokens > 0
    assert record.is_estimated is True
