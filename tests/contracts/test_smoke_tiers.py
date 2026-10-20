import json
from pathlib import Path

import pytest

from free_claude_code.core.anthropic.stream_contracts import SSEEvent
from smoke.lib.e2e import assert_native_thinking_stream
from smoke.lib.outcomes import classify_outcome, is_upstream_unavailable_text
from smoke.lib.report_summary import format_summary, summarize_reports
from smoke.lib.skips import (
    skip_if_upstream_unavailable_events,
    skip_if_upstream_unavailable_exception,
)


def test_smoke_report_summary_counts_regression_classes(tmp_path: Path) -> None:
    report = {
        "outcomes": [
            {"classification": "missing_env"},
            {"classification": "product_failure"},
            {"classification": "upstream_unavailable"},
        ]
    }
    (tmp_path / "report-one.json").write_text(json.dumps(report), encoding="utf-8")

    summary = summarize_reports(tmp_path)

    assert summary.reports == 1
    assert summary.outcomes == 3
    assert summary.classifications["product_failure"] == 1
    assert summary.has_regression
    assert "status=regression" in format_summary(summary)


def test_target_disabled_skip_is_not_missing_env() -> None:
    classification = classify_outcome(
        nodeid="smoke/product/test_api_product_live.py::test_api_basic_conversation_e2e",
        outcome="skipped",
        detail="Skipped: smoke target disabled: api",
    )

    assert classification == "target_disabled"


def test_explicit_missing_env_skip_wins_over_network_words() -> None:
    classification = classify_outcome(
        nodeid="smoke/prereq/test_local_provider_endpoints_prereq_live.py::"
        "test_ollama_endpoint_prereq_live",
        outcome="skipped",
        detail="Skipped: missing_env: ollama local server is not reachable: timed out",
    )

    assert classification == "missing_env"


def _upstream_unavailable_events() -> list[SSEEvent]:
    return [
        SSEEvent("message_start", {"message": {"content": []}}, ""),
        SSEEvent(
            "content_block_start",
            {"index": 0, "content_block": {"type": "text", "text": ""}},
            "",
        ),
        SSEEvent(
            "content_block_delta",
            {
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "Upstream provider MISTRAL returned HTTP 429.",
                },
            },
            "",
        ),
        SSEEvent("content_block_stop", {"index": 0}, ""),
        SSEEvent("message_delta", {"delta": {"stop_reason": "end_turn"}}, ""),
        SSEEvent("message_stop", {}, ""),
    ]


def test_provider_error_text_stream_is_upstream_unavailable_skip() -> None:
    with pytest.raises(pytest.skip.Exception) as excinfo:
        skip_if_upstream_unavailable_events(_upstream_unavailable_events())

    assert "upstream_unavailable" in str(excinfo.value)


def test_native_thinking_probe_skips_upstream_unavailable_stream() -> None:
    with pytest.raises(pytest.skip.Exception) as excinfo:
        assert_native_thinking_stream(
            _upstream_unavailable_events(), context="Mistral smoke"
        )

    assert "upstream_unavailable" in str(excinfo.value)


def test_native_thinking_probe_skips_terminal_upstream_error() -> None:
    events = [
        SSEEvent("message_start", {"message": {"content": []}}, ""),
        SSEEvent(
            "error",
            {"error": {"message": "Upstream provider MISTRAL returned HTTP 503."}},
            "",
        ),
    ]

    with pytest.raises(pytest.skip.Exception) as excinfo:
        assert_native_thinking_stream(events, context="Mistral smoke")

    assert "upstream_unavailable" in str(excinfo.value)


def test_non_200_stream_transient_is_upstream_unavailable_skip() -> None:
    exc = AssertionError(
        "stream request failed: HTTP 429 Upstream provider MISTRAL returned HTTP 429."
    )

    with pytest.raises(pytest.skip.Exception) as excinfo:
        skip_if_upstream_unavailable_exception(exc)

    assert "upstream_unavailable" in str(excinfo.value)


def test_deterministic_provider_error_is_not_upstream_unavailable() -> None:
    detail = (
        "Upstream provider GROQ returned HTTP 400: "
        "property 'reasoning_content' is unsupported"
    )

    assert not is_upstream_unavailable_text(detail)
    assert (
        classify_outcome(nodeid="groq_reasoning", outcome="failed", detail=detail)
        == "product_failure"
    )


@pytest.mark.parametrize(
    "detail",
    (
        "Upstream provider GROQ returned HTTP 429.",
        "Upstream provider GROQ returned HTTP 529.",
        "httpx.ConnectError: connection refused",
    ),
)
def test_failed_transient_provider_errors_are_upstream_unavailable(detail: str) -> None:
    assert is_upstream_unavailable_text(detail)
    assert (
        classify_outcome(nodeid="provider", outcome="failed", detail=detail)
        == "upstream_unavailable"
    )
