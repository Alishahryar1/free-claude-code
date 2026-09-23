"""Credential-free product smoke for application-owned model fallback."""

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    execution_failure,
    fallback_client,
    messages_payload,
    responses_payload,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("api")]


@pytest.mark.parametrize(
    ("path", "payload", "start_event", "complete_event"),
    (
        (
            "/v1/messages",
            messages_payload(stream=True),
            "message_start",
            "message_stop",
        ),
        (
            "/v1/responses",
            responses_payload(),
            "response.created",
            "response.completed",
        ),
    ),
)
def test_model_fallback_e2e(
    path: str,
    payload: dict[str, object],
    start_event: str,
    complete_event: str,
) -> None:
    primary = ControlledFallbackProvider(
        failure=execution_failure("controlled primary overload")
    )
    fallback = ControlledFallbackProvider(text="MODEL_FALLBACK_SMOKE_OK")

    with fallback_client(primary, fallback) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 200, response.text
    events = parse_sse_text(response.text)
    names = [event.event for event in events]
    assert names.count(start_event) == 1
    assert names.count(complete_event) == 1
    assert "MODEL_FALLBACK_SMOKE_OK" in response.text
    assert "controlled primary overload" not in response.text
    assert primary.close_calls == fallback.close_calls == 1
