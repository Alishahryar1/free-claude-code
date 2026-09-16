"""Public model-fallback contracts across both wire APIs."""

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    execution_failure,
    fallback_client,
    messages_payload,
    responses_created_event,
    responses_payload,
    text_stream,
)


def test_messages_fallback_emits_one_lifecycle_with_original_model() -> None:
    primary = ControlledFallbackProvider(
        failure=execution_failure("primary overloaded")
    )
    fallback = ControlledFallbackProvider(text="fallback worked")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/messages", json=messages_payload(stream=True))

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    event_names = [event.event for event in events]
    assert event_names.count("message_start") == 1
    assert event_names.count("message_stop") == 1
    assert "fallback worked" in response.text
    assert "primary overloaded" not in response.text
    assert fallback.stream_models == ["fallback-model"]
    assert fallback.response_models == ["nvidia_nim/primary-model"]
    assert primary.close_calls == fallback.close_calls == 1


def test_responses_fallback_emits_one_stable_response_lifecycle() -> None:
    primary = ControlledFallbackProvider(
        failure=execution_failure("primary overloaded")
    )
    fallback = ControlledFallbackProvider(text="fallback worked")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    names = [event.event for event in events]
    assert names.count("response.created") == 1
    assert names.count("response.completed") == 1
    assert "response.failed" not in names
    response_ids = {
        event.data["response"]["id"]
        for event in events
        if isinstance(event.data.get("response"), dict)
    }
    assert len(response_ids) == 1
    assert "fallback worked" in response.text


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/v1/messages", messages_payload(stream=True)),
        ("/v1/responses", responses_payload()),
    ),
)
@pytest.mark.parametrize(
    "failure",
    (
        ExecutionFailure(
            kind=FailureKind.PERMISSION,
            status_code=403,
            message="primary quota exhausted",
            retryable=False,
        ),
        ExecutionFailure(
            kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
            status_code=400,
            message="primary context exhausted",
            retryable=False,
        ),
    ),
    ids=["permission", "context_window"],
)
def test_nonretryable_provider_failure_uses_fallback_before_output(
    path: str,
    payload: dict[str, object],
    failure: ExecutionFailure,
) -> None:
    primary = ControlledFallbackProvider(failure=failure)
    fallback = ControlledFallbackProvider(text="fallback worked")

    with fallback_client(primary, fallback) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 200
    assert "fallback worked" in response.text
    assert failure.message not in response.text
    assert primary.close_calls == fallback.close_calls == 1


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/v1/messages", messages_payload(stream=True)),
        ("/v1/responses", responses_payload()),
    ),
)
def test_final_failure_uses_last_candidate_typed_error(
    path: str,
    payload: dict[str, object],
) -> None:
    primary = ControlledFallbackProvider(
        failure=execution_failure("primary overloaded")
    )
    final_failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="fallback rate limited",
        retryable=True,
    )
    fallback = ControlledFallbackProvider(failure=final_failure)

    with fallback_client(primary, fallback) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    payload = response.json()
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["message"] == "fallback rate limited"
    assert "primary overloaded" not in response.text


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/v1/messages", messages_payload(stream=True)),
        ("/v1/responses", responses_payload()),
    ),
)
def test_lazy_fallback_validation_error_remains_ordinary(
    path: str,
    payload: dict[str, object],
) -> None:
    primary = ControlledFallbackProvider(
        failure=execution_failure("primary overloaded")
    )
    fallback = ControlledFallbackProvider(
        validation_error=InvalidRequestError("fallback configuration is invalid")
    )

    with fallback_client(primary, fallback) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 400
    assert "x-should-retry" not in response.headers
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == "fallback configuration is invalid"
    assert fallback.stream_models == []


def test_postframe_failure_never_opens_fallback_for_streaming_messages() -> None:
    first = format_sse_event(
        "message_start",
        {"type": "message_start", "message": {}},
    )
    primary = ControlledFallbackProvider(
        chunks_before_failure=(first,),
        failure=execution_failure("primary failed after start"),
    )
    fallback = ControlledFallbackProvider(text="must not run")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/messages", json=messages_payload(stream=True))

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == ["message_start", "error"]
    assert fallback.stream_models == []


def test_postframe_failure_never_opens_fallback_for_responses() -> None:
    first = responses_created_event(model="nvidia_nim/primary-model")
    primary = ControlledFallbackProvider(
        responses_chunks_before_failure=(first,),
        failure=execution_failure("primary failed after start"),
    )
    fallback = ControlledFallbackProvider(text="must not run")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/responses", json=responses_payload())

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == [
        "response.created",
        "response.failed",
    ]
    assert fallback.stream_models == []


def test_nonstreaming_partial_failure_discards_content_without_fallback() -> None:
    partial_text = "PARTIAL_ASSISTANT_OUTPUT"
    partial = text_stream(partial_text, model="nvidia_nim/primary-model")[:3]
    primary = ControlledFallbackProvider(
        chunks_before_failure=tuple(partial),
        failure=execution_failure("primary stream failed"),
    )
    fallback = ControlledFallbackProvider(text="must not run")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/messages", json=messages_payload(stream=False))

    assert response.status_code == 529
    assert response.headers["x-should-retry"] == "false"
    assert partial_text not in response.text
    assert fallback.stream_models == []
