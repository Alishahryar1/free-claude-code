"""Protocol-neutral diagnostic redaction and detail contracts."""

from httpx import ConnectError, HTTPStatusError, Request, Response

from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    UpstreamErrorDetail,
    attach_upstream_error_body,
    exception_cause_types,
    extract_upstream_error_detail,
    format_execution_failure_message,
    format_user_error_preview,
    redact_sensitive_error_text,
    safe_exception_message,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind


def test_redaction_preserves_context_and_covers_recognizable_credentials() -> None:
    sanitized = redact_sensitive_error_text(
        '{"authorization":"Bearer AUTH_TOKEN","api_key":"sk-live-secret-key",'
        '"client_secret":"CLIENT_SECRET"} raw=nvapi-standalone-secret '
        "token=PLAIN_TOKEN"
    )

    assert sanitized == (
        '{"authorization":"<redacted>","api_key":"<redacted>",'
        '"client_secret":"<redacted>"} raw=<redacted> token=<redacted>'
    )


def test_safe_exception_message_is_detailed_redacted_and_non_empty() -> None:
    assert (
        safe_exception_message(
            RuntimeError("gateway failed api_key=SECRET useful detail")
        )
        == "gateway failed api_key=<redacted> useful detail"
    )
    assert safe_exception_message(RuntimeError()) == (
        "Provider request failed unexpectedly."
    )
    assert format_user_error_preview(ValueError("x" * 500), max_len=20) == "x" * 20


def test_extract_upstream_error_detail_compacts_json_and_redacts_secrets() -> None:
    response = Response(
        status_code=400,
        request=Request("POST", "https://provider.test/v1/messages"),
        json={
            "error": {
                "type": "BadRequest",
                "message": "bad field api_key=SECRET authorization: Bearer TOKEN",
            }
        },
    )
    error = HTTPStatusError(
        "Bad Request",
        request=response.request,
        response=response,
    )

    detail = extract_upstream_error_detail(error)

    assert isinstance(detail, UpstreamErrorDetail)
    assert detail.status_code == 400
    assert detail.body_text == (
        '{"error":{"type":"BadRequest","message":'
        '"bad field api_key=<redacted> authorization: <redacted>"}}'
    )
    assert detail.exception_text == "Bad Request"
    assert detail.cause_chain_text is None
    assert detail.category_hint == "BadRequest"
    assert not detail.body_truncated
    assert "SECRET" not in repr(detail)
    assert "TOKEN" not in repr(detail)


def test_attached_upstream_body_is_capped_after_redaction() -> None:
    assert ERROR_DETAIL_DISPLAY_CAP_BYTES == 16_384
    response = Response(
        status_code=500,
        request=Request("POST", "https://provider.test/v1/messages"),
        content=b"",
    )
    error = HTTPStatusError(
        "Server Error",
        request=response.request,
        response=response,
    )
    attach_upstream_error_body(
        error,
        "token=SECRET " + "x" * ERROR_DETAIL_DISPLAY_CAP_BYTES,
    )

    detail = extract_upstream_error_detail(error)

    assert detail.body_text is not None
    assert detail.body_truncated
    assert "SECRET" not in detail.body_text
    assert "token=<redacted>" in detail.body_text
    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in detail.body_text


def test_cause_chain_is_redacted_capped_and_has_safe_type_metadata() -> None:
    request = Request("POST", "https://provider.test/v1/messages")
    error = RuntimeError("provider connection failed")
    error.__cause__ = ConnectError(
        "connect failed authorization: Bearer SECRET token=ALSO_SECRET "
        + "x" * ERROR_DETAIL_DISPLAY_CAP_BYTES,
        request=request,
    )

    detail = extract_upstream_error_detail(error)

    assert exception_cause_types(error) == ("ConnectError",)
    assert detail.cause_chain_text is not None
    assert "ConnectError: connect failed authorization: <redacted>" in (
        detail.cause_chain_text
    )
    assert "SECRET" not in detail.cause_chain_text
    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in (
        detail.cause_chain_text
    )


def test_execution_failure_format_uses_semantic_category_and_request_id() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.INVALID_REQUEST,
        status_code=400,
        message="Invalid request sent to provider.",
        retryable=False,
    )
    detail = UpstreamErrorDetail(
        status_code=400,
        body_text='{"error":{"message":"bad field token=<redacted>"}}',
        exception_text="Bad Request",
        category_hint=None,
    )

    message = format_execution_failure_message(
        failure,
        detail,
        upstream_name="ACME",
        request_id="req_diagnostic",
    )

    assert "Upstream provider ACME returned HTTP 400." in message
    assert "Category: invalid_request" in message
    assert "Mapped message: Invalid request sent to provider." in message
    assert '{"error":{"message":"bad field token=<redacted>"}}' in message
    assert "Request ID: req_diagnostic" in message
    assert "invalid_request_error" not in message
