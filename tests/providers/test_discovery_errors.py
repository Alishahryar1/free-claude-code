"""Shareable discovery errors retain transport causes without upstream secrets."""

import socket
import ssl

import httpx
import httpx2
import openai
import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.runtime.discovery import _provider_query_failure_reason


@pytest.mark.parametrize(
    "cause, expected",
    [
        (httpx2.ProxyError("400 Bad Request secret-canary"), "proxy"),
        (httpx.ProxyError("secret-canary"), "proxy"),
        (socket.gaierror(-2, "secret-canary"), "DNS"),
        (ssl.SSLCertVerificationError(1, "secret-canary"), "certificate"),
        (ssl.SSLError("secret-canary"), "TLS"),
        (httpx2.ReadTimeout("secret-canary"), "timed out"),
        (httpx.ConnectError("secret-canary"), "connect"),
        (ModelListResponseError("secret-canary"), "invalid model list"),
        (ApplicationUnavailableError("secret-canary"), "unavailable"),
        (
            ExecutionFailure(FailureKind.TIMEOUT, 504, "secret-canary", True),
            "timed out",
        ),
    ],
)
def test_wrapped_discovery_errors_are_specific_and_secret_safe(cause, expected):
    error = openai.APIConnectionError(
        message="secret-canary", request=httpx2.Request("GET", "https://example.test")
    )
    error.__cause__ = cause
    message = _provider_query_failure_reason(error)
    assert expected in message
    assert "secret-canary" not in message


def test_sdk_status_error_preserves_status_without_body():
    response = httpx2.Response(
        401, request=httpx2.Request("GET", "https://example.test")
    )
    error = openai.APIStatusError(
        "secret-canary", response=response, body={"secret": "secret-canary"}
    )
    assert (
        _provider_query_failure_reason(error)
        == "The model-list request returned HTTP 401."
    )


@pytest.mark.parametrize("client", [httpx, httpx2])
def test_http_status_is_preserved_without_url_body_or_exception_message(client):
    request = client.Request("GET", "https://user:secret-canary@example.test/models")
    response = client.Response(403, request=request, text="secret-canary")
    error = client.HTTPStatusError("secret-canary", request=request, response=response)
    message = _provider_query_failure_reason(error)
    assert "HTTP 403" in message
    assert "secret-canary" not in message


def test_exception_context_and_cycles_are_safe():
    error = RuntimeError("secret-canary")
    cause = socket.gaierror(-2, "secret-canary")
    error.__context__ = cause
    cause.__context__ = error
    assert "DNS" in _provider_query_failure_reason(error)
    error.__suppress_context__ = True
    message = _provider_query_failure_reason(error)
    assert "DNS" not in message
    assert "secret-canary" not in message
