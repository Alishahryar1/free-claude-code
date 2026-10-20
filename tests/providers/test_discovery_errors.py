"""Shareable discovery errors retain transport causes without upstream secrets."""

import socket
import ssl
from unittest.mock import AsyncMock

import anyio
import anyio.streams.tls
import httpx
import httpx2
import openai
import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.failure_policy import ProviderRecoveryExhausted
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, expected", [("dns", "DNS"), ("certificate", "certificate")]
)
async def test_sdk_connection_pool_preserves_os_failure_diagnostics(
    monkeypatch, failure, expected
):
    stream = AsyncMock(spec=anyio.abc.ByteStream)
    connect = AsyncMock(return_value=stream)
    if failure == "dns":
        connect.side_effect = socket.gaierror(-2, "secret-canary")
    monkeypatch.setattr(anyio, "connect_tcp", connect)
    monkeypatch.setattr(
        anyio.streams.tls.TLSStream,
        "wrap",
        AsyncMock(side_effect=ssl.SSLCertVerificationError(1, "secret-canary")),
    )
    async with openai.AsyncOpenAI(
        api_key="secret-canary",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(trust_env=False),
    ) as client:
        with pytest.raises(openai.APIConnectionError) as caught:
            await client.models.list()
    message = _provider_query_failure_reason(caught.value)
    assert expected in message
    assert "secret-canary" not in message


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
    assert "DNS" in message
    assert "secret-canary" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [httpx2.ReadTimeout("secret-canary"), httpx2.ConnectError("secret-canary")]
)
async def test_real_admission_cooldown_retains_discovery_failure(error):
    controller = ProviderAdmissionController(
        provider_name="TEST",
        max_attempts=1,
        base_delay=30,
        max_delay=30,
        jitter=0,
    )
    fetch = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await controller.start_execution().run_call(
            fetch, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )
    with pytest.raises(ProviderRecoveryExhausted) as cooldown:
        await controller.start_execution().run_call(
            fetch, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )
    fetch.assert_awaited_once()
    assert cooldown.value.__cause__ is None
    assert cooldown.value.__context__ is None
    assert _provider_query_failure_reason(
        cooldown.value
    ) == _provider_query_failure_reason(error)


def test_recovery_wrapper_cycles_do_not_hang_classification():
    wrapper = ProviderRecoveryExhausted(RuntimeError("secret-canary"))
    wrapper.last_error = wrapper
    assert (
        _provider_query_failure_reason(wrapper)
        == "Could not load models. Check the provider's settings and retry."
    )
