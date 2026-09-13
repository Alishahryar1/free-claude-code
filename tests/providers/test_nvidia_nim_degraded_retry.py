"""NVIDIA Cloud Function deployment failures use provider-owned retry semantics."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx2
import openai
import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.admission import (
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    overloaded_provider_failure,
    retryable_upstream_status,
)
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import make_provider_config

_FUNCTION_ID = "87ea0ddc-cff1-4bca-bf8b-3bd98a35ddd0"
_DEGRADED_DETAIL = f"Function id '{_FUNCTION_ID}': DEGRADED function cannot be invoked"


def _config(base_url: str) -> ProviderConfig:
    return make_provider_config(
        api_key="test_key",
        base_url=base_url,
        rate_limit=1_000_000,
        rate_window=1,
        max_concurrency=1_000,
        http_read_timeout=30.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )


def _admission(*, max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS):
    return ProviderAdmissionController(
        provider_name="nvidia_nim",
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
        max_attempts=max_attempts,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )


def _bad_request(
    detail: str = _DEGRADED_DETAIL,
    *,
    body_extra: dict[str, str] | None = None,
) -> openai.BadRequestError:
    request = httpx2.Request(
        "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    response = httpx2.Response(400, request=request)
    body: dict[str, object] = {
        "status": 400,
        "title": "Bad Request",
        "detail": detail,
    }
    if body_extra is not None:
        body.update(body_extra)
    return openai.BadRequestError("Bad Request", response=response, body=body)


def _context_window_error(
    *,
    message: str = (
        "max_tokens must be at least 1, got -853. (parameter=max_tokens, value=-853)"
    ),
    param: str = "max_tokens",
    nested: bool = False,
    status_code: int = 400,
) -> openai.BadRequestError | openai.InternalServerError:
    request = httpx2.Request(
        "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    response = httpx2.Response(status_code, request=request)
    error_body: dict[str, object] = {
        "message": message,
        "type": "BadRequestError",
        "param": param,
        "code": status_code,
    }
    body = {"error": error_body} if nested else error_body
    if status_code == 500:
        return openai.InternalServerError(
            "Internal Server Error", response=response, body=body
        )
    return openai.BadRequestError("Bad Request", response=response, body=body)


def _successful_stream(text: str = "Recovered"):
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(content=text, reasoning_content=""),
            finish_reason="stop",
        )
    ]
    chunk.usage = None

    async def stream():
        yield chunk

    return stream()


def _nim(admission: ProviderAdmissionController) -> NvidiaNimProvider:
    return NvidiaNimProvider(
        _config("https://integrate.api.nvidia.com/v1"),
        nim_settings=NimSettings(),
        admission=admission,
    )


@pytest.mark.asyncio
async def test_degraded_function_retries_unchanged_request_then_succeeds() -> None:
    admission = _admission()
    provider = _nim(admission)

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[_bad_request(), _successful_stream()],
        ) as create,
    ):
        events = [
            event
            async for event in provider.stream_messages(
                make_messages_request(), request_id="req_recovered"
            )
        ]

    assert create.await_count == 2
    assert create.call_args_list[0].kwargs == create.call_args_list[1].kwargs
    event_text = "".join(events)
    assert "Recovered" in event_text
    assert "event: message_stop" in event_text
    assert "event: error" not in event_text


@pytest.mark.asyncio
async def test_degraded_function_exhaustion_is_detailed_redacted_overload() -> None:
    admission = _admission()
    provider = _nim(admission)
    error = _bad_request(
        body_extra={
            "authorization": "Bearer NIM_AUTH_SECRET",
            "api_key": "NIM_API_SECRET",
        }
    )

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ) as create,
        patch("free_claude_code.providers.openai_chat.transport.trace_event") as trace,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_messages(
                make_messages_request(), request_id="req_degraded"
            )
        ]

    assert create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS

    failure = exc_info.value
    assert failure.kind is FailureKind.OVERLOADED
    assert failure.status_code == 529
    assert failure.retryable is True
    assert "Upstream provider NIM returned HTTP 400." in failure.message
    assert _DEGRADED_DETAIL in failure.message
    assert "Request ID: req_degraded" in failure.message
    assert "NIM_AUTH_SECRET" not in failure.message
    assert "NIM_API_SECRET" not in failure.message

    error_traces = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.response.error"
    ]
    assert error_traces[-1]["exc_type"] == "BadRequestError"
    assert error_traces[-1]["failure_kind"] == "overloaded"
    assert error_traces[-1]["status_code"] == 529
    assert error_traces[-1]["provider_retryable"] is True
    assert "error_message" not in error_traces[-1]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.asyncio
async def test_negative_derived_max_tokens_is_context_window_failure(
    nested: bool,
) -> None:
    provider = _nim(_admission())

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_context_window_error(nested=nested),
        ) as create,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_messages(
                make_messages_request(), request_id="req_context"
            )
        ]

    assert create.await_count == 1
    failure = exc_info.value
    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert failure.status_code == 400
    assert failure.retryable is False
    assert "Mapped message: Provider input exceeds the model context window." in (
        failure.message
    )
    assert "max_tokens must be at least 1, got -853" in failure.message
    assert "Request ID: req_context" in failure.message
    assert "prompt is too long" not in failure.message


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.asyncio
async def test_negative_derived_max_tokens_is_context_window_failure_on_500(
    nested: bool,
) -> None:
    provider = _nim(_admission())

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_context_window_error(nested=nested, status_code=500),
        ) as create,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_messages(
                make_messages_request(), request_id="req_context_500"
            )
        ]

    assert create.await_count == 1
    failure = exc_info.value
    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert failure.status_code == 400
    assert failure.retryable is False
    assert "Mapped message: Provider input exceeds the model context window." in (
        failure.message
    )
    assert "max_tokens must be at least 1, got -853" in failure.message
    assert "Request ID: req_context_500" in failure.message
    assert "prompt is too long" not in failure.message


@pytest.mark.parametrize(
    ("message", "param"),
    [
        ("max_tokens must be at least 1, got 0", "max_tokens"),
        ("max_tokens must be at least 1, got -853", "temperature"),
        ("max_tokens must be less than or equal to 8192", "max_tokens"),
    ],
)
@pytest.mark.asyncio
async def test_other_nim_max_token_errors_remain_invalid_requests(
    message: str,
    param: str,
) -> None:
    provider = _nim(_admission())

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_context_window_error(message=message, param=param),
        ) as create,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [event async for event in provider.stream_messages(make_messages_request())]

    assert create.await_count == 1
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "detail",
    [
        "Unsupported field: top_k",
        "Validation failed: DEGRADED function cannot be invoked",
        f"Function id '{_FUNCTION_ID}': DEGRADING function cannot be invoked",
        f"Function id '{_FUNCTION_ID}': DEGRADED function is waiting",
    ],
)
@pytest.mark.asyncio
async def test_unrelated_nim_bad_request_is_not_retried(detail: str) -> None:
    admission = _admission()
    provider = _nim(admission)

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_bad_request(detail),
        ) as create,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [event async for event in provider.stream_messages(make_messages_request())]

    assert create.await_count == 1
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_degraded_wording_remains_non_retryable_for_other_providers() -> None:
    admission = _admission()
    provider = OpenRouterProvider(
        _config("https://openrouter.ai/api/v1"), admission=admission
    )
    error = _bad_request()

    assert retryable_upstream_status(error) is None

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ) as create,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [event async for event in provider.stream_messages(make_messages_request())]

    assert create.await_count == 1
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_admission_override_preserves_raw_exception_after_exhaustion() -> None:
    admission = _admission(max_attempts=2)
    errors = (_bad_request(), _bad_request())
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        error = errors[attempts]
        attempts += 1
        raise error

    override = Mock(return_value=overloaded_provider_failure())
    with (
        pytest.raises(openai.BadRequestError) as exc_info,
    ):
        await admission.start_execution().run_call(
            fail,
            operation_kind=ProviderOperationKind.GENERATION,
            provider_failure_override=override,
        )

    assert attempts == 2
    assert override.call_count == 2
    assert exc_info.value is errors[-1]
