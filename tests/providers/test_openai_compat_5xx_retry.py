"""OpenAI-chat providers route upstream 5xx and 429 through the same retry path."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from httpx2 import Request, Response

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)


def _internal_5xx(code: int) -> openai.InternalServerError:
    return openai.InternalServerError(
        "unavailable",
        response=Response(code, request=Request("POST", "http://x")),
        body={},
    )


def _connection_error(message: str = "connect failed") -> openai.APIConnectionError:
    error = openai.APIConnectionError(
        request=Request("POST", "https://test.api.nvidia.com/v1/chat/completions")
    )
    error.__cause__ = httpx.ConnectError(message)
    return error


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@pytest.mark.asyncio
async def test_nim_stream_retries_on_openai_5xx_then_streams(status_code):
    config = make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=100,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    provider = NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_messages_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Hi", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create,
    ):
        mock_create.side_effect = [_internal_5xx(status_code), mock_stream()]
        events = [e async for e in provider.stream_messages(req)]

    assert mock_create.await_count == 2
    assert any("Hi" in e for e in events)


@pytest.mark.asyncio
async def test_nim_stream_retries_on_pre_stream_connection_error_then_streams():
    config = make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=100,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    provider = NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_messages_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create,
    ):
        mock_create.side_effect = [_connection_error(), mock_stream()]
        events = [e async for e in provider.stream_messages(req)]

    assert mock_create.await_count == 2
    assert any("Recovered" in e for e in events)


@pytest.mark.asyncio
async def test_nim_stream_connection_error_exhausted_emits_cause_chain():
    config = make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=100,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    provider = NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_messages_request()
    error = _connection_error("upstream disconnected")

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ) as mock_create,
        patch("free_claude_code.providers.openai_chat.transport.trace_event") as trace,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [e async for e in provider.stream_messages(req, request_id="req_conn")]

    assert mock_create.await_count == 5
    error_traces = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.response.error"
    ]
    assert error_traces[-1]["request_id"] == "req_conn"
    assert error_traces[-1]["exc_type"] == "APIConnectionError"
    assert "error_message" not in error_traces[-1]
    assert "Caused by:\nConnectError: upstream disconnected" in exc_info.value.message


@pytest.mark.parametrize(
    ("status_code", "expect_substr"),
    [
        (500, "provider api request failed"),
        (502, "temporarily unavailable"),
        (503, "temporarily unavailable"),
        (504, "temporarily unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_nim_stream_openai_5xx_exhausted_emits_user_message(
    status_code,
    expect_substr,
):
    config = make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=100,
        rate_window=60,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    provider = NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )
    req = make_messages_request()

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create,
    ):
        mock_create.side_effect = _internal_5xx(status_code)
        with pytest.raises(ExecutionFailure) as exc_info:
            [e async for e in provider.stream_messages(req)]

    assert mock_create.await_count == 5
    assert expect_substr in exc_info.value.message.lower()
