"""Tests that API and SSE logging avoid raw sensitive payloads by default."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from free_claude_code.api import request_errors
from free_claude_code.api.handlers import MessagesHandler, TokenCountHandler
from free_claude_code.application import execution
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import AnthropicStreamLedger
from free_claude_code.core.anthropic.models import Message, MessagesRequest


@pytest.mark.asyncio
async def test_create_message_skips_full_payload_debug_log_by_default():
    settings = Settings()
    assert settings.log_raw_api_payloads is False
    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: ping\ndata: {}\n\n"

    mock_provider.stream_messages = fake_stream
    service = MessagesHandler(
        settings, provider_resolver=AsyncMock(side_effect=lambda _: mock_provider)
    )

    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="secret-user-text")],
    )

    with patch.object(execution.logger, "debug") as mock_debug:
        await service.create(request)

    full_payload_calls = [
        c
        for c in mock_debug.call_args_list
        if c.args and str(c.args[0]) == "FULL_PAYLOAD [{}]: {}"
    ]
    assert not full_payload_calls


@pytest.mark.asyncio
async def test_create_message_logs_full_payload_when_opt_in():
    settings = Settings()
    settings.log_raw_api_payloads = True
    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: ping\ndata: {}\n\n"

    mock_provider.stream_messages = fake_stream
    service = MessagesHandler(
        settings, provider_resolver=AsyncMock(side_effect=lambda _: mock_provider)
    )
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="visible")],
    )

    with patch.object(execution.logger, "debug") as mock_debug:
        await service.create(request)

    keys = [c.args[0] for c in mock_debug.call_args_list if c.args]
    assert any(k == "FULL_PAYLOAD [{}]: {}" for k in keys)


def test_stream_ledger_default_debug_has_no_serialized_json_content():
    with patch(
        "free_claude_code.core.anthropic.streaming.emitter.logger.debug"
    ) as mock_debug:
        ledger = AnthropicStreamLedger("msg_x", "m", 1, log_raw_events=False)
        ledger.message_start()

    assert mock_debug.call_count == 0


def test_stream_ledger_raw_logging_includes_event_body_when_enabled():
    with patch(
        "free_claude_code.core.anthropic.streaming.emitter.logger.debug"
    ) as mock_debug:
        ledger = AnthropicStreamLedger("msg_x", "m", 1, log_raw_events=True)
        ledger.message_start()

    assert mock_debug.call_count == 1
    message = str(mock_debug.call_args)
    assert "message_start" in message
    assert "role" in message


def _flatten_log_calls(mock_log) -> str:
    parts: list[str] = []
    for call in mock_log.call_args_list:
        parts.extend(str(arg) for arg in call.args)
        parts.append(repr(call.kwargs))
    return " ".join(parts)


@pytest.mark.asyncio
async def test_create_message_unexpected_error_default_logs_exclude_exception_text():
    settings = Settings()
    assert settings.log_api_error_tracebacks is False
    secret = "upstream-secret-token-abc"

    mock_provider = MagicMock()

    def stream_boom(*_a, **_kw):
        raise RuntimeError(secret)

    mock_provider.stream_messages = stream_boom
    service = MessagesHandler(
        settings, provider_resolver=AsyncMock(side_effect=lambda _: mock_provider)
    )
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="hi")],
    )

    with patch.object(request_errors.logger, "error") as log_err:
        response = await service.create(request)

    blob = _flatten_log_calls(log_err)
    assert secret not in blob
    assert "RuntimeError" in blob
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    assert response.body


@pytest.mark.asyncio
async def test_create_message_unexpected_error_terminal_json_ignores_status_code():
    """Non-provider stream failures must not leak arbitrary HTTP status attributes."""

    class WeirdError(Exception):
        status_code = 418

    settings = Settings()
    mock_provider = MagicMock()

    def stream_boom(*_a, **_kw):
        raise WeirdError("no")

    mock_provider.stream_messages = stream_boom
    service = MessagesHandler(
        settings, provider_resolver=AsyncMock(side_effect=lambda _: mock_provider)
    )
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="hi")],
    )

    response = await service.create(request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    payload = bytes(response.body).decode("utf-8")
    assert '"type":"api_error"' in payload
    assert '"message":"no"' in payload


def test_parse_cli_event_error_logs_metadata_by_default():
    """CLI parser must not log raw error text unless LOG_RAW_CLI_DIAGNOSTICS is on."""
    from free_claude_code.messaging.event_parser import parse_cli_event

    secret = "user-secret-parser-leak-xyz"
    with patch("free_claude_code.messaging.event_parser.logger.info") as log_info:
        parse_cli_event(
            {"type": "error", "error": {"message": secret}}, log_raw_cli=False
        )
    flat = " ".join(str(c) for c in log_info.call_args_list)
    assert secret not in flat
    assert "message_chars" in flat


def test_parse_cli_event_error_logs_text_when_log_raw_cli_enabled():
    from free_claude_code.messaging.event_parser import parse_cli_event

    secret = "visible-cli-parser-msg"
    with patch("free_claude_code.messaging.event_parser.logger.info") as log_info:
        parse_cli_event(
            {"type": "error", "error": {"message": secret}}, log_raw_cli=True
        )
    flat = " ".join(str(c) for c in log_info.call_args_list)
    assert secret in flat


def test_count_tokens_unexpected_error_default_logs_exclude_exception_text():
    settings = Settings()
    assert settings.log_api_error_tracebacks is False
    secret = "count-tokens-leak-xyz"

    def boom(*_a, **_kw):
        raise ValueError(secret)

    service = TokenCountHandler(
        settings,
        token_counter=boom,
    )
    from free_claude_code.core.anthropic.models import TokenCountRequest

    req = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="x")],
    )

    with (
        patch.object(request_errors.logger, "error") as log_err,
        pytest.raises(HTTPException),
    ):
        service.count(req)

    blob = _flatten_log_calls(log_err)
    assert secret not in blob
    assert "ValueError" in blob
