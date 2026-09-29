"""Check logging levels in a subprocess without changing pytest's global sinks."""

import json
import subprocess
import sys
from unittest.mock import AsyncMock, Mock, patch

import httpx2
import pytest
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.api.handlers import token_count
from free_claude_code.application import execution
from free_claude_code.config.settings import Settings
from free_claude_code.core import trace
from free_claude_code.core.anthropic import TokenCountRequest
from free_claude_code.providers.openai_chat import transport as chat_transport
from tests.application.test_execution import (
    FakeProvider,
    ResponsesFakeProvider,
    _routed_request,
    _routed_responses_request,
)
from tests.providers.test_openai_chat_transport import _consume, _success, _transport


async def _probe(level: str, raw: bool) -> None:
    logger.remove()
    rows = []
    if level != "NONE":
        logger.add(
            lambda message: rows.append(json.loads(message)),
            level=level,
            serialize=True,
        )
    expected = int(level == "DEBUG")
    fields = Mock(return_value={"snapshot": {"text": "visible", "api_key": "hidden"}})
    raw_payload = Mock(return_value={"text": "raw conversation"})
    with (
        patch.object(
            trace, "sanitize_trace_value", wraps=trace.sanitize_trace_value
        ) as sanitize,
        patch.object(
            execution,
            "anthropic_request_snapshot",
            wraps=execution.anthropic_request_snapshot,
        ) as ingress,
        patch.object(
            token_count,
            "anthropic_request_snapshot",
            wraps=token_count.anthropic_request_snapshot,
        ) as counted,
        patch.object(
            chat_transport,
            "provider_chat_body_snapshot",
            wraps=chat_transport.provider_chat_body_snapshot,
        ) as provider_snapshot,
        logger.contextualize(request_id="req_lazy", instance_id="instance_lazy"),
    ):
        trace.trace_event(fields, stage="unit", event="lazy.probe", source="test")
        for wire, routed, provider in (
            ("messages", _routed_request(), FakeProvider()),
            ("responses", _routed_responses_request(), ResponsesFakeProvider()),
        ):
            executor = execution.ProviderExecutor(
                AsyncMock(return_value=provider),
                progress_timeout_seconds=60,
                token_counter=lambda *args: 1,
                responses_token_counter=lambda request: 1,
                log_raw_payloads=raw,
            )
            output = [
                chunk
                async for chunk in getattr(executor, f"stream_{wire}")(
                    routed, raw_log_payload=raw_payload, request_id="req_lazy"
                )
            ]
            assert output
        routed = _routed_request()
        handler = token_count.TokenCountHandler(
            Settings(),
            model_router=Mock(resolve_token_count_request=Mock(return_value=routed)),
            token_counter=lambda *args: 7,
        )
        assert (
            handler.count(
                TokenCountRequest(
                    model=routed.request.model, messages=routed.request.messages
                )
            ).input_tokens
            == 7
        )
        async with AsyncOpenAI(
            api_key="test",
            base_url="https://provider.invalid/v1",
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(lambda request: _success())
            ),
        ) as client:
            assert "hello" in await _consume(_transport(client), "messages")
        assert fields.call_count == expected
        assert (
            ingress.call_count
            == counted.call_count
            == provider_snapshot.call_count
            == expected
        )
        assert raw_payload.call_count == 2 * expected * raw
        assert bool(sanitize.call_count) == bool(expected)
    if expected:
        payloads = [
            row["record"]["extra"]["trace_payload"]
            for row in rows
            if "trace_payload" in row["record"]["extra"]
        ]
        probe = next(
            payload for payload in payloads if payload["event"] == "lazy.probe"
        )
        assert probe == {
            "stage": "unit",
            "event": "lazy.probe",
            "source": "test",
            "snapshot": {"text": "visible", "api_key": "<redacted>"},
        }
        ingress_payload = next(
            payload
            for payload in payloads
            if payload["event"] == "free_claude_code.api.request.received"
        )
        assert ingress_payload["snapshot"]["model"] == "gateway-model"
        assert (
            ingress_payload["snapshot"]["messages"]
            == routed.request.model_dump()["messages"]
        )
        assert rows[0]["record"]["extra"]["request_id"] == "req_lazy"
        assert rows[0]["record"]["extra"]["instance_id"] == "instance_lazy"
    else:
        assert all(row["record"]["level"]["no"] >= 20 for row in rows)
        assert all("trace_payload" not in row["record"]["extra"] for row in rows)
    assert (
        sum(row["record"]["message"].startswith("FULL_") for row in rows)
        == 2 * expected * raw
    )

    # Adding/removing DEBUG sinks must take effect immediately, with one build per event.
    logger.remove()
    logger.add(lambda message: None, level="INFO")
    fields.reset_mock()
    trace.trace_event(fields, stage="unit", event="levels", source="test")
    assert fields.call_count == 0
    sinks = [logger.add(lambda message: None, level="DEBUG") for _ in range(2)]
    trace.trace_event(fields, stage="unit", event="levels", source="test")
    assert fields.call_count == 1
    for sink in sinks:
        logger.remove(sink)
    trace.trace_event(fields, stage="unit", event="levels", source="test")
    assert fields.call_count == 1


@pytest.mark.parametrize("level", ["INFO", "DEBUG", "NONE"])
@pytest.mark.parametrize("raw", [False, True])
def test_debug_payloads_are_built_only_when_enabled(level, raw):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys; from tests.core.test_lazy_debug_logging import _probe; asyncio.run(_probe(sys.argv[1], sys.argv[2] == 'True'))",
            level,
            str(raw),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
