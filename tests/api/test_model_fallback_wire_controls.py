"""Fallback efforts reach real provider encoders and their HTTP wire bodies."""

import json
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from fastapi.responses import JSONResponse

from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.reasoning import FALLBACK_REASONING_PREFERENCES
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.custom import CustomProvider
from free_claude_code.providers.runtime.config import build_custom_provider_config
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    execution_failure,
)
from tests.api.test_classifier_provider_compatibility import classifier_request
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.support import immediate_admission
from tests.providers.test_custom_provider import completion, definition, provider
from tests.web_tools_support import StubWebToolsClient


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", FALLBACK_REASONING_PREFERENCES)
async def test_classifier_fallback_keeps_manual_thinking_compatible(effort):
    captured = []

    def reply(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=completion("anthropic_messages").replace(
                "Hello", "<severity>0</severity>"
            ),
            headers={"content-type": "text/event-stream"},
        )

    configuration = definition(
        api_format="anthropic_messages", reasoning_format="messages_manual"
    )
    fallback = provider(configuration, messages_handler=reply)
    primary = ControlledFallbackProvider(failure=execution_failure("primary failed"))
    settings = Settings(
        MODEL="nvidia_nim/primary",
        custom_providers=(configuration,),
        MODEL_FALLBACKS=[
            {"model": f"{configuration.provider_id}/backup", "reasoning": effort}
        ],
    )
    handler = MessagesHandler(
        settings,
        provider_resolver=AsyncMock(
            side_effect=lambda provider_id, _api_key_number: (
                primary if provider_id == "nvidia_nim" else fallback
            )
        ),
        web_tools=StubWebToolsClient(),
    )
    try:
        response = await handler.create(
            classifier_request().model_copy(update={"model": "nvidia_nim/primary"})
        )
        assert isinstance(response, JSONResponse) and response.status_code == 200
        assert json.loads(bytes(response.body))["content"] == [
            {"type": "text", "text": "<severity>0</severity>"}
        ]
        assert len(captured) == 1
        assert captured[0]["thinking"] == {"type": "disabled"}
        assert captured[0]["max_tokens"] > 64
    finally:
        await fallback.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize("egress", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize("effort", ["off", "low", "high", "max"])
async def test_fallback_effort_replaces_budget_and_preserves_wire_options(
    wire_api, egress, effort
):
    captured = []

    def reply(request):
        captured.append(json.loads(request.content))
        response = completion(egress)
        if egress == "openai_chat":
            delta = {
                "id": "reasoning",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "backup",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Candidate reasoning"},
                        "finish_reason": None,
                    }
                ],
            }
            response = f"data: {json.dumps(delta)}\n\n" + response
        return httpx2.Response(
            200, text=response, headers={"content-type": "text/event-stream"}
        )

    configuration = definition(
        api_format=egress,
        reasoning_format="openai_effort"
        if egress == "openai_chat"
        else "native_responses",
    )
    fallback = provider(configuration, openai_handler=reply)
    primary = ControlledFallbackProvider(failure=execution_failure("primary failed"))
    settings = Settings(
        MODEL="nvidia_nim/primary",
        custom_providers=(configuration,),
        MODEL_FALLBACKS=[
            {"model": f"{configuration.provider_id}/backup", "reasoning": effort}
        ],
    )
    executor = ProviderExecutor(
        AsyncMock(
            side_effect=lambda provider_id, _api_key_number: (
                primary if provider_id == "nvidia_nim" else fallback
            )
        ),
        progress_timeout_seconds=10,
    )
    router = ModelRouter(settings)
    if wire_api == "messages":
        request = MessagesRequest.model_validate(
            {
                "model": "client-alias",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8192,
                "thinking": {"type": "enabled", "budget_tokens": 6000},
                "output_config": {"effort": "high"},
            }
        )
        routed = router.resolve_messages_request(request)
    else:
        request = OpenAIResponsesRequest(
            model="client-alias",
            input="Hello",
            reasoning={"effort": "high", "summary": "detailed", "context": "all_turns"},
            metadata={"keep": True},
        )
        routed = router.resolve_responses_request(request)
    before = request.model_dump()
    try:
        output = "".join(
            [
                event
                async for event in getattr(executor, f"stream_{wire_api}")(
                    routed, raw_log_payload=dict, request_id="fallback-wire"
                )
            ]
        )
        assert "Hello" in output and request.model_dump() == before
        assert len(captured) == 1 and captured[0]["model"] == "backup"
        body = captured[0]
        expected = (
            "none"
            if effort == "off"
            else "xhigh"
            if effort == "max" and egress == "openai_chat"
            else effort
        )
        if egress == "openai_chat":
            assert body["reasoning_effort"] == expected
            assert ("Candidate reasoning" in output) is (effort != "off")
        else:
            assert body["reasoning"]["effort"] == expected
            if wire_api == "responses":
                assert body["reasoning"]["summary"] == "detailed"
                assert body["reasoning"]["context"] == "all_turns"
                assert body["metadata"] == {"keep": True}
    finally:
        await fallback.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
async def test_candidate_efforts_survive_retry_and_continuation(monkeypatch, wire_api):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    bodies = []
    attempts = {"primary": 0, "low": 0, "inherited": 0}

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        model = body["model"]
        attempts[model] += 1
        text = (
            ["Hello ", "there. "][attempts[model] - 1]
            if model == "primary"
            else ["Low ", "work. "][attempts[model] - 1]
            if model == "low"
            else "Hello there. Low work. Finished."
        )
        events = text_events("responses", text, complete=model == "inherited")
        raw = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return httpx2.Response(
            200, text=raw, headers={"content-type": "text/event-stream"}
        )

    configuration = definition(
        api_format="openai_responses", reasoning_format="native_responses"
    )
    settings = Settings(
        MODEL=f"{configuration.provider_id}/primary",
        custom_providers=(configuration,),
        REASONING_POLICY="high",
        MODEL_FALLBACKS=[
            {"model": f"{configuration.provider_id}/low", "reasoning": "low"},
            {"model": f"{configuration.provider_id}/inherited", "reasoning": "inherit"},
        ],
    )
    instance = CustomProvider(
        build_custom_provider_config(configuration, settings),
        definition=configuration,
        admission=immediate_admission(max_attempts=2),
        openai_transport=httpx2.MockTransport(reply),
    )
    executor = ProviderExecutor(
        AsyncMock(return_value=instance), progress_timeout_seconds=10
    )
    router = ModelRouter(settings)
    if wire_api == "messages":
        routed = router.resolve_messages_request(
            MessagesRequest.model_validate(
                {
                    "model": "client-alias",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "output_config": {"effort": "medium"},
                }
            )
        )
    else:
        routed = router.resolve_responses_request(
            OpenAIResponsesRequest(
                model="client-alias", input="Hello", reasoning={"effort": "medium"}
            )
        )
    try:
        stream = getattr(executor, f"stream_{wire_api}")(
            routed, raw_log_payload=dict, request_id="fallback-continuation"
        )
        events = parse_sse_text(await delivered(stream, wire_api))
        assert_completed(events, wire_api)
        assert public_text(events, wire_api) == "Hello there. Low work. Finished."
        assert [(body["model"], body["reasoning"]["effort"]) for body in bodies] == [
            ("primary", "high"),
            ("primary", "high"),
            ("low", "low"),
            ("low", "low"),
            ("inherited", "high"),
        ]
        assert "Hello there. " in str(bodies[2]["input"])
        assert "Low work. " in str(bodies[4]["input"])
    finally:
        await instance.cleanup()
