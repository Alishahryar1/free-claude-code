"""Unsupported request features skip a target before its first generation call."""

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.core.recovery import CandidateIncompatible, RecoveryCheckpoint
from tests.providers.support import attempt_events
from tests.providers.test_anthropic_messages_transport import Endpoint
from tests.providers.test_history_transports import _harness


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages"])
@pytest.mark.parametrize(
    "features",
    [
        {"tools": [{"type": "web_search"}]},
        {"future_native_control": {"handle": "opaque"}},
        {"previous_response_id": "resp_stored"},
    ],
)
async def test_unrepresentable_responses_features_skip_candidate_without_http(
    protocol, features
):
    async with _harness(protocol) as (_, bodies, provider):
        request = OpenAIResponsesRequest.model_validate(
            {"model": "model", "input": "hello", **features}
        )
        kwargs = {"endpoint_context": Endpoint()} if protocol == "messages" else {}
        async with provider.open_responses(request, **kwargs) as candidate:
            with pytest.raises(CandidateIncompatible):
                _ = [
                    event
                    async for event in attempt_events(
                        candidate, RecoveryCheckpoint("responses", recovering=True)
                    )
                ]
        assert bodies == []


@pytest.mark.asyncio
async def test_unrepresentable_messages_control_skips_responses_candidate():
    async with _harness("responses") as (_, bodies, provider):
        request = MessagesRequest(
            model="model",
            messages=[Message(role="user", content="hi")],
            stop_sequences=["END"],
        )
        async with provider.open_messages(
            request,
            input_tokens=0,
            request_id=None,
            response_model="model",
            reasoning=DEFAULT_REASONING_POLICY,
        ) as candidate:
            with pytest.raises(CandidateIncompatible):
                _ = [
                    event
                    async for event in attempt_events(
                        candidate, RecoveryCheckpoint("messages", recovering=True)
                    )
                ]
        assert bodies == []


@pytest.mark.asyncio
async def test_malformed_history_is_terminal_validation_not_candidate_incompatibility():
    async with _harness("responses") as (_, bodies, provider):
        request = OpenAIResponsesRequest(
            model="model",
            input=[{"type": "reasoning", "encrypted_content": "fcc:history:v1:broken"}],
        )
        async with provider.open_responses(
            request,
            input_tokens=0,
            request_id=None,
            response_model="model",
            reasoning=DEFAULT_REASONING_POLICY,
        ) as candidate:
            with pytest.raises(InvalidRequestError):
                _ = [
                    event
                    async for event in attempt_events(
                        candidate, RecoveryCheckpoint("responses", recovering=True)
                    )
                ]
        assert bodies == []


@pytest.mark.asyncio
async def test_stored_response_handle_is_not_dropped_to_admit_a_recovery_target():
    async with _harness("responses") as (_, bodies, provider):
        request = OpenAIResponsesRequest(
            model="model", input="continue", previous_response_id="resp_remote"
        )
        async with provider.open_responses(
            request,
            input_tokens=0,
            request_id=None,
            response_model="model",
            reasoning=DEFAULT_REASONING_POLICY,
        ) as candidate:
            with pytest.raises(CandidateIncompatible):
                _ = [
                    event
                    async for event in attempt_events(
                        candidate, RecoveryCheckpoint("responses", recovering=True)
                    )
                ]
        assert bodies == []
