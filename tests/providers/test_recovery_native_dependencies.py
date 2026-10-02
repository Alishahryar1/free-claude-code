"""Recovery cannot discard original native state or replay uncertain hosted work."""

import pytest

from free_claude_code.core.anthropic.recovery_stream import MessagesRecoveryWriter
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesRecoveryWriter,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.core.recovery import RecoveryCheckpoint
from free_claude_code.providers.history_replay import requires_native_origin
from free_claude_code.providers.stream_candidate import request_may_run_server_tools
from tests.providers.test_history_transports import _harness, _native


@pytest.mark.parametrize(
    "body,protocol",
    [
        ({"input": "hi", "future_control": {"handle": "opaque"}}, "responses"),
        ({"messages": [], "future_control": {"handle": "opaque"}}, "messages"),
        ({"input": [{"type": "future_state", "handle": "opaque"}]}, "responses"),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "hi",
                                "future_reference": "opaque",
                            }
                        ],
                    }
                ]
            },
            "responses",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hi", "future_reference": "opaque"}
                        ],
                    }
                ]
            },
            "messages",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "future_state", "handle": "opaque"}],
                    }
                ]
            },
            "messages",
        ),
    ],
)
def test_unknown_native_request_dependencies_keep_their_origin(body, protocol):
    assert requires_native_origin(body, protocol)


def test_user_tool_argument_fields_are_not_provider_references():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call",
                        "name": "read",
                        "input": {"file_id": "user-data", "container_id": "user-data"},
                    }
                ],
            }
        ]
    }
    assert not requires_native_origin(body, "messages")


@pytest.mark.parametrize(
    "body,protocol",
    [
        ({"mcp_servers": [{"url": "https://tools.invalid"}]}, "messages"),
        ({"container": {"id": "existing"}}, "messages"),
        (
            {"tools": [{"type": "namespace", "tools": [{"type": "web_search"}]}]},
            "responses",
        ),
    ],
)
def test_uncertain_server_operations_include_nested_and_native_declarations(
    body, protocol
):
    assert request_may_run_server_tools(body, protocol)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_dispatched_native_history_restricts_recovery_before_response_output(
    wire,
):
    async with _harness("responses") as (_, bodies, provider):
        request = OpenAIResponsesRequest(
            model="model",
            input=[_native("responses"), {"role": "user", "content": "hi"}],
        )
        writer = (
            ResponsesRecoveryWriter(model="model", input_tokens=0)
            if wire == "responses"
            else MessagesRecoveryWriter(model="model", input_tokens=0)
        )
        async with provider.open_responses(
            request,
            input_tokens=0,
            request_id=None,
            response_model="model",
            reasoning=DEFAULT_REASONING_POLICY,
        ) as candidate:
            checkpoint = RecoveryCheckpoint("responses")
            await candidate.prepare(checkpoint)
            source = candidate.stream_attempt(
                checkpoint, wait_for_recovery=True, can_correct=lambda: True
            )
            writer.begin_attempt()
            try:
                event = await anext(source)
                writer.feed(event)
                assert writer.checkpoint.required_origins == (event.origin,)
                assert bodies == []
            finally:
                await source.aclose()
