from copy import deepcopy
from typing import Any

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat.request_policy import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

_POLICY = OpenAIChatRequestPolicy(
    provider_name="TEST",
    reasoning_replay=ReasoningReplayMode.THINK_TAGS,
)


def _request(name: str, *, continued: bool = False) -> MessagesRequest:
    messages: list[dict] = [
        {"role": "user", "content": "Use the tool"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": name,
                    "input": {"value": "x"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "done",
                }
            ],
        },
    ]
    if continued:
        messages.append({"role": "user", "content": "Continue"})
    return MessagesRequest.model_validate(
        {
            "model": "test-model",
            "messages": messages,
            "tools": [
                {
                    "name": name,
                    "description": "Long tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
                {
                    "name": "safe_tool",
                    "description": "Safe tool",
                    "input_schema": {"type": "object"},
                },
            ],
            "tool_choice": {"type": "tool", "name": name},
        }
    )


def _body(request: MessagesRequest, *, postprocessors=()) -> dict[str, Any]:
    return build_openai_chat_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
        policy=_POLICY,
        postprocessors=postprocessors,
    )


def test_chat_request_uses_one_alias_in_definition_choice_and_history() -> None:
    original = "mcp__long_server__" + "x" * 90
    request = _request(original)
    snapshot = request.model_dump()

    body = _body(request)

    tools = body["tools"]
    assert isinstance(tools, list)
    alias = tools[0]["function"]["name"]
    assert alias != original
    assert len(alias) <= 64
    assert tools[1]["function"]["name"] == "safe_tool"
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": alias},
    }
    messages = body["messages"]
    assert isinstance(messages, list)
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"] == {
        "name": alias,
        "arguments": '{"value": "x"}',
    }
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert request.model_dump() == snapshot


def test_chat_name_encoding_runs_after_provider_postprocessors() -> None:
    original = "mcp__postprocessor_order__" + "x" * 80
    request = _request(original)
    observed: list[str] = []

    def observe_names(body, _request, _reasoning) -> None:
        observed.append(body["tools"][0]["function"]["name"])

    body = _body(request, postprocessors=(observe_names,))

    assert observed == [original]
    assert body["tools"][0]["function"]["name"] != original


def test_chat_aliases_remain_stable_for_append_only_requests() -> None:
    original = "mcp__cache_stable__" + "x" * 90

    prefix_body = _body(_request(original))
    continued_body = _body(_request(original, continued=True))

    assert continued_body["tools"] == prefix_body["tools"]
    assert continued_body["tool_choice"] == prefix_body["tool_choice"]
    prefix_messages = prefix_body["messages"]
    assert continued_body["messages"][: len(prefix_messages)] == prefix_messages


def test_chat_encoding_does_not_rewrite_unrelated_name_fields() -> None:
    original = "mcp__canonical_name_only__" + "x" * 80
    request = _request(original)
    request.extra_body = {"metadata": {"name": original}}
    policy = OpenAIChatRequestPolicy(
        provider_name="TEST",
        reasoning_replay=ReasoningReplayMode.THINK_TAGS,
        include_extra_body=True,
    )
    snapshot = deepcopy(request.extra_body)

    body = build_openai_chat_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
        policy=policy,
    )

    assert body["extra_body"] == snapshot
    assert body["tools"][0]["function"]["name"] != original
