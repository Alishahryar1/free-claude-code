"""Retained local tool data must not corrupt ordinary tool groups or reasoning."""

import json
from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.native import NativeMessagesOptions
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    prepare_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_chat import ChatToolResultContext
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    build_responses_chat_request,
    build_responses_messages_request,
)
from free_claude_code.core.openai_responses.native import build_native_responses_request
from free_claude_code.core.openai_responses.web_history import (
    TOOL_CONTEXT_TYPE,
    decode_records,
    prepare_web_history,
    replay_item,
)
from free_claude_code.core.reasoning import ReasoningPolicy


def record():
    return {
        "id": "ws_source",
        "action": {"type": "search", "query": "docs"},
        "result": {
            "results": [
                {
                    "url": "https://example.org/",
                    "title": "Docs",
                    "snippet": "Retained search evidence",
                }
            ]
        },
    }


def fixture(protocol):
    origin = ReplayOrigin("test", protocol, "", "", "model")
    native: JsonObject = (
        {"type": "thinking", "thinking": "I need tools.", "signature": "signed-thought"}
        if protocol == "messages"
        else {
            "reasoning_content": "I need tools.",
            "reasoning_details": [
                {"type": "reasoning.encrypted", "data": "opaque-detail"}
            ],
        }
    )
    thinking = {
        "type": "reasoning",
        "id": "rs_thought",
        "summary": [],
        "encrypted_content": encode_replay(ReplayRecord(origin, native)),
    }
    calls = [
        {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": f"client_{n}",
            "name": f"tool_{n}",
            "arguments": "{}",
        }
        for n in (1, 2)
    ]
    request = OpenAIResponsesRequest(
        model="model",
        tools=[
            {"type": "function", "name": f"tool_{n}", "parameters": {"type": "object"}}
            for n in (1, 2)
        ],
        input=[
            {"role": "user", "content": "Research"},
            thinking,
            {
                "type": "web_search_call",
                "id": "ws_source",
                "status": "completed",
                "action": {"type": "search", "query": "docs"},
            },
            *calls,
            replay_item([record()]),
            {"type": "function_call_output", "call_id": "client_2", "output": "second"},
            {"type": "function_call_output", "call_id": "client_1", "output": "first"},
        ],
    )
    return request, origin, native


@pytest.mark.parametrize("protocol", ["chat", "messages"])
def test_replay_after_real_results_preserves_native_reasoning(protocol):
    original, origin, native = fixture(protocol)
    before = original.model_dump()
    projected = prepare_web_history(original).request
    assert projected.model_dump()["input"][-1]["type"] == TOOL_CONTEXT_TYPE
    assert "Retained search evidence" in projected.model_dump()["input"][-1]["content"]
    if protocol == "chat":
        prepared = build_responses_chat_request(
            projected,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            structured_reasoning_details=True,
        )
        wire = object_value(prepare_history(cast(JsonObject, prepared.body), origin))
        calls = next(
            message for message in wire["messages"] if message.get("tool_calls")
        )
        assert calls["reasoning_content"] == native["reasoning_content"]
        assert calls["reasoning_details"] == native["reasoning_details"]
        assert isinstance(wire["messages"][-1], ChatToolResultContext)
    else:
        prepared = build_responses_messages_request(
            projected, options=NativeMessagesOptions(model="model", max_tokens=1000)
        )
        wire = object_value(prepare_history(prepared.body, origin))
        assistant = next(
            message for message in wire["messages"] if message["role"] == "assistant"
        )
        assert assistant["content"][0] == native
        assert [part["type"] for part in wire["messages"][-1]["content"]] == [
            "tool_result",
            "tool_result",
            "text",
        ]
    assert TOOL_CONTEXT_TYPE not in json.dumps(wire)
    assert "fcc:web:" not in json.dumps(wire)
    assert original.model_dump() == before


def test_switch_to_native_responses_consumes_owned_context_only():
    original, _, _ = fixture("chat")
    projected = prepare_web_history(original).request
    body = object_value(
        build_native_responses_request(
            projected, model="model", reasoning=ReasoningPolicy.provider_default()
        )
    )
    assert body["input"][-1]["role"] == "user"
    assert "Retained search evidence" in body["input"][-1]["content"]
    assert "fcc:web:" not in json.dumps(body)
    assert TOOL_CONTEXT_TYPE not in json.dumps(body)


def test_missing_client_result_is_not_fabricated():
    original, _, _ = fixture("chat")
    original.input.pop()
    with pytest.raises(ResponsesConversionError, match="unresolved"):
        prepare_web_history(original)


def test_duplicate_carrier_does_not_duplicate_results():
    original, _, _ = fixture("chat")
    original.input.append(replay_item([record()]))
    history = prepare_web_history(original)
    assert len(history.records) == 1
    assert (
        sum(
            item.get("type") == TOOL_CONTEXT_TYPE
            for item in history.request.model_dump()["input"]
        )
        == 1
    )


def test_conflicting_carrier_is_rejected():
    original, _, _ = fixture("chat")
    conflict = record()
    conflict["result"] = {"error": "different"}
    original.input.append(replay_item([conflict]))
    with pytest.raises(ResponsesConversionError, match="Conflicting"):
        prepare_web_history(original)


@pytest.mark.parametrize(
    "value",
    ["fcc:web:v2:abc", "fcc:web:v1:!bad", "fcc:web:v1:" + "A" * 2097152],
    ids=["version", "malformed", "oversized"],
)
def test_bad_carrier_is_rejected_before_upstream(value):
    with pytest.raises(ResponsesConversionError):
        decode_records(value)


def test_carrier_after_client_compaction_does_not_need_a_group_anchor():
    original = OpenAIResponsesRequest(
        model="model",
        input=[
            {"role": "user", "content": "Compacted prior work"},
            replay_item([record()]),
        ],
    )
    projected = prepare_web_history(original).request
    assert projected.model_dump()["input"][-1]["type"] == TOOL_CONTEXT_TYPE


def test_unrelated_native_server_tool_search_is_not_a_client_result():
    original = OpenAIResponsesRequest(
        model="model",
        input=[
            {"role": "user", "content": "Research"},
            {
                "type": "tool_search_call",
                "execution": "server",
                "call_id": "native",
                "arguments": {},
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": "native",
                "tools": [],
            },
            replay_item([record()]),
        ],
    )
    before = deepcopy(original.model_dump()["input"][:3])
    assert prepare_web_history(original).request.model_dump()["input"][:3] == before


def object_value(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)
