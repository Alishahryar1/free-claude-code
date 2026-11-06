"""Native Google requests preserve Parts and do not parse text tags."""

from typing import Any, cast

import pytest

from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.gemini.native_request import chat_to_gemini


def test_chat_to_gemini_keeps_thoughts_typed_and_tool_ids():
    request = ChatRequest(
        model="models/model",
        messages=[
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "<thought>literal answer</thought>",
                "reasoning_content": "actual thought",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file"},
        ],
    )
    body = cast(dict[str, Any], chat_to_gemini(request))
    parts = body["contents"][1]["parts"]
    assert parts[:2] == [
        {"text": "actual thought", "thought": True},
        {"text": "<thought>literal answer</thought>"},
    ]
    assert parts[2]["functionCall"] == {"id": "call_1", "name": "read", "args": {}}
    assert body["contents"][2]["parts"][0]["functionResponse"]["id"] == "call_1"


def test_messages_auto_choice_without_tools_has_no_function_config():
    from free_claude_code.core.anthropic.models import MessagesRequest
    from free_claude_code.core.gemini.native_request import messages_to_gemini

    request = MessagesRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        tool_choice={"type": "auto"},
    )
    assert "toolConfig" not in messages_to_gemini(request)


def test_messages_object_tool_result_retains_complete_value():
    from free_claude_code.core.anthropic.models import MessagesRequest
    from free_claude_code.core.gemini.native_request import messages_to_gemini

    request = MessagesRequest(
        model="model",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call", "name": "read", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call",
                        "content": {"count": 3, "nested": {"ok": True}},
                    }
                ],
            },
        ],
    )
    result = cast(dict[str, Any], messages_to_gemini(request))
    assert result["contents"][-1]["parts"][0]["functionResponse"]["response"][
        "output"
    ] == {"count": 3, "nested": {"ok": True}}


def test_gemini_25_effort_uses_supported_numeric_budget():
    from free_claude_code.core.gemini.native_request import apply_reasoning
    from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy

    for model in ("gemini-2.5-flash", "models/gemini-2.5-pro", "gemini-2.5-flash-lite"):
        for effort in ReasoningEffort:
            body: dict[str, Any] = {"model": model}
            apply_reasoning(body, ReasoningPolicy.on(effort=effort))
            assert body["generationConfig"]["thinkingConfig"] == {
                "includeThoughts": True,
                "thinkingBudget": effort.budget_tokens,
            }


def test_messages_strict_tool_guarantee_is_rejected_for_gemini():
    from free_claude_code.core.anthropic.models import MessagesRequest
    from free_claude_code.core.gemini.native_request import messages_to_gemini

    request = MessagesRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "read", "input_schema": {"type": "object"}, "strict": True}],
    )
    with pytest.raises(ValueError, match="strict"):
        messages_to_gemini(request)


def test_messages_noop_context_management_keeps_claude_code_compatible():
    from free_claude_code.core.anthropic.models import MessagesRequest
    from free_claude_code.core.gemini.native_request import messages_to_gemini

    request = MessagesRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        context_management={
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    )
    result = messages_to_gemini(request)
    assert result["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert "context_management" not in result
