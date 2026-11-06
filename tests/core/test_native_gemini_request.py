"""Native Google requests preserve Parts and do not parse text tags."""

from typing import Any, cast

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
