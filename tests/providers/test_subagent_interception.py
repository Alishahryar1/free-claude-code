import json

from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
)
from free_claude_code.providers.openai_chat.tool_calls import (
    iter_heuristic_tool_use_events,
)


def test_task_tool_interception() -> None:
    output = AnthropicChatStreamOutput(
        message_id="msg_test",
        model="test-model",
        input_tokens=1,
    )
    tool_use = {
        "type": "tool_use",
        "id": "tool_123",
        "name": "Task",
        "input": {
            "description": "test task",
            "prompt": "do something",
            "run_in_background": True,
        },
    }

    events = list(iter_heuristic_tool_use_events(output, tool_use))
    delta = next(event for event in events if '"type": "input_json_delta"' in event)
    payload = json.loads(delta.split("data: ", 1)[1])
    args_passed = json.loads(payload["delta"]["partial_json"])
    assert args_passed["run_in_background"] is False
