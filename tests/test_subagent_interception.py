import json
import pytest
from unittest.mock import MagicMock
from providers.nvidia_nim import NvidiaNimProvider
from providers.common import ContentBlockManager
from providers.base import ProviderConfig
from config.nim import NimSettings


@pytest.mark.asyncio
async def test_task_tool_interception():
    # Setup provider
    config = ProviderConfig(api_key="test")
    provider = NvidiaNimProvider(config, nim_settings=NimSettings())

    # Mock request and sse builder with real ContentBlockManager
    request = MagicMock()
    request.model = "test-model"

    sse = MagicMock()
    sse.blocks = ContentBlockManager()

    # Tool call data (Task tool)
    tc = {
        "index": 0,
        "id": "tool_123",
        "function": {
            "name": "Task",
            "arguments": json.dumps(
                {
                    "description": "test task",
                    "prompt": "do something",
                    "run_in_background": True,
                }
            ),
        },
    }

    # Call the method
    events = []
    for event in provider._process_tool_call(tc, sse):
        events.append(event)

    # Find the emit_tool_delta call and check args
    calls = sse.emit_tool_delta.call_args_list
    assert len(calls) > 0
    args_passed = json.loads(calls[0][0][1])
    assert args_passed["run_in_background"] is False


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_task_tool_interception())
