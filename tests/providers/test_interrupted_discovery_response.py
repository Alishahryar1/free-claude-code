"""A real Messages transport retains output and the original failure through FCC."""

import json

import httpx
import pytest
from starlette.responses import JSONResponse

from free_claude_code.api.response_streams import (
    openai_responses_sse_streaming_response,
)
from free_claude_code.providers.github_copilot.types import CopilotEgress
from tests.application.test_responses_web_lifetime import _consume
from tests.application.test_responses_web_workflow import WebClient
from tests.core.openai_responses.test_client_tool_discovery import SEARCH
from tests.providers.test_anthropic_messages_transport import Wire, _events, _sse
from tests.providers.test_copilot_execution_lifetime import executor_stream
from tests.providers.test_github_copilot_provider import Harness


@pytest.mark.asyncio
@pytest.mark.parametrize("with_web", [False, True])
@pytest.mark.parametrize("ending", ["timeout", "eof", "error"])
@pytest.mark.parametrize("arguments", ["", '{"query":'])
async def test_interrupted_messages_discovery_keeps_completed_text_and_original_failure(
    tmp_path, monkeypatch, with_web, ending, arguments
):
    harness = Harness(tmp_path, CopilotEgress.MESSAGES)
    text = "Completed before interruption. " * 2500
    chunks: list[bytes | Exception] = [
        _sse(
            *_events(text)[:4],
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "aborted",
                    "name": "fcc_tool_search",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            },
        )
    ]
    if ending == "timeout":
        chunks.append(httpx.ReadTimeout("Provider stalled during discovery"))
    elif ending == "error":
        chunks.append(
            _sse(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "Upstream failed"},
                }
            )
        )
    wire = Wire(chunks)

    def messages(incoming):
        harness.seen.append(incoming)
        assert any(
            tool["name"] == "fcc_tool_search"
            for tool in json.loads(incoming.content)["tools"]
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    monkeypatch.setattr(harness.provider._messages_pool, "handler", messages)
    try:
        response = await openai_responses_sse_streaming_response(
            executor_stream(
                harness, web_client=WebClient() if with_web else None, tools=[SEARCH]
            ),
            headers={},
            request_id="interrupted-discovery",
            pre_start_error_response=lambda exc: JSONResponse({"error": str(exc)}, 500),
        )
        assert response.status_code == 200
        events = await _consume(response)
        terminals = [
            event
            for event in events
            if event.event
            in {"response.failed", "response.completed", "response.incomplete"}
        ]
        assert len(terminals) == 1 and terminals[0].event == "response.failed"
        final = terminals[0].data["response"]
        assert final["error"]["type"] == "api_error"
        assert (
            final["error"]["message"].splitlines()[0]
            == {
                "timeout": "Provider request timed out after 3s.",
                "eof": "Messages stream ended without message_stop.",
                "error": "Upstream failed",
            }[ending]
        )
        assert final["output"][0]["content"][0]["text"] == text
        partial = final["output"][1]
        assert (
            partial["type"] == "tool_search_call" and partial["status"] == "incomplete"
        )
        # Messages publishes arguments only after a validated content-block stop.
        assert partial["call_id"] == "aborted" and partial["arguments"] == ""
        completed = [
            event.data["item"]
            for event in events
            if event.event == "response.output_item.done"
        ]
        assert completed == [final["output"][0]]
        assert len(harness.seen) == 1 and wire.closed
        assert harness.provider._active == 0
    finally:
        await harness.provider.cleanup()
        await harness.auth.close()
