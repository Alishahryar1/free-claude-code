import json
from typing import Any, cast

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from free_claude_code.providers.runtime.factory import create_provider
from tests.core.openai_responses.test_client_tool_discovery import AGENTS, SEARCH
from tests.providers.support import immediate_admission
from tests.providers.test_opencode import (
    _catalog_payload,
    _provider_with_wire_transports,
    _responses_event_stream,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_id", ["open_router", "nvidia_nim", "groq", "mistral", "opencode_zen"]
)
@pytest.mark.parametrize("custom", [False, True])
async def test_provider_discovery_call_and_result_round_trip(
    provider_id: str, custom: bool
) -> None:
    native = provider_id == "opencode_zen"
    requests: list[dict[str, Any]] = []

    def upstream(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        functions = (
            {tool["name"]: tool for tool in body.get("tools", [])}
            if native
            else {
                tool["function"]["name"]: tool["function"]
                for tool in body.get("tools", [])
            }
        )
        messages = body["input"] if native else body["messages"]
        result_field = "output" if native else "content"
        step = len(requests)
        if step == 1:
            assert len(functions) == 1
            name = next(iter(functions))
            assert (
                functions[name]["parameters"]["properties"]["query"]["type"] == "string"
            )
            args = '{"query":"agent"}'
        elif step == 2:
            name = "editor__edit" if custom else "agents__spawn_agent"
            assert name in functions
            assert (
                messages[-1].get("type") == "function_call_output"
                if native
                else messages[-1]["role"] == "tool"
            )
            assert name.split("__")[-1] in messages[-1][result_field]
            args = '{"input":"patch"}' if custom else '{"message":"hello"}'
        else:
            assert (
                messages[-1].get("type") == "function_call_output"
                if native
                else messages[-1]["role"] == "tool"
            )
            assert messages[-1][result_field] == "completed"
            name, args = "", ""
        packets: list[dict[str, Any]]
        if native:
            packets = [
                json.loads(line[6:])
                for line in _responses_event_stream("done").splitlines()
                if line.startswith("data: ")
            ]
            if name:
                call = {
                    "type": "function_call",
                    "id": f"fc_{step}",
                    "call_id": f"call_{step}",
                    "name": name,
                    "arguments": args,
                    "status": "completed",
                }
                packets = [
                    packets[0],
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**call, "status": "in_progress", "arguments": ""},
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": call,
                    },
                    {
                        **packets[-1],
                        "response": {**packets[-1]["response"], "output": [call]},
                    },
                ]
            if not name:
                message = {
                    "id": "msg_done",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
                packets = [
                    packets[0],
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**message, "status": "in_progress"},
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": message,
                    },
                    {
                        **packets[-1],
                        "response": {**packets[-1]["response"], "output": [message]},
                    },
                ]
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(
                    "data: " + json.dumps({**packet, "sequence_number": i}) + "\n\n"
                    for i, packet in enumerate(packets)
                ),
            )
        delta = (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{step}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ]
            }
            if name
            else {"content": "done"}
        )
        packets = [
            {
                "id": "completion",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "example",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "completion",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "example",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if name else "stop",
                    }
                ],
            },
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(packet) + "\n\n" for packet in packets)
            + "data: [DONE]\n\n",
        )

    if native:
        provider, _, _ = _provider_with_wire_transports(
            _catalog_payload(), generation_response=upstream
        )
    else:
        provider = cast(
            OpenAIChatProvider,
            create_provider(
                provider_id,
                Settings(
                    open_router_api_key="test",
                    nvidia_nim_api_key="test",
                    groq_api_key="test",
                    mistral_api_key="test",
                ),
            ),
        )
        provider._admission = immediate_admission(provider_name=provider_id)
        await provider._client.close()
        provider._client = AsyncOpenAI(
            api_key="test",
            base_url="https://provider.test/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(upstream)),
            max_retries=0,
        )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Find a tool and use it"}
    ]

    async def turn() -> dict[str, Any]:
        request = OpenAIResponsesRequest.model_validate(
            {
                "model": "responses-selector" if native else "example",
                "input": history,
                "tools": [SEARCH],
                "max_output_tokens": 128,
            }
        )
        output = "".join([chunk async for chunk in provider.stream_responses(request)])
        events = parse_sse_text(output)
        assert events[-1].event == "response.completed"
        response = events[-1].data["response"]
        done_items = [
            event.data["item"]
            for event in events
            if event.event == "response.output_item.done"
        ]
        assert response["output"] == done_items
        sequences = [event.data["sequence_number"] for event in events]
        assert sequences == sorted(set(sequences))
        return response

    try:
        search = (await turn())["output"][0]
        assert search["type"] == "tool_search_call"
        assert search["execution"] == "client"
        assert search["arguments"] == {"query": "agent"}
        assert "name" not in search
        definitions = (
            [
                {
                    "type": "namespace",
                    "name": "editor",
                    "tools": [
                        {"type": "custom", "name": "edit", "format": {"type": "text"}}
                    ],
                }
            ]
            if custom
            else [AGENTS]
        )
        history.extend(
            [
                search,
                {
                    "type": "tool_search_output",
                    "execution": "client",
                    "status": "completed",
                    "call_id": search["call_id"],
                    "tools": definitions,
                },
            ]
        )
        call = (await turn())["output"][0]
        assert call["type"] == ("custom_tool_call" if custom else "function_call")
        assert call["namespace"] == ("editor" if custom else "agents")
        assert call["name"] == ("edit" if custom else "spawn_agent")
        if custom:
            assert call["input"] == "patch"
        else:
            assert json.loads(call["arguments"]) == {"message": "hello"}
        history.extend(
            [
                call,
                {
                    "type": "custom_tool_call_output"
                    if custom
                    else "function_call_output",
                    "call_id": call["call_id"],
                    "output": "completed",
                },
            ]
        )
        assert (await turn())["output"][0]["content"][0]["text"] == "done"
        assert len(requests) == 3
    finally:
        await provider.cleanup()
