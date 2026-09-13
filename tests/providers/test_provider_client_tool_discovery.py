import json
from typing import Any, cast
from unittest.mock import patch

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
@pytest.mark.parametrize("omit_discovery_metadata", [False, True])
@pytest.mark.parametrize(
    ("tool_name", "namespaced"),
    [(None, True), ("apply patch", False), ("x" * 80, False), ("apply patch", True)],
)
async def test_provider_discovery_call_and_result_round_trip(
    provider_id: str,
    custom: bool,
    omit_discovery_metadata: bool,
    tool_name: str | None,
    namespaced: bool,
) -> None:
    native = provider_id == "opencode_zen"
    requests: list[dict[str, Any]] = []
    client_name = tool_name or ("edit" if custom else "spawn_agent")
    namespace = ("editor" if custom else "agents") if namespaced else None
    result_text = json.dumps([{"type": "function", "name": client_name}])

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
            name = next(name for name in functions if name != "fcc_tool_search")
            if not native:
                assert len(name) <= 64
                assert all(
                    char.isascii() and (char.isalnum() or char in "_-") for char in name
                )
            assert body["tool_choice"] == (
                {"type": "function", "name": name}
                if native
                else {"type": "function", "function": {"name": name}}
            )
            assert (
                messages[-1].get("type") == "function_call_output"
                if native
                else messages[-1]["role"] == "tool"
            )
            discovered = json.loads(messages[-1][result_field])
            assert len(discovered) == 1
            assert discovered[0]["name"] == name
            assert discovered[0]["parameters"] == functions[name]["parameters"]
            args = '{"input":"patch"}' if custom else '{"message":"hello"}'
        else:
            if native:
                replayed_call = messages[-2]
                assert replayed_call["type"] == "function_call"
                assert replayed_call["name"] == requests[1]["tool_choice"]["name"]
                assert "namespace" not in replayed_call
                assert "namespace" not in functions[replayed_call["name"]]
                assert json.loads(replayed_call["arguments"]) == (
                    {"input": "patch"} if custom else {"message": "hello"}
                )
            else:
                replayed_call = messages[-2]["tool_calls"][0]
                function = replayed_call["function"]
                assert (
                    function["name"] == requests[1]["tool_choice"]["function"]["name"]
                )
                assert json.loads(function["arguments"]) == (
                    {"input": "patch"} if custom else {"message": "hello"}
                )
                assert function["name"] in functions
                assert "namespace" not in function
            assert (
                messages[-1].get("type") == "function_call_output"
                if native
                else messages[-1]["role"] == "tool"
            )
            assert messages[-1][result_field] == result_text
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
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://provider.test/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(upstream)),
            max_retries=0,
        )
        with (
            patch(
                "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
                return_value=client,
            ),
            patch(
                "free_claude_code.providers.runtime.factory.ProviderAdmissionController",
                return_value=immediate_admission(provider_name=provider_id),
            ),
        ):
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
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Find a tool and use it"}
    ]

    async def turn() -> dict[str, Any]:
        request = OpenAIResponsesRequest.model_validate(
            {
                "model": "responses-selector" if native else "example",
                "input": history,
                "tools": [SEARCH],
                "tool_choice": {
                    "type": "tool",
                    "namespace": namespace,
                    "name": client_name,
                }
                if len(requests) == 1
                else "auto",
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
        definition = (
            {"type": "custom", "name": client_name, "format": {"type": "text"}}
            if custom
            else {**cast(list[dict[str, Any]], AGENTS["tools"])[0], "name": client_name}
        )
        definitions = (
            [{"type": "namespace", "name": namespace, "tools": [definition]}]
            if namespaced
            else [definition]
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
        if omit_discovery_metadata:
            history[-1].pop("execution")
            history[-1].pop("status")
        call = (await turn())["output"][0]
        assert call["type"] == ("custom_tool_call" if custom else "function_call")
        assert call.get("namespace") == namespace
        assert call["name"] == client_name
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
                    "output": result_text,
                },
            ]
        )
        assert (await turn())["output"][0]["content"][0]["text"] == "done"
        assert len(requests) == 3
    finally:
        await provider.cleanup()
