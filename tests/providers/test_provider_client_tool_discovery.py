import json
from typing import Any, cast
from unittest.mock import patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.constants import PROVIDER_MAX_ATTEMPTS_DEFAULT
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject, load_json_value
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from free_claude_code.providers.runtime.runtime import create_provider
from tests.core.openai_responses.test_client_tool_discovery import AGENTS, SEARCH
from tests.providers.test_opencode import (
    _catalog_payload,
    _provider_with_wire_transports,
    _responses_event_stream,
)
from tests.wire_payload import (
    array_payload,
    decode_object,
    int_payload,
    object_array,
    object_payload,
    string_payload,
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
        body = decode_object(request.content)
        requests.append(body)
        functions = (
            {
                object_payload(tool)["name"]: tool
                for tool in array_payload(body.get("tools", []))
            }
            if native
            else {
                object_payload(object_payload(tool)["function"])[
                    "name"
                ]: object_payload(tool)["function"]
                for tool in array_payload(body.get("tools", []))
            }
        )
        messages = body["input"] if native else body["messages"]
        result_field = "output" if native else "content"
        step = len(requests)
        if step == 1:
            assert len(functions) == 1
            name = next(iter(functions))
            assert (
                object_payload(
                    object_payload(
                        object_payload(object_payload(functions[name])["parameters"])[
                            "properties"
                        ]
                    )["query"]
                )["type"]
                == "string"
            )
            args = '{"query":"agent"}'
        elif step == 2:
            name = string_payload(
                next(name for name in functions if name != "fcc_tool_search")
            )
            if not native:
                assert len(string_payload(name)) <= 64
                assert all(
                    char.isascii() and (char.isalnum() or char in "_-") for char in name
                )
            assert body["tool_choice"] == (
                {"type": "function", "name": name}
                if native
                else {"type": "function", "function": {"name": name}}
            )
            assert (
                object_payload(array_payload(messages)[-1]).get("type")
                == "function_call_output"
                if native
                else object_array(messages)[-1]["role"] == "tool"
            )
            discovered = object_array(
                load_json_value(
                    string_payload(object_array(messages)[-1][result_field])
                )
            )
            assert len(discovered) == 1
            assert object_array(discovered)[0]["name"] == name
            assert (
                object_array(discovered)[0]["parameters"]
                == object_payload(functions[name])["parameters"]
            )
            args = '{"input":"patch"}' if custom else '{"message":"hello"}'
        else:
            if native:
                replayed_call = object_array(messages)[-2]
                assert object_payload(replayed_call)["type"] == "function_call"
                assert (
                    object_payload(replayed_call)["name"]
                    == requests[1]["tool_choice"]["name"]
                )
                assert "namespace" not in replayed_call
                assert "namespace" not in object_payload(
                    functions[replayed_call["name"]]
                )
                assert load_json_value(
                    string_payload(object_payload(replayed_call)["arguments"])
                ) == ({"input": "patch"} if custom else {"message": "hello"})
            else:
                replayed_call = array_payload(object_array(messages)[-2]["tool_calls"])[
                    0
                ]
                function = object_payload(replayed_call)["function"]
                assert (
                    object_payload(function)["name"]
                    == requests[1]["tool_choice"]["function"]["name"]
                )
                assert load_json_value(
                    string_payload(object_payload(function)["arguments"])
                ) == ({"input": "patch"} if custom else {"message": "hello"})
                assert object_payload(function)["name"] in functions
                assert "namespace" not in object_payload(function)
            assert (
                object_payload(array_payload(messages)[-1]).get("type")
                == "function_call_output"
                if native
                else object_array(messages)[-1]["role"] == "tool"
            )
            assert object_array(messages)[-1][result_field] == result_text
            name, args = "", ""
        packets: list[JsonObject]
        if native:
            packets = [
                decode_object(line[6:])
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
                        "response": {
                            **object_payload(packets[-1]["response"]),
                            "output": [call],
                        },
                    },
                ]
            if not name:
                message: JsonObject = {
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
                        "response": {
                            **object_payload(packets[-1]["response"]),
                            "output": [message],
                        },
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
        with patch(
            "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
            return_value=client,
        ):
            provider = cast(
                OpenAIChatProvider,
                (
                    await create_provider(
                        provider_id,
                        Settings(
                            open_router_api_keys="test",
                            nvidia_nim_api_keys="test",
                            groq_api_keys="test",
                            mistral_api_keys="test",
                        ),
                        ProviderAdmissionRegistry(
                            ProviderAdmissionLimits(
                                rate_limit=1_000_000,
                                rate_window=1.0,
                                max_concurrency=1_000,
                                max_attempts=PROVIDER_MAX_ATTEMPTS_DEFAULT,
                            )
                        ),
                    )
                ),
            )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Find a tool and use it"}
    ]

    async def turn() -> JsonObject:
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
        output = "".join(
            [
                frame
                async for chunk in provider.stream_responses(request)
                for frame in chunk.frames
            ]
        )
        events = parse_sse_text(output)
        assert events[-1].event == "response.completed"
        response = object_payload(events[-1].data["response"])
        done_items = [
            event.data["item"]
            for event in events
            if event.event == "response.output_item.done"
        ]
        assert object_payload(response)["output"] == done_items
        sequences = [int_payload(event.data["sequence_number"]) for event in events]
        assert sequences == sorted(set(sequences))
        return response

    try:
        search = object_array((await turn())["output"])[0]
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
        call = object_array((await turn())["output"])[0]
        assert call["type"] == ("custom_tool_call" if custom else "function_call")
        assert call.get("namespace") == namespace
        assert call["name"] == client_name
        if custom:
            assert call["input"] == "patch"
        else:
            assert load_json_value(string_payload(call["arguments"])) == {
                "message": "hello"
            }
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
        assert (
            object_array(object_array((await turn())["output"])[0]["content"])[0][
                "text"
            ]
            == "done"
        )
        assert len(requests) == 3
    finally:
        await provider.cleanup()
