"""Executable contract for the two-ingress by two-upstream protocol matrix."""

import json
from collections.abc import AsyncIterator, Callable
from unittest.mock import patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import MessagesRequest, ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from tests.providers.support import immediate_admission, make_provider_config


def _messages_request() -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "upstream-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
        }
    )


def _responses_request() -> OpenAIResponsesRequest:
    return OpenAIResponsesRequest.model_validate(
        {
            "model": "upstream-model",
            "input": "hello",
            "max_output_tokens": 64,
        }
    )


def _image_messages_request() -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "upstream-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_image",
                            "name": "read_image",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_image",
                            "content": [
                                {"type": "text", "text": "result"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://images.example.test/result.png",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
            "max_tokens": 64,
        }
    )


def _image_responses_request() -> OpenAIResponsesRequest:
    return OpenAIResponsesRequest.model_validate(
        {
            "model": "upstream-model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_image",
                    "name": "read_image",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_image",
                    "output": [
                        {"type": "input_text", "text": "result"},
                        {
                            "type": "input_image",
                            "image_url": "https://images.example.test/result.png",
                        },
                    ],
                },
            ],
            "max_output_tokens": 64,
        }
    )


def _client(handler: Callable[[httpx2.Request], httpx2.Response]) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _chat_sse(text: str) -> str:
    chunks = (
        {
            "id": "chatcmpl_matrix",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_matrix",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        },
    )
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def _response_payload(*, status: str, text: str) -> dict[str, object]:
    output = (
        []
        if status == "in_progress"
        else [
            {
                "id": "msg_matrix",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ]
    )
    return {
        "id": "resp_matrix",
        "created_at": 1,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "model": "upstream-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "background": False,
        "conversation": None,
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "prompt_cache_key": None,
        "reasoning": None,
        "safety_identifier": None,
        "service_tier": "default",
        "status": status,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": (
            None
            if status == "in_progress"
            else {
                "input_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 3,
            }
        ),
        "user": None,
        "store": False,
    }


def _responses_sse(text: str) -> str:
    events = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _response_payload(status="in_progress", text=text),
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_matrix",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": _response_payload(status="completed", text=text),
        },
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_chat_upstream_accepts_both_ingress_protocols_directly() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_sse("chat-ok"),
        )

    client = _client(handler)
    config = make_provider_config(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=client,
    ):
        provider = OpenAIChatProvider(
            config,
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    provider_name="MATRIX_CHAT",
                    reasoning_replay=ReasoningReplayMode.DISABLED,
                ),
                NO_REASONING,
            ),
            admission=immediate_admission(provider_name="MATRIX_CHAT"),
        )
    try:
        messages_body = await _collect(
            provider.stream_messages(
                _messages_request(),
                input_tokens=2,
                request_id="req_matrix_messages_chat",
                response_model="public-model",
            )
        )
        responses_body = await _collect(
            provider.stream_responses(
                _responses_request(),
                input_tokens=2,
                request_id="req_matrix_responses_chat",
                response_model="public-model",
            )
        )
    finally:
        await provider.cleanup()

    assert [request.url.path for request in requests] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    messages_events = parse_sse_text(messages_body)
    assert_anthropic_stream_contract(messages_events)
    assert text_content(messages_events) == "chat-ok"
    responses_events = parse_sse_text(responses_body)
    assert responses_events[0].event == "response.created"
    assert responses_events[-1].event == "response.completed"
    assert responses_events[-1].data["response"]["model"] == "public-model"
    assert (
        responses_events[-1].data["response"]["output"][0]["content"][0]["text"]
        == "chat-ok"
    )


@pytest.mark.asyncio
async def test_responses_upstream_accepts_both_ingress_protocols_directly() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_responses_sse("responses-ok"),
        )

    client = _client(handler)
    transport = OpenAIResponsesTransport(
        client=client,
        admission=immediate_admission(provider_name="MATRIX_RESPONSES"),
        provider_name="MATRIX_RESPONSES",
        read_timeout_s=120.0,
        log_raw_sse_events=False,
    )
    try:
        messages_body = await _collect(
            transport.stream_messages(
                _messages_request(),
                input_tokens=2,
                request_id="req_matrix_messages_responses",
                response_model="public-model",
                reasoning=DEFAULT_REASONING_POLICY,
            )
        )
        responses_body = await _collect(
            transport.stream_responses(
                _responses_request(),
                input_tokens=2,
                request_id="req_matrix_responses_responses",
                response_model="public-model",
                reasoning=DEFAULT_REASONING_POLICY,
            )
        )
    finally:
        await client.close()

    assert [request.url.path for request in requests] == [
        "/v1/responses",
        "/v1/responses",
    ]
    messages_events = parse_sse_text(messages_body)
    assert_anthropic_stream_contract(messages_events)
    assert text_content(messages_events) == "responses-ok"
    responses_events = parse_sse_text(responses_body)
    assert [event.event for event in responses_events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert responses_events[0].data["response"]["id"] == "resp_matrix"
    assert responses_events[-1].data["response"]["model"] == "public-model"
    assert (
        responses_events[-1].data["response"]["output"][0]["content"][0]["text"]
        == "responses-ok"
    )


@pytest.mark.asyncio
async def test_image_tool_output_remains_visual_across_all_protocol_cells() -> None:
    chat_requests: list[httpx2.Request] = []
    responses_requests: list[httpx2.Request] = []

    def chat_handler(request: httpx2.Request) -> httpx2.Response:
        chat_requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_sse("chat-ok"),
        )

    def responses_handler(request: httpx2.Request) -> httpx2.Response:
        responses_requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_responses_sse("responses-ok"),
        )

    chat_client = _client(chat_handler)
    config = make_provider_config(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=chat_client,
    ):
        chat_provider = OpenAIChatProvider(
            config,
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    provider_name="MATRIX_CHAT",
                    reasoning_replay=ReasoningReplayMode.DISABLED,
                ),
                NO_REASONING,
            ),
            admission=immediate_admission(provider_name="MATRIX_CHAT"),
        )

    responses_client = _client(responses_handler)
    responses_transport = OpenAIResponsesTransport(
        client=responses_client,
        admission=immediate_admission(provider_name="MATRIX_RESPONSES"),
        provider_name="MATRIX_RESPONSES",
        read_timeout_s=120.0,
        log_raw_sse_events=False,
    )
    try:
        await _collect(
            chat_provider.stream_messages(
                _image_messages_request(),
                input_tokens=2,
                request_id="req_matrix_image_messages_chat",
                response_model="public-model",
            )
        )
        await _collect(
            chat_provider.stream_responses(
                _image_responses_request(),
                input_tokens=2,
                request_id="req_matrix_image_responses_chat",
                response_model="public-model",
            )
        )
        await _collect(
            responses_transport.stream_messages(
                _image_messages_request(),
                input_tokens=2,
                request_id="req_matrix_image_messages_responses",
                response_model="public-model",
                reasoning=DEFAULT_REASONING_POLICY,
            )
        )
        await _collect(
            responses_transport.stream_responses(
                _image_responses_request(),
                input_tokens=2,
                request_id="req_matrix_image_responses_responses",
                response_model="public-model",
                reasoning=DEFAULT_REASONING_POLICY,
            )
        )
    finally:
        await chat_provider.cleanup()
        await responses_client.close()

    image_url = "https://images.example.test/result.png"
    messages_chat = json.loads(chat_requests[0].content)
    responses_chat = json.loads(chat_requests[1].content)
    messages_responses = json.loads(responses_requests[0].content)
    responses_responses = json.loads(responses_requests[1].content)

    for chat_body in (messages_chat, responses_chat):
        tool_message = next(
            message for message in chat_body["messages"] if message["role"] == "tool"
        )
        assert image_url not in tool_message["content"]
        assert any(
            part.get("image_url", {}).get("url") == image_url
            for message in chat_body["messages"]
            if message["role"] == "user" and isinstance(message["content"], list)
            for part in message["content"]
        )

    assert messages_responses["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_image",
        "output": [
            {"type": "input_text", "text": "result"},
            {"type": "input_image", "image_url": image_url},
        ],
    }
    assert responses_responses["input"] == _image_responses_request().input
