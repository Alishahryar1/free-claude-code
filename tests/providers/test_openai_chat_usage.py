"""OpenAI-chat streamed usage helper tests."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import openai
import pytest
from httpx2 import Request, Response
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.openai_chat import (
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.openai_chat.reasoning import NO_REASONING
from free_claude_code.providers.openai_chat.usage import (
    clone_without_stream_usage,
    is_stream_usage_rejection,
    request_stream_usage,
    usage_int,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)


class _UsageTestBehavior(OpenAIChatBehavior):
    def build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        return {"model": request.model, "messages": [{"role": "user", "content": "x"}]}


class _UsageTestProvider(OpenAIChatProvider):
    def __init__(self):
        super().__init__(
            make_provider_config(
                api_key="test_key",
                base_url="https://provider.example/v1",
                rate_limit=100,
                rate_window=60,
            ),
            behavior=_UsageTestBehavior(
                OpenAIChatProfile(
                    OpenAIChatRequestPolicy(
                        provider_name="USAGE_TEST",
                        reasoning_replay=ReasoningReplayMode.DISABLED,
                    ),
                    NO_REASONING,
                )
            ),
            admission=immediate_admission(),
        )


def _bad_request(message: str, body: object | None = None) -> openai.BadRequestError:
    response = Response(
        400,
        request=Request("POST", "https://provider.example/v1/chat/completions"),
    )
    return openai.BadRequestError(message, response=response, body=body)


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
):
    if content is None and finish_reason is None:
        return SimpleNamespace(choices=[], usage=usage)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def test_request_stream_usage_adds_stream_options_when_absent():
    body = {"model": "m"}

    request_stream_usage(body)

    assert body["stream_options"] == {"include_usage": True}


def test_request_stream_usage_preserves_existing_stream_options():
    stream_options = {"foo": "bar"}
    body = {"model": "m", "stream_options": stream_options}

    request_stream_usage(body)

    assert body["stream_options"] == {"foo": "bar", "include_usage": True}
    assert body["stream_options"] is stream_options


def test_clone_without_stream_usage_removes_only_include_usage():
    body = {
        "model": "m",
        "stream_options": {"foo": "bar", "include_usage": True},
    }

    retry_body = clone_without_stream_usage(body)

    assert retry_body == {"model": "m", "stream_options": {"foo": "bar"}}
    assert body["stream_options"] == {"foo": "bar", "include_usage": True}


def test_clone_without_stream_usage_drops_empty_stream_options():
    body = {"model": "m", "stream_options": {"include_usage": True}}

    retry_body = clone_without_stream_usage(body)

    assert retry_body == {"model": "m"}


def test_usage_int_reads_dict_object_and_model_extra():
    assert usage_int({"prompt_tokens": 11}, "prompt_tokens") == 11
    assert usage_int(SimpleNamespace(completion_tokens=7), "completion_tokens") == 7
    assert (
        usage_int(
            SimpleNamespace(model_extra={"prompt_cache_hit_tokens": 3}),
            "prompt_cache_hit_tokens",
        )
        == 3
    )
    assert usage_int(SimpleNamespace(prompt_tokens=None), "prompt_tokens") is None
    assert usage_int({"prompt_tokens": True}, "prompt_tokens") is None


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"prompt_tokens_details": {"cache_write_tokens": 5}},
            5,
        ),
        (
            CompletionUsage(
                completion_tokens=4,
                prompt_tokens=22,
                total_tokens=26,
                prompt_tokens_details=PromptTokensDetails(cache_write_tokens=5),
            ),
            5,
        ),
        (
            SimpleNamespace(
                prompt_tokens_details=None,
                model_extra={
                    "prompt_tokens_details": {"cache_write_tokens": 5},
                },
            ),
            5,
        ),
        ({"prompt_tokens_details": {"cache_write_tokens": 0}}, 0),
        ({"prompt_tokens_details": {"cache_write_tokens": True}}, None),
        ({"prompt_tokens_details": {"cache_write_tokens": 5.0}}, None),
        ({"prompt_tokens_details": {"cache_write_tokens": "5"}}, None),
        ({"prompt_tokens_details": None}, None),
    ],
)
def test_extracts_standard_chat_cache_write_tokens(usage, expected) -> None:
    provider = _UsageTestProvider()

    assert provider._behavior.cache_write_input_tokens(usage) == expected


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {
                "prompt_tokens": 22,
                "prompt_tokens_details": {
                    "cached_tokens": 15,
                    "cache_write_tokens": 3,
                },
            },
            {
                "input_tokens": 4,
                "cache_read_input_tokens": 15,
                "cache_creation_input_tokens": 3,
            },
        ),
        (
            CompletionUsage(
                completion_tokens=4,
                prompt_tokens=22,
                total_tokens=26,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=15,
                    cache_write_tokens=3,
                ),
            ),
            {
                "input_tokens": 4,
                "cache_read_input_tokens": 15,
                "cache_creation_input_tokens": 3,
            },
        ),
        (
            SimpleNamespace(
                prompt_tokens=22,
                prompt_tokens_details=None,
                model_extra={
                    "prompt_tokens_details": {
                        "cached_tokens": 15,
                        "cache_write_tokens": 3,
                    },
                },
            ),
            {
                "input_tokens": 4,
                "cache_read_input_tokens": 15,
                "cache_creation_input_tokens": 3,
            },
        ),
        (
            {
                "prompt_tokens": 22,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            {"input_tokens": 22, "cache_read_input_tokens": 0},
        ),
        (
            {
                "prompt_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            {"input_tokens": 0, "cache_read_input_tokens": 0},
        ),
        (
            {
                "prompt_tokens": 22,
                "prompt_tokens_details": {"cache_write_tokens": 3},
            },
            {"input_tokens": 19, "cache_creation_input_tokens": 3},
        ),
        (
            {
                "prompt_tokens": 22,
                "prompt_tokens_details": {
                    "cached_tokens": 15,
                    "cache_write_tokens": 8,
                },
            },
            {"input_tokens": 7, "cache_read_input_tokens": 15},
        ),
    ],
)
def test_maps_standard_chat_cache_usage_to_anthropic_fields(usage, expected):
    provider = _UsageTestProvider()

    assert provider._behavior.anthropic_usage_fields(usage) == expected


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens_details": {"cached_tokens": 15}},
        {"prompt_tokens": 22},
        {"prompt_tokens": 22, "prompt_tokens_details": None},
        {"prompt_tokens": 22, "prompt_tokens_details": "invalid"},
        {
            "prompt_tokens": -1,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        {
            "prompt_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": -1},
        },
        {
            "prompt_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": 23},
        },
        {
            "prompt_tokens": True,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        {
            "prompt_tokens": 22.0,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        {
            "prompt_tokens": "22",
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        {
            "prompt_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": True},
        },
        {
            "prompt_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": 15.0},
        },
        {
            "prompt_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": "15"},
        },
    ],
)
def test_ignores_incomplete_or_inconsistent_standard_cache_usage(usage):
    provider = _UsageTestProvider()

    assert provider._behavior.anthropic_usage_fields(usage) == {}


def test_stream_usage_rejection_matches_usage_option_400():
    error = _bad_request(
        "Unrecognized request argument supplied: stream_options",
        {"error": {"message": "stream_options is unsupported"}},
    )

    assert is_stream_usage_rejection(error)


def test_stream_usage_rejection_does_not_match_unrelated_400():
    error = _bad_request(
        "messages: invalid role",
        {"error": {"message": "messages contains invalid role"}},
    )

    assert not is_stream_usage_rejection(error)


@pytest.mark.asyncio
async def test_openai_chat_stream_requests_usage_and_uses_provider_prompt_tokens():
    provider = _UsageTestProvider()
    request = make_messages_request(model="m")
    usage = CompletionUsage(
        completion_tokens=4,
        prompt_tokens=22,
        total_tokens=26,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=15,
            cache_write_tokens=3,
        ),
    )
    create = AsyncMock(
        return_value=_stream(
            [
                _chunk(content="hello"),
                _chunk(finish_reason="stop"),
                _chunk(usage=usage),
            ]
        )
    )

    with patch.object(provider._client.chat.completions, "create", create):
        events = [
            event async for event in provider.stream_messages(request, input_tokens=7)
        ]

    create.assert_awaited_once()
    await_args = create.await_args
    assert await_args is not None
    assert await_args.kwargs["stream_options"] == {"include_usage": True}
    parsed = parse_sse_text("".join(events))
    start_usage = next(
        event.data["message"]["usage"]
        for event in parsed
        if event.event == "message_start"
    )
    final_usage = next(
        event.data["usage"] for event in parsed if event.event == "message_delta"
    )
    assert start_usage["input_tokens"] == 7
    assert final_usage == {
        "input_tokens": 4,
        "output_tokens": 4,
        "cache_read_input_tokens": 15,
        "cache_creation_input_tokens": 3,
    }
    assert sum(event.event == "message_delta" for event in parsed) == 1
    assert sum(event.event == "message_stop" for event in parsed) == 1


@pytest.mark.asyncio
async def test_openai_chat_nonstream_message_uses_final_cache_partition():
    provider = _UsageTestProvider()
    request = make_messages_request(model="m")
    usage = CompletionUsage(
        completion_tokens=4,
        prompt_tokens=22,
        total_tokens=26,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=15,
            cache_write_tokens=3,
        ),
    )
    create = AsyncMock(
        return_value=_stream(
            [
                _chunk(content="hello"),
                _chunk(finish_reason="stop"),
                _chunk(usage=usage),
            ]
        )
    )

    with patch.object(provider._client.chat.completions, "create", create):
        message, error, _complete = await aggregate_anthropic_sse_to_message(
            provider.stream_messages(request, input_tokens=7)
        )

    assert error is None
    assert message["usage"] == {
        "input_tokens": 4,
        "output_tokens": 4,
        "cache_read_input_tokens": 15,
        "cache_creation_input_tokens": 3,
    }


@pytest.mark.asyncio
async def test_openai_chat_responses_stream_preserves_cache_write_usage():
    provider = _UsageTestProvider()
    request = OpenAIResponsesRequest.model_validate({"model": "m", "input": "hello"})
    usage = CompletionUsage(
        completion_tokens=4,
        prompt_tokens=22,
        total_tokens=26,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=15,
            cache_write_tokens=3,
        ),
    )
    create = AsyncMock(
        return_value=_stream(
            [
                _chunk(content="hello"),
                _chunk(finish_reason="stop"),
                _chunk(usage=usage),
            ]
        )
    )

    with patch.object(provider._client.chat.completions, "create", create):
        events = [
            event async for event in provider.stream_responses(request, input_tokens=7)
        ]

    completed = next(
        event.data["response"]
        for event in parse_sse_text("".join(events))
        if event.event == "response.completed"
    )
    assert completed["usage"] == {
        "input_tokens": 22,
        "input_tokens_details": {
            "cached_tokens": 15,
            "cache_write_tokens": 3,
        },
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 26,
    }


@pytest.mark.asyncio
async def test_openai_chat_stream_keeps_response_model_separate_from_upstream_model():
    provider = _UsageTestProvider()
    request = make_messages_request(model="upstream/model")
    create = AsyncMock(
        return_value=_stream(
            [
                _chunk(content="hello"),
                _chunk(finish_reason="stop"),
            ]
        )
    )

    with patch.object(provider._client.chat.completions, "create", create):
        events = [
            event
            async for event in provider.stream_messages(
                request,
                response_model="anthropic/test/upstream/model",
            )
        ]

    assert create.await_args is not None
    assert create.await_args.kwargs["model"] == "upstream/model"
    message_start = next(
        event.data["message"]
        for event in parse_sse_text("".join(events))
        if event.event == "message_start"
    )
    assert message_start["model"] == "anthropic/test/upstream/model"


@pytest.mark.asyncio
async def test_openai_chat_stream_retries_without_usage_when_option_is_rejected():
    provider = _UsageTestProvider()
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    request_stream_usage(body)
    create = AsyncMock(
        side_effect=[
            _bad_request(
                "stream_options is unsupported",
                {"error": {"message": "stream_options is unsupported"}},
            ),
            object(),
        ]
    )

    with patch.object(provider._client.chat.completions, "create", create):
        (
            _stream_obj,
            used_body,
            attempt,
            _sent_body,
        ) = await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["stream_options"] == {"include_usage": True}
    assert "stream_options" not in create.await_args_list[1].kwargs
    assert "stream_options" not in used_body
