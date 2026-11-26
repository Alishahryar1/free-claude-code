"""Tests for DeepSeek OpenAI-compatible Chat Completions provider."""

import json
import logging
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from openai import BadRequestError

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import DEEPSEEK_DEFAULT_BASE
from free_claude_code.core.anthropic.models import (
    ContentBlockDocument,
    ContentBlockImage,
    ContentBlockText,
    Message,
    MessagesRequest,
    Tool,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_chat import is_synthetic_chat_tool_turn_boundary
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.deepseek import DeepSeekProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    SDKStreamDouble,
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    reasoning_for,
    sdk_chat_chunk,
)
from tests.wire_payload import (
    array_payload,
    object_array,
    object_payload,
    string_payload,
)


@pytest.fixture
def deepseek_config():
    return make_provider_config(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_DEFAULT_BASE,
    )


@pytest.fixture
def deepseek_provider(deepseek_config: ProviderConfig):
    return DeepSeekProvider(deepseek_config, admission=immediate_admission())


def test_default_base_url_alias() -> None:
    assert DEEPSEEK_DEFAULT_BASE == "https://api.deepseek.com"


def test_init(deepseek_config: ProviderConfig) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_client:
        provider = DeepSeekProvider(deepseek_config, admission=immediate_admission())
    assert provider._api_key == "test_deepseek_key"
    assert provider._base_url == "https://api.deepseek.com"
    assert mock_client.called


def test_responses_request_uses_deepseek_chat_policy(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "deepseek-chat",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file_path":"x"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
                {"role": "user", "content": "Continue"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )

    translated = deepseek_provider._chat._build_responses_request_body(
        request, reasoning=REASONING_ON
    )

    assert translated.body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert translated.body["tool_choice"] == "auto"
    assert translated.body["extra_body"] == {"thinking": {"type": "enabled"}}


def test_responses_tool_history_keeps_deepseek_replayable_reasoning(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "deepseek-chat",
            "input": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Inspect first"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file_path":"x"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
        }
    )

    translated = deepseek_provider._chat._build_responses_request_body(
        request, reasoning=REASONING_ON
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assert messages[0]["reasoning_content"] == "Inspect first"
    assert translated.body["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.parametrize(
    ("usage", "anthropic_expected", "responses_cached"),
    [
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": 20,
            },
            {"input_tokens": 20, "cache_read_input_tokens": 10},
            10,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 30,
            },
            {"input_tokens": 30, "cache_read_input_tokens": 0},
            0,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": 30,
                "prompt_cache_miss_tokens": 0,
            },
            {"input_tokens": 0, "cache_read_input_tokens": 30},
            30,
        ),
        (
            {"prompt_cache_hit_tokens": 10, "prompt_cache_miss_tokens": 20},
            {"input_tokens": 20, "cache_read_input_tokens": 10},
            None,
        ),
        (
            {"prompt_tokens": 30, "prompt_cache_miss_tokens": 20},
            {},
            None,
        ),
        (
            {"prompt_tokens": 30, "prompt_cache_hit_tokens": 10},
            {},
            None,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": True,
                "prompt_cache_miss_tokens": 20,
            },
            {},
            None,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": 20.0,
            },
            {},
            None,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": -1,
                "prompt_cache_miss_tokens": 31,
            },
            {},
            None,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": -1,
            },
            {},
            None,
        ),
        (
            {
                "prompt_tokens": -1,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": 20,
            },
            {"input_tokens": 20, "cache_read_input_tokens": 10},
            None,
        ),
        (
            {
                "prompt_tokens": 30,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": 19,
            },
            {},
            None,
        ),
    ],
)
def test_maps_only_complete_consistent_cache_usage(
    deepseek_provider: DeepSeekProvider,
    usage: JsonObject,
    anthropic_expected: JsonObject,
    responses_cached: int | None,
) -> None:
    assert (
        deepseek_provider._behavior.anthropic_usage_fields(usage) == anthropic_expected
    )
    assert deepseek_provider._behavior.cached_input_tokens(usage) == responses_cached


def test_build_request_body_openai_chat_shape(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest(
        model="deepseek-v4-pro",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="S",
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert body["model"] == "deepseek-v4-pro"
    assert "stream" not in body
    assert array_payload(body["messages"])[0] == {"role": "system", "content": "S"}
    assert object_payload(array_payload(body["messages"])[1])["role"] == "user"
    assert array_payload(body["messages"])[1] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100
    assert "stream_options" not in body


def test_build_request_body_default_max_tokens(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert object_payload(body["extra_body"])["thinking"] == {"type": "enabled"}


def test_build_request_body_tool_list_keeps_thinking(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert object_payload(body["extra_body"])["thinking"] == {"type": "enabled"}
    assert (
        object_payload(object_payload(array_payload(body["tools"])[0])["function"])[
            "name"
        ]
        == "Read"
    )


def test_build_request_body_tool_choice_keeps_thinking(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "auto"},
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert object_payload(body["extra_body"])["thinking"] == {"type": "enabled"}
    assert body["tool_choice"] == "auto"


def test_build_request_body_forced_tool_choice_is_rejected(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "tool", "name": "Read"},
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    with pytest.raises(InvalidRequestError, match="named tool_choice"):
        deepseek_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_build_request_body_encodes_reasoning_off() -> None:
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._chat._build_request_body(request, reasoning=REASONING_OFF)
    assert object_payload(body["extra_body"])["thinking"] == {"type": "disabled"}
    assert "stream_options" not in body


def test_non_tool_thinking_survives_conversion(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "plain",
                            "signature": None,
                        },
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert array_payload(body["messages"])[0] == {
        "role": "assistant",
        "content": "out",
        "reasoning_content": "plain",
    }


def test_redacted_thinking_is_kept_until_destination_preparation(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert object_payload(array_payload(body["messages"])[0])["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque"}
    ]
    assert request.model_dump()["messages"][0]["content"][0]["data"] == "opaque"


def test_tool_history_with_replayable_thinking_preserves_thinking(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "hidden",
                            "signature": "sig_123",
                        },
                        {"type": "redacted_thinking", "data": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "output_config": {"effort": "high"},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "high"
    assert "context_management" not in body
    assert "output_config" not in body
    assistant = array_payload(body["messages"])[0]
    assert object_payload(assistant)["content"] == ""
    assert object_payload(assistant)["reasoning_content"] == "hidden"
    assert (
        object_payload(
            object_payload(array_payload(object_payload(assistant)["tool_calls"])[0])[
                "function"
            ]
        )["name"]
        == "Read"
    )
    assert (
        object_payload(
            object_payload(array_payload(object_payload(assistant)["tool_calls"])[0])[
                "function"
            ]
        )["arguments"]
        == '{"file_path": "x"}'
    )
    assert array_payload(body["messages"])[1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "ok",
    }


def test_tool_history_with_unsigned_thinking_preserves_thinking(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plain"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert object_payload(body["extra_body"])["thinking"] == {"type": "enabled"}
    assert (
        object_payload(array_payload(body["messages"])[0])["reasoning_content"]
        == "plain"
    )


def test_tool_history_without_thinking_keeps_selected_effort(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "other_edit", "keep": "all"},
                ],
                "other": True,
            },
            "output_config": {"effort": "high"},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "high"
    assert object_payload(body["extra_body"]).get("thinking") != {"type": "disabled"}
    assert object_payload(array_payload(body["messages"])[0])["reasoning_content"] == ""
    assert "context_management" not in body
    assert "output_config" not in body
    assert (
        object_payload(object_payload(array_payload(body["tools"])[0])["function"])[
            "name"
        ]
        == "Read"
    )
    assert body["tool_choice"] == "auto"
    assert (
        object_payload(
            object_payload(
                array_payload(
                    object_payload(array_payload(body["messages"])[0])["tool_calls"]
                )[0]
            )["function"]
        )["name"]
        == "Read"
    )
    assert object_payload(array_payload(body["messages"])[1])["role"] == "tool"


def test_tool_history_with_empty_thinking_preserves_reasoning_state(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert object_payload(body["extra_body"])["thinking"] == {"type": "enabled"}
    assert object_payload(array_payload(body["messages"])[0])["reasoning_content"] == ""
    assert (
        object_payload(
            object_payload(
                array_payload(
                    object_payload(array_payload(body["messages"])[0])["tool_calls"]
                )[0]
            )["function"]
        )["name"]
        == "Read"
    )


def test_tool_history_with_empty_top_level_reasoning_preserves_reasoning_state(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                    "reasoning_content": "",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert object_payload(body["extra_body"])["thinking"] == {"type": "enabled"}
    assert object_payload(array_payload(body["messages"])[0])["reasoning_content"] == ""
    assert (
        object_payload(
            object_payload(
                array_payload(
                    object_payload(array_payload(body["messages"])[0])["tool_calls"]
                )[0]
            )["function"]
        )["name"]
        == "Read"
    )


def test_thinking_off_preserves_historical_reasoning() -> None:
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "sec"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
        }
    )
    body = provider._chat._build_request_body(request, reasoning=REASONING_OFF)
    assert object_payload(body["extra_body"])["thinking"] == {"type": "disabled"}
    assert (
        object_payload(array_payload(body["messages"])[0])["reasoning_content"] == "sec"
    )


def test_thinking_off_still_replays_required_tool_reasoning() -> None:
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "required"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )

    body = provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert object_payload(body["extra_body"])["thinking"] == {"type": "disabled"}
    assert (
        object_payload(array_payload(body["messages"])[0])["reasoning_content"]
        == "required"
    )


def test_passthrough_tool_use_and_result(deepseek_provider: DeepSeekProvider) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "n",
                            "input": {"a": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert (
        object_payload(
            object_payload(
                array_payload(
                    object_payload(array_payload(body["messages"])[0])["tool_calls"]
                )[0]
            )["function"]
        )["name"]
        == "n"
    )
    assert object_payload(array_payload(body["messages"])[1])["role"] == "tool"


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-flash",
        "deepseek-v4-flash-vision-exp",
        "deepseek-v4-flash",
        "deepseek-chat",
        "gateway/future-model",
    ],
)
@pytest.mark.parametrize(
    ("source", "expected_url"),
    [
        (
            {"type": "base64", "media_type": "image/png", "data": "YQ=="},
            "data:image/png;base64,YQ==",
        ),
        (
            {"type": "url", "url": "https://images.example.test/shot.png"},
            "https://images.example.test/shot.png",
        ),
    ],
)
def test_forwards_user_image_for_any_model(
    deepseek_provider: DeepSeekProvider,
    model: str,
    source: JsonObject,
    expected_url: str,
) -> None:
    request = MessagesRequest(
        model=model,
        messages=[
            Message(
                role="user",
                content=[ContentBlockImage(type="image", source=source)],
            )
        ],
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert body["model"] == model
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": expected_url}}],
        }
    ]


def test_vision_model_strips_user_document() -> None:
    """Vision models must receive the text alongside an unsupported document."""
    request = MessagesRequest(
        model="deepseek-v4-flash-vision-exp",
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="what is a PDF?"),
                    ContentBlockDocument(
                        type="document",
                        source={
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "YQ==",
                        },
                    ),
                ],
            )
        ],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    assert (
        object_payload(array_payload(body["messages"])[0])["content"]
        == "what is a PDF?"
    )


def test_request_omits_mcp_servers() -> None:
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        mcp_servers=[{"type": "url", "url": "https://x"}],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    assert "mcp_servers" not in body


def test_request_omits_listed_server_tools_in_tools_list() -> None:
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        tools=[Tool(name="web_search", type="web_search_20250305", input_schema={})],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    assert "tools" not in body


def test_startup_preserves_completed_server_tool_history() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"q": "a"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s1",
                            "content": [],
                        },
                    ],
                }
            ],
        }
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )
    provider.stream_messages(request)
    body = provider._chat._build_request_body(request)
    assert "[Earlier tool record]" in string_payload(
        object_array(body["messages"])[0]["content"]
    )


def test_non_tool_top_level_reasoning_survives_conversion(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest(
        model="m",
        messages=[
            Message(
                role="assistant",
                content="hi",
                reasoning_content="r",
            )
        ],
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert array_payload(body["messages"])[0] == {
        "role": "assistant",
        "content": "hi",
        "reasoning_content": "r",
    }


def test_tool_call_top_level_reasoning_is_replayed(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        }
                    ],
                    "reasoning_content": "required",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert (
        object_payload(array_payload(body["messages"])[0])["reasoning_content"]
        == "required"
    )


@pytest.mark.asyncio
async def test_wire_messages_keep_prefix_and_effort_when_old_reasoning_is_absent(
    deepseek_provider: DeepSeekProvider,
) -> None:
    prefix_messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "ordinary reasoning"},
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "use the first tool"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "required tool reasoning"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "one"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "one",
                }
            ],
        },
        {"role": "user", "content": "use the second tool"},
    ]
    continued_messages = [
        *prefix_messages,
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "two"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "two",
                }
            ],
        },
    ]

    def build(messages: Sequence[Mapping[str, JsonValue]]) -> JsonObject:
        request = MessagesRequest.model_validate(
            {
                "model": "deepseek-v4-pro",
                "messages": messages,
                "thinking": {"type": "enabled"},
            }
        )
        return deepseek_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )

    first_wire = await capture_openai_chat_wire_body(build(prefix_messages))
    continued_wire = await capture_openai_chat_wire_body(build(continued_messages))
    first_messages = first_wire["messages"]
    continued = continued_wire["messages"]

    assert (
        array_payload(continued)[: len(object_array(first_messages))] == first_messages
    )
    assistant_messages = [
        message
        for message in object_array(first_messages)
        if object_payload(message)["role"] == "assistant"
    ]
    assert (
        object_payload(assistant_messages[0])["reasoning_content"]
        == "ordinary reasoning"
    )
    assert (
        object_payload(assistant_messages[1])["reasoning_content"]
        == "required tool reasoning"
    )
    assert first_wire["thinking"] == {"type": "enabled"}
    assert continued_wire["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_stream_uses_chat_completions_and_maps_cache_usage(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )

    async def fake_stream():
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="hello", reasoning_content=None, tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None, reasoning_content=None, tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        )
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    completion_tokens=3,
                    prompt_tokens=30,
                    prompt_cache_hit_tokens=10,
                    prompt_cache_miss_tokens=20,
                ),
            )
        )

    create = AsyncMock(return_value=SDKStreamDouble(fake_stream()))
    with patch.object(deepseek_provider._client.chat.completions, "create", create):
        chunks = [
            frame
            async for chunk in deepseek_provider.stream_messages(
                request, input_tokens=7, request_id="r1"
            )
            for frame in chunk.frames
        ]

    create.assert_awaited_once()
    await_args = create.await_args
    assert await_args is not None
    assert await_args.kwargs["model"] == "m"
    assert await_args.kwargs["stream"] is True
    assert await_args.kwargs["stream_options"] == {"include_usage": True}
    parsed = parse_sse_text("".join(chunks))
    usage = next(
        event.data["usage"] for event in parsed if event.event == "message_delta"
    )
    assert "cache_creation_input_tokens" not in object_payload(usage)
    assert usage == {
        "input_tokens": 20,
        "output_tokens": 3,
        "cache_read_input_tokens": 10,
    }


@pytest.mark.asyncio
async def test_responses_stream_maps_deepseek_cache_usage(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = OpenAIResponsesRequest(model="m", input="hi")

    async def fake_stream():
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="hello", reasoning_content=None, tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None, reasoning_content=None, tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        )
        yield sdk_chat_chunk(
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    completion_tokens=3,
                    prompt_tokens=30,
                    model_extra={
                        "prompt_cache_hit_tokens": 10,
                        "prompt_cache_miss_tokens": 20,
                    },
                ),
            )
        )

    create = AsyncMock(return_value=SDKStreamDouble(fake_stream()))
    with patch.object(deepseek_provider._client.chat.completions, "create", create):
        chunks = [
            frame
            async for chunk in deepseek_provider.stream_responses(
                request, input_tokens=7, request_id="r1"
            )
            for frame in chunk.frames
        ]

    parsed = parse_sse_text("".join(chunks))
    completed = next(
        event.data["response"]
        for event in parsed
        if event.event == "response.completed"
    )
    assert object_payload(completed)["usage"] == {
        "input_tokens": 30,
        "input_tokens_details": {"cached_tokens": 10},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 33,
    }


def test_preserves_extra_body_for_openai_chat_request(
    deepseek_provider: DeepSeekProvider,
) -> None:
    raw = {
        "model": "m",
        "max_tokens": 3,
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"note": 1},
    }
    r = MessagesRequest.model_validate(raw)
    body = deepseek_provider._chat._build_request_body(r, reasoning=reasoning_for(r))
    assert body["extra_body"] == {"note": 1}


def test_normalizes_tool_result_content_array_to_string(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """Test that tool_result content arrays are normalized to strings for DeepSeek API."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "list_dir",
                            "input": {"path": "/"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "file1.txt"},
                                {"type": "text", "text": "file2.txt"},
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    tool_result = array_payload(body["messages"])[1]
    assert object_payload(tool_result)["role"] == "tool"
    assert isinstance(object_payload(tool_result)["content"], str)
    assert "file1.txt" in string_payload(object_payload(tool_result)["content"])
    assert "file2.txt" in string_payload(object_payload(tool_result)["content"])


def test_strips_document_blocks_for_deepseek(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """Document blocks (e.g. PDFs from Claude Code) are stripped since DeepSeek can't process them."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "PDF text extracted",
                        },
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_abc"},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert array_payload(body["messages"])[0] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "PDF text extracted",
    }


def test_preserves_user_image_and_text_order(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "YQ==",
                            },
                        },
                        {"type": "text", "text": "then explain"},
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,YQ=="},
                },
                {"type": "text", "text": "then explain"},
            ],
        }
    ]


def test_normalizes_tool_result_content_dict_to_string(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """Test that tool_result content dicts are normalized to JSON strings."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_data",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": {"status": "success", "data": [1, 2, 3]},
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    tool_result = array_payload(body["messages"])[1]
    assert object_payload(tool_result)["role"] == "tool"
    assert isinstance(object_payload(tool_result)["content"], str)
    assert "status" in string_payload(object_payload(tool_result)["content"])
    assert "success" in string_payload(object_payload(tool_result)["content"])


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-flash-vision-exp"])
def test_preserves_image_and_text_inside_tool_result(
    deepseek_provider: DeepSeekProvider, model: str
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": model,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"path": "shot.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "screenshot saved"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "YQ==",
                                    },
                                },
                                {"type": "text", "text": "inspect the screenshot"},
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert [
        object_payload(message)["role"] for message in array_payload(body["messages"])
    ] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert array_payload(body["messages"])[1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "[Image-bearing tool output follows in user content.]",
    }
    assert is_synthetic_chat_tool_turn_boundary(array_payload(body["messages"])[2])
    assert object_payload(array_payload(body["messages"])[3])["content"] == [
        {"type": "text", "text": 'Image-bearing output for tool call "t1":'},
        {"type": "text", "text": "screenshot saved"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
        {"type": "text", "text": "inspect the screenshot"},
    ]


def test_image_only_tool_result_reaches_user_content(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Screenshot",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://images.example.test/shot.png",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert [
        object_payload(message)["role"] for message in array_payload(body["messages"])
    ] == [
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert array_payload(body["messages"])[1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "[Image-bearing tool output follows in user content.]",
    }
    assert is_synthetic_chat_tool_turn_boundary(array_payload(body["messages"])[2])
    assert object_payload(array_payload(body["messages"])[3])["content"] == [
        {"type": "text", "text": 'Image-bearing output for tool call "t1":'},
        {
            "type": "image_url",
            "image_url": {"url": "https://images.example.test/shot.png"},
        },
    ]


@pytest.mark.parametrize(
    "tool_content",
    [
        [],
        [{"type": "document", "source": {"type": "file", "file_id": "file_pdf"}}],
    ],
)
def test_document_or_empty_tool_result_retains_tool_association(
    deepseek_provider: DeepSeekProvider,
    tool_content: list[JsonObject],
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "paper.pdf"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": tool_content,
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    tool_result = array_payload(body["messages"])[1]
    assert object_payload(tool_result)["role"] == "tool"
    assert object_payload(tool_result)["tool_call_id"] == "t1"
    assert object_payload(tool_result)["content"] == ""


def test_document_only_message_is_rejected(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_pdf"},
                        },
                    ],
                },
            ],
        }
    )

    with pytest.raises(InvalidRequestError, match="latest user"):
        deepseek_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://images.example.test/shot.png",
                },
            }
        ],
    ],
)
def test_no_warning_when_no_documents(
    deepseek_provider: DeepSeekProvider,
    caplog: pytest.LogCaptureFixture,
    content: list[JsonObject] | str,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": content}],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )

    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_parallel_tool_results_close_before_image_content(
    deepseek_provider: DeepSeekProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-flash",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                        {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "content": "text result",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "YQ==",
                                    },
                                }
                            ],
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(request, reasoning=REASONING_ON)

    messages = body["messages"]
    assert [object_payload(message)["role"] for message in array_payload(messages)] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
    ]
    assert [
        object_payload(message)["tool_call_id"]
        for message in array_payload(messages)[1:3]
    ] == ["t1", "t2"]
    assert (
        object_payload(array_payload(messages)[1])["content"]
        == "[Image-bearing tool output follows in user content.]"
    )
    assert object_payload(array_payload(messages)[2])["content"] == "text result"
    assert is_synthetic_chat_tool_turn_boundary(array_payload(messages)[3])
    assert object_payload(array_payload(messages)[3])["reasoning_content"] == ""
    assert object_payload(array_payload(messages)[4])["content"] == [
        {"type": "text", "text": 'Image-bearing output for tool call "t1":'},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
    ]


@pytest.mark.parametrize("inside_tool_result", [False, True])
def test_mixed_attachment_content_omits_only_document(
    deepseek_provider: DeepSeekProvider, inside_tool_result: bool
) -> None:
    content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "YQ=="},
        },
        {"type": "document", "source": {"type": "file", "file_id": "private_pdf"}},
    ]
    messages = [{"role": "user", "content": content}]
    if inside_tool_result:
        messages: list[JsonObject] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": content}
                ],
            },
        ]
    request = MessagesRequest.model_validate(
        {"model": "deepseek-flash", "messages": messages}
    )
    original = request.model_dump()

    body = deepseek_provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    images = [
        part
        for message in array_payload(body["messages"])
        if isinstance(object_payload(message).get("content"), list)
        for part in array_payload(object_payload(message)["content"])
        if object_payload(part).get("type") == "image_url"
    ]
    assert images == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}
    ]
    assert "private_pdf" not in json.dumps(body["messages"])
    assert request.model_dump() == original


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            {
                "type": "base64",
                "media_type": "image/png",
                "data": "private_invalid_base64",
            },
            "Image data is not valid base64",
        ),
        (
            {"type": "file", "file_id": "private_file"},
            "cannot cross this protocol boundary",
        ),
        ({"type": "url", "url": ""}, "Image URL must be a non-empty string"),
    ],
)
def test_rejects_unportable_image_without_exposing_source(
    deepseek_provider: DeepSeekProvider, source: JsonObject, error: str
) -> None:
    request = MessagesRequest(
        model="deepseek-flash",
        messages=[
            Message(
                role="user", content=[ContentBlockImage(type="image", source=source)]
            )
        ],
    )

    with pytest.raises(InvalidRequestError, match=error) as exc_info:
        deepseek_provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert "private_" not in str(exc_info.value)


def test_shared_conversion_rejects_assistant_image(
    deepseek_provider: DeepSeekProvider,
) -> None:
    image = {
        "type": "image",
        "source": {"type": "url", "url": "https://images.example.test/shot.png"},
    }
    request = MessagesRequest.model_validate(
        {
            "model": "deepseek-flash",
            "messages": [{"role": "assistant", "content": [image]}],
        }
    )

    with pytest.raises(InvalidRequestError, match="Assistant image blocks"):
        deepseek_provider._chat._build_request_body(request, reasoning=REASONING_OFF)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
async def test_upstream_image_rejection_replaces_image_with_a_note(
    deepseek_provider: DeepSeekProvider, wire_api: str
) -> None:
    if wire_api == "messages":
        request = MessagesRequest(
            model="deepseek-flash",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "YQ==",
                            },
                        )
                    ],
                )
            ],
        )
        stream = deepseek_provider.stream_messages(request, reasoning=REASONING_OFF)
    else:
        responses_request = OpenAIResponsesRequest.model_validate(
            {
                "model": "deepseek-flash",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,YQ==",
                            }
                        ],
                    }
                ],
            }
        )
        stream = deepseek_provider.stream_responses(
            responses_request, reasoning=REASONING_OFF
        )
    error = BadRequestError(
        "This model does not support image inputs",
        response=httpx2.Response(
            400,
            request=httpx2.Request("POST", "https://provider.invalid/chat/completions"),
        ),
        body={
            "error": {
                "type": "invalid_request_error",
                "message": "This model does not support image inputs",
            }
        },
    )

    async def answered():
        for delta, finish_reason in (("I cannot see the image.", None), (None, "stop")):
            yield sdk_chat_chunk(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=delta, reasoning_content=None, tool_calls=None
                            ),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=None,
                )
            )

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}
    note = {
        "type": "text",
        "text": (
            "[image omitted: this model cannot accept image input. "
            "Tell the user the image was not seen.]"
        ),
    }
    try:
        with patch.object(
            deepseek_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[error, SDKStreamDouble(answered())],
        ) as create:
            text = "".join([frame async for chunk in stream for frame in chunk.frames])

        assert "I cannot see the image." in text
        assert [call.kwargs["messages"] for call in create.call_args_list] == [
            [{"role": "user", "content": [image]}],
            [{"role": "user", "content": [note]}],
        ]
    finally:
        await deepseek_provider.cleanup()
