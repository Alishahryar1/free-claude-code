"""Tests for DeepSeek OpenAI-compatible Chat Completions provider."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import DEEPSEEK_DEFAULT_BASE
from free_claude_code.core.anthropic.models import (
    ContentBlockDocument,
    ContentBlockImage,
    Message,
    MessagesRequest,
    Tool,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.providers.deepseek import DeepSeekProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    reasoning_for,
)


@pytest.fixture
def deepseek_config():
    return make_provider_config(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config, admission=immediate_admission())


def test_default_base_url_alias():
    assert DEEPSEEK_DEFAULT_BASE == "https://api.deepseek.com"


def test_init(deepseek_config):
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_client:
        provider = DeepSeekProvider(deepseek_config, admission=immediate_admission())
    assert provider._api_key == "test_deepseek_key"
    assert provider._base_url == "https://api.deepseek.com"
    assert mock_client.called


def test_responses_request_uses_deepseek_chat_policy(deepseek_provider):
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
            "tool_choice": {"type": "function", "name": "Read"},
        }
    )

    translated = deepseek_provider._chat._build_responses_request_body(
        request, reasoning=REASONING_ON
    )

    assert translated.body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert translated.body["tool_choice"] == "auto"
    assert translated.body["extra_body"] == {"thinking": {"type": "enabled"}}


def test_responses_tool_history_keeps_deepseek_replayable_reasoning(
    deepseek_provider,
):
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

    assert translated.body["messages"][0]["reasoning_content"] == "Inspect first"
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
    deepseek_provider, usage, anthropic_expected, responses_cached
) -> None:
    assert (
        deepseek_provider._behavior.anthropic_usage_fields(usage) == anthropic_expected
    )
    assert deepseek_provider._behavior.cached_input_tokens(usage) == responses_cached


def test_build_request_body_openai_chat_shape(deepseek_provider):
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
    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100
    assert "stream_options" not in body


def test_build_request_body_default_max_tokens(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(deepseek_provider):
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
    assert body["extra_body"]["thinking"] == {"type": "enabled"}


def test_build_request_body_tool_list_keeps_thinking(deepseek_provider):
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

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["tools"][0]["function"]["name"] == "Read"


def test_build_request_body_tool_choice_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["tool_choice"] == "auto"


def test_build_request_body_forced_tool_choice_downgrades_to_auto(
    deepseek_provider,
):
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

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["tool_choice"] == "auto"


def test_build_request_body_encodes_reasoning_off():
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
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
    assert body["extra_body"]["thinking"] == {"type": "disabled"}
    assert "stream_options" not in body


def test_non_tool_thinking_survives_conversion(deepseek_provider):
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
    assert body["messages"][0] == {
        "role": "assistant",
        "content": "out",
        "reasoning_content": "plain",
    }


def test_redacted_thinking_is_kept_until_destination_preparation(deepseek_provider):
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
    assert body["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque"}
    ]
    assert request.model_dump()["messages"][0]["content"][0]["data"] == "opaque"


def test_tool_history_with_replayable_thinking_preserves_thinking(deepseek_provider):
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
    assistant = body["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "hidden"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"file_path": "x"}'
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "ok",
    }


def test_tool_history_with_unsigned_thinking_preserves_thinking(deepseek_provider):
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

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["reasoning_content"] == "plain"


def test_tool_history_without_thinking_keeps_selected_effort(deepseek_provider):
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
            "output_config": {"effort": "high", "format": "text"},
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "high"
    assert body["extra_body"].get("thinking") != {"type": "disabled"}
    assert body["messages"][0]["reasoning_content"] == ""
    assert "context_management" not in body
    assert "output_config" not in body
    assert body["tools"][0]["function"]["name"] == "Read"
    assert body["tool_choice"] == "auto"
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"
    assert body["messages"][1]["role"] == "tool"


def test_tool_history_with_empty_thinking_preserves_reasoning_state(deepseek_provider):
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

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["reasoning_content"] == ""
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"


def test_tool_history_with_empty_top_level_reasoning_preserves_reasoning_state(
    deepseek_provider,
):
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

    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["reasoning_content"] == ""
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"


def test_thinking_off_preserves_historical_reasoning():
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
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
    assert body["extra_body"]["thinking"] == {"type": "disabled"}
    assert body["messages"][0]["reasoning_content"] == "sec"


def test_thinking_off_still_replays_required_tool_reasoning():
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
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

    assert body["extra_body"]["thinking"] == {"type": "disabled"}
    assert body["messages"][0]["reasoning_content"] == "required"


def test_passthrough_tool_use_and_result(deepseek_provider):
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
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "n"
    assert body["messages"][1]["role"] == "tool"


def test_preflight_strips_user_image():
    """Image blocks are silently stripped (DeepSeek lacks vision); request must not fail."""
    request = MessagesRequest(
        model="m",
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
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    # Should not raise; image is stripped.
    provider.preflight_messages(request, reasoning=REASONING_ON)
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    content = body["messages"][0]["content"]
    assert "attachment omitted" in content.lower()
    assert "image or document inputs" in content.lower()


def test_vision_model_forwards_user_image():
    """Vision-capable DeepSeek models receive image blocks as OpenAI image_url parts."""
    request = MessagesRequest(
        model="deepseek-v4-flash-vision-exp",
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
                    ),
                    {"type": "text", "text": "Describe this image"},
                ],
            )
        ],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    # Must not raise on preflight (no InvalidRequestError for image blocks).
    provider.preflight_messages(request, reasoning=REASONING_ON)
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    image_parts = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "data:image/png;base64,YQ=="
    text_parts = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert text_parts[0]["text"] == "Describe this image"


def test_vision_model_strips_user_document():
    """Vision models still omit documents; conversion has no document parts."""
    request = MessagesRequest(
        model="deepseek-v4-flash-vision-exp",
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockDocument(
                        type="document",
                        source={
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "YQ==",
                        },
                    )
                ],
            )
        ],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    provider.preflight_messages(request, reasoning=REASONING_ON)
    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    content = body["messages"][0]["content"]
    assert content
    if isinstance(content, str):
        lowered = content.lower()
    else:
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        lowered = "\n".join(texts).lower()
        assert not any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        )
    assert "attachment omitted" in lowered
    assert "image or document inputs" in lowered


def test_preflight_rejects_mcp_servers():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        mcp_servers=[{"type": "url", "url": "https://x"}],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    with pytest.raises(InvalidRequestError, match="mcp_servers"):
        provider.preflight_messages(request)


def test_preflight_rejects_listed_server_tools_in_tools_list():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        tools=[Tool(name="web_search", type="web_search_20250305", input_schema={})],
    )
    provider = DeepSeekProvider(
        make_provider_config(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    with pytest.raises(InvalidRequestError, match="web_search"):
        provider.preflight_messages(request)


def test_preflight_preserves_completed_server_tool_history():
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
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    provider.preflight_messages(request)
    body = provider._chat._build_request_body(request)
    assert "[Earlier tool record]" in body["messages"][0]["content"]


def test_non_tool_top_level_reasoning_survives_conversion(deepseek_provider):
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
    assert body["messages"][0] == {
        "role": "assistant",
        "content": "hi",
        "reasoning_content": "r",
    }


def test_tool_call_top_level_reasoning_is_replayed(deepseek_provider):
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

    assert body["messages"][0]["reasoning_content"] == "required"


@pytest.mark.asyncio
async def test_wire_messages_keep_prefix_and_effort_when_old_reasoning_is_absent(
    deepseek_provider,
):
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

    def build(messages: list[dict]) -> dict:
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

    assert continued[: len(first_messages)] == first_messages
    assistant_messages = [
        message for message in first_messages if message["role"] == "assistant"
    ]
    assert assistant_messages[0]["reasoning_content"] == "ordinary reasoning"
    assert assistant_messages[1]["reasoning_content"] == "required tool reasoning"
    assert first_wire["thinking"] == {"type": "enabled"}
    assert continued_wire["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_stream_uses_chat_completions_and_maps_cache_usage(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )

    async def fake_stream():
        yield SimpleNamespace(
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
        yield SimpleNamespace(
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
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                completion_tokens=3,
                prompt_tokens=30,
                prompt_cache_hit_tokens=10,
                prompt_cache_miss_tokens=20,
            ),
        )

    create = AsyncMock(return_value=fake_stream())
    with patch.object(deepseek_provider._client.chat.completions, "create", create):
        chunks = [
            chunk
            async for chunk in deepseek_provider.stream_messages(
                request, input_tokens=7, request_id="r1"
            )
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
    assert "cache_creation_input_tokens" not in usage
    assert usage == {
        "input_tokens": 20,
        "output_tokens": 3,
        "cache_read_input_tokens": 10,
    }


@pytest.mark.asyncio
async def test_responses_stream_maps_deepseek_cache_usage(deepseek_provider):
    request = OpenAIResponsesRequest(model="m", input="hi")

    async def fake_stream():
        yield SimpleNamespace(
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
        yield SimpleNamespace(
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
        yield SimpleNamespace(
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

    create = AsyncMock(return_value=fake_stream())
    with patch.object(deepseek_provider._client.chat.completions, "create", create):
        chunks = [
            chunk
            async for chunk in deepseek_provider.stream_responses(
                request, input_tokens=7, request_id="r1"
            )
        ]

    parsed = parse_sse_text("".join(chunks))
    completed = next(
        event.data["response"]
        for event in parsed
        if event.event == "response.completed"
    )
    assert completed["usage"] == {
        "input_tokens": 30,
        "input_tokens_details": {"cached_tokens": 10},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 33,
    }


def test_preserves_extra_body_for_openai_chat_request(deepseek_provider):
    raw = {
        "model": "m",
        "max_tokens": 3,
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"note": 1},
    }
    r = MessagesRequest.model_validate(raw)
    body = deepseek_provider._chat._build_request_body(r, reasoning=reasoning_for(r))
    assert body["extra_body"] == {"note": 1}


def test_normalizes_tool_result_content_array_to_string(deepseek_provider):
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

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert "file1.txt" in tool_result["content"]
    assert "file2.txt" in tool_result["content"]


def test_strips_document_blocks_for_deepseek(deepseek_provider):
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

    assert body["messages"][0] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "PDF text extracted",
    }


def test_strips_image_blocks_for_deepseek(deepseek_provider):
    """Image blocks are stripped for DeepSeek since it doesn't support vision."""
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
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["messages"][0] == {"role": "user", "content": "describe this"}


def test_normalizes_tool_result_content_dict_to_string(deepseek_provider):
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

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert "status" in tool_result["content"]
    assert "success" in tool_result["content"]


def test_strips_image_block_inside_tool_result(deepseek_provider):
    """Image blocks nested inside tool_result.content are stripped, not rejected."""
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
                                        "data": "abc",
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

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    # After stripping + string-normalization, no base64/image marker survives.
    assert isinstance(tool_result["content"], str)
    assert "screenshot saved" in tool_result["content"]
    assert "base64" not in tool_result["content"]
    assert "abc" not in tool_result["content"]


def test_image_only_tool_result_replaced_with_placeholder(deepseek_provider):
    """A tool_result whose only inner block is an image becomes a placeholder string."""
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
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
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

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert tool_result["content"] != ""
    assert "attachment omitted" in tool_result["content"].lower()
    assert "image or document inputs" in tool_result["content"].lower()


def test_document_only_tool_result_replaced_with_generic_placeholder(
    deepseek_provider,
):
    """A document-only tool_result uses the generic attachment placeholder."""
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
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "file",
                                        "file_id": "file_pdf",
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

    tool_result = body["messages"][1]
    assert tool_result["role"] == "tool"
    assert isinstance(tool_result["content"], str)
    assert "attachment omitted" in tool_result["content"].lower()
    assert "document inputs" in tool_result["content"].lower()
    assert "image omitted" not in tool_result["content"].lower()


def test_image_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level image-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    content = body["messages"][0]["content"]
    assert "attachment omitted" in content.lower()
    assert "image or document inputs" in content.lower()


def test_document_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level document-only message remains non-empty after stripping."""
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

    body = deepseek_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    content = body["messages"][0]["content"]
    assert "attachment omitted" in content.lower()
    assert "document inputs" in content.lower()


def test_warns_when_stripping_attachment_blocks(deepseek_provider, caplog):
    """A warning is emitted when image/document blocks are dropped so users notice."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
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
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stripped unsupported attachment blocks" in r.message for r in warnings)


def test_no_warning_when_no_attachments(deepseek_provider, caplog):
    """No warning is emitted on plain text-only requests."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )

    assert not any(
        "stripped unsupported attachment blocks" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
