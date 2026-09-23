import pytest
from pydantic import ValidationError

from free_claude_code.core.anthropic.conversion import (
    OpenAIConversionError,
    build_base_request_body,
)
from free_claude_code.core.anthropic.models import (
    ContentBlockDocument,
    ContentBlockWebFetchToolResult,
    Message,
    MessagesRequest,
    TokenCountRequest,
)


def test_messages_request_parses_without_model_mapping_side_effects():
    request = MessagesRequest(
        model="claude-3-opus",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )

    assert request.model == "claude-3-opus"
    assert request.stream is False


def test_messages_request_rejects_null_stream() -> None:
    with pytest.raises(ValidationError):
        MessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": None,
            }
        )


def test_messages_request_preserves_system_role_message_order():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "second"},
            ],
        }
    )

    assert [message.role for message in request.messages] == [
        "user",
        "system",
        "user",
    ]
    assert request.messages[1].content == "system prompt"
    assert request.system is None


def test_messages_request_keeps_top_level_and_inline_system_content_distinct():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "system": "existing system",
            "messages": [
                {"role": "system", "content": "message system"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert request.system == "existing system"
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == "message system"


def test_messages_request_preserves_inline_system_block_metadata():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "system": [
                {
                    "type": "text",
                    "text": "existing system",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "message system",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert len(request.messages) == 2
    assert isinstance(request.system, list)
    assert [block.text for block in request.system] == ["existing system"]
    assert request.system[0].model_dump()["cache_control"] == {"type": "ephemeral"}
    inline_content = request.messages[0].content
    assert isinstance(inline_content, list)
    assert inline_content[0].model_dump() == {
        "type": "text",
        "text": "message system",
        "cache_control": {"type": "ephemeral"},
    }


def test_messages_request_ignores_internal_routing_fields_when_supplied():
    request = MessagesRequest.model_validate(
        {
            "model": "target-model",
            "original_model": "claude-3-opus",
            "resolved_provider_model": "nvidia_nim/target-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert request.model == "target-model"
    assert "original_model" not in request.model_dump()
    assert "resolved_provider_model" not in request.model_dump()


def test_token_count_request_parses_without_model_mapping_side_effects():
    request = TokenCountRequest(
        model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
    )

    assert request.model == "claude-3-sonnet"


def test_token_count_request_preserves_system_role_messages():
    request = TokenCountRequest.model_validate(
        {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "system", "content": "counting system"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == "counting system"
    assert request.system is None


def test_messages_request_preserves_thinking_signature():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "signed thought",
                            "signature": "sig_123",
                        }
                    ],
                }
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["messages"][0]["content"][0]["signature"] == "sig_123"


def test_messages_request_preserves_native_thinking_budget():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "think hard"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["thinking"]["type"] == "enabled"
    assert dumped["thinking"]["budget_tokens"] == 4096


def test_messages_request_accepts_adaptive_thinking_type():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive"},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["thinking"]["type"] == "adaptive"


def test_messages_request_accepts_anthropic_server_tool_without_input_schema():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["tools"] == [{"name": "web_search", "type": "web_search_20250305"}]


def test_messages_request_accepts_redacted_thinking_blocks():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "opaque"}],
                }
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["messages"][0]["content"][0] == {
        "type": "redacted_thinking",
        "data": "opaque",
    }


def test_document_and_web_fetch_blocks_preserve_protocol_extensions() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "data": "encoded"},
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "web_fetch_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": {"url": "https://example.com"},
                            "provider_extension": True,
                        },
                    ],
                }
            ],
        }
    )

    content = request.messages[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], ContentBlockDocument)
    assert content[0].model_dump()["cache_control"] == {"type": "ephemeral"}
    assert isinstance(content[1], ContentBlockWebFetchToolResult)
    assert content[1].model_dump()["provider_extension"] is True


def test_content_block_descriptions_remain_in_the_public_schema() -> None:
    definitions = MessagesRequest.model_json_schema()["$defs"]

    assert definitions["ContentBlockDocument"]["description"] == (
        "Anthropic document block (e.g. PDF files via the Files API)."
    )
    assert definitions["ContentBlockServerToolUse"]["description"] == (
        "Anthropic server-side tool invocation (e.g. ``web_search``, ``web_fetch``)."
    )


def test_messages_request_dump_preserves_public_defaults_and_excludes_internal_fields() -> (
    None
):
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive"},
            "original_model": "original",
            "resolved_provider_model": "provider/model",
            "betas": ["feature-beta"],
            "client_extension": {"enabled": True},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "thinking": {"enabled": True, "type": "adaptive"},
        "client_extension": {"enabled": True},
    }


def test_token_count_request_accepts_extras_but_excludes_internal_fields() -> None:
    request = TokenCountRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "original_model": "original",
            "resolved_provider_model": "provider/model",
            "betas": ["feature-beta"],
            "client_extension": "accepted",
        }
    )

    assert request.model_extra == {"client_extension": "accepted"}
    assert request.model_dump(exclude_none=True) == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "client_extension": "accepted",
    }


def test_openai_conversion_rejects_unknown_top_level_anthropic_extensions() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "client_extension": True,
        }
    )

    with pytest.raises(OpenAIConversionError, match="client_extension"):
        build_base_request_body(request)
