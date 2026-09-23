"""Anthropic request parsing and public-field serialization."""

from free_claude_code.core.anthropic import dump_messages_request
from free_claude_code.core.anthropic.models import (
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockWebSearchToolResult,
    Message,
    MessagesRequest,
)


def test_dump_preserves_public_fields_and_nested_extensions() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 20,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "context_management": {"edits": [{"type": "clear"}]},
            "output_config": {"some": "hint"},
        }
    )

    body = dump_messages_request(request)

    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["context_management"] == {"edits": [{"type": "clear"}]}
    assert body["output_config"] == {"some": "hint"}


def test_dump_excludes_unknown_client_hints_and_fcc_routing_state() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "reasoning_effort": "none",
            "unknown_client_hint": {"mode": "local"},
        }
    )
    request.original_model = "claude"
    request.resolved_provider_model = "upstream"

    body = dump_messages_request(request)

    assert "reasoning_effort" not in body
    assert "unknown_client_hint" not in body
    assert "original_model" not in body
    assert "resolved_provider_model" not in body


def test_pydantic_discriminator_still_distinguishes_blocks() -> None:
    message = Message.model_validate(
        {
            "role": "user",
            "content": [{"type": "text", "text": "a", "z": 1}],
        }
    )

    block = message.content[0]

    assert isinstance(block, ContentBlockText)
    assert block.model_dump()["z"] == 1


def test_server_tool_history_remains_valid_anthropic_input() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {"query": "q"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [],
                        },
                    ],
                }
            ],
        }
    )

    blocks = request.messages[0].content
    assert isinstance(blocks, list)
    assert isinstance(blocks[0], ContentBlockServerToolUse)
    assert isinstance(blocks[1], ContentBlockWebSearchToolResult)
