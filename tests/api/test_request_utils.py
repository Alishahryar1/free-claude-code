"""Tests for token counting helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.core.anthropic import get_token_count
from free_claude_code.core.anthropic.models import (
    Message,
    SystemContent,
)


class TestGetTokenCount:
    """Tests for get_token_count function."""

    def test_empty_messages(self):
        """Test token count with empty messages."""
        count = get_token_count([])
        assert count >= 1  # Returns max(1, tokens)

    def test_simple_message(self):
        """Test token count with simple text message."""
        msg = MagicMock()
        msg.content = "Hello world"

        count = get_token_count([msg])
        assert count > 0
        # "Hello world" is ~2-3 tokens plus overhead
        assert count >= 3

    def test_special_token_text_is_counted_as_plain_text(self):
        """Tiktoken special-token strings should not break token estimates."""
        msg = MagicMock()
        msg.content = "<|endoftext|>"

        count = get_token_count([msg], system="<|endoftext|>")
        assert count > 0

    def test_message_with_system_prompt(self):
        """Test token count includes system prompt."""
        msg = MagicMock()
        msg.content = "Hello"

        count = get_token_count([msg], system="You are a helpful assistant")
        assert count > 0

    def test_message_with_list_content(self):
        """Test token count with list content blocks."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"

        msg = MagicMock()
        msg.content = [text_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_thinking_block(self):
        """Test token count includes thinking blocks."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me think about this..."

        msg = MagicMock()
        msg.content = [thinking_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_tool_use(self):
        """Test token count includes tool use blocks."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search"
        tool_block.input = {"query": "test"}

        msg = MagicMock()
        msg.content = [tool_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_tool_result(self):
        """Test token count includes tool result blocks."""
        result_block = MagicMock()
        result_block.type = "tool_result"
        result_block.content = "Search results here"

        msg = MagicMock()
        msg.content = [result_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_tools(self):
        """Test token count includes tool definitions."""
        msg = MagicMock()
        msg.content = "Use the search tool"

        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search for information"
        tool.input_schema = {"type": "object", "properties": {}}

        count = get_token_count([msg], tools=[tool])
        assert count > 0

    def test_system_as_list(self):
        """Test token count with system as list of blocks."""
        msg = MagicMock()
        msg.content = "Hello"

        block = MagicMock()
        block.text = "System prompt"

        count = get_token_count([msg], system=[block])
        assert count > 0

    def test_tool_result_with_dict_content(self):
        """Test token count with tool result containing dict content."""
        result_block = MagicMock()
        result_block.type = "tool_result"
        result_block.content = {"result": "data"}

        msg = MagicMock()
        msg.content = [result_block]

        count = get_token_count([msg])
        assert count > 0

    def test_multiple_messages_overhead(self):
        """Test that multiple messages include overhead."""
        msg1 = MagicMock()
        msg1.content = "Hi"

        msg2 = MagicMock()
        msg2.content = "Hello"

        count_single = get_token_count([msg1])
        count_double = get_token_count([msg1, msg2])

        # Double message should have more tokens (including overhead)
        assert count_double > count_single

    def test_inline_system_message_contributes_to_token_count(self):
        """Mid-conversation system content remains part of the counted transcript."""
        user_message = Message(role="user", content="Hello")
        inline_system = Message(role="system", content="New instructions")

        without_inline_system = get_token_count([user_message])
        with_inline_system = get_token_count([user_message, inline_system])

        assert with_inline_system > without_inline_system

    def test_per_message_overhead_four_tokens(self):
        """Per-message overhead is 4 tokens (was 3)."""
        msg = MagicMock()
        msg.content = "x"  # Minimal content
        count = get_token_count([msg])
        # 1 msg * 4 overhead + content tokens
        assert count >= 5

    def test_system_overhead_added(self):
        """System prompt adds ~4 tokens overhead."""
        msg = MagicMock()
        msg.content = "Hi"
        count_no_sys = get_token_count([msg])
        count_with_sys = get_token_count([msg], system="You are helpful")
        assert count_with_sys >= count_no_sys + 4

    def test_system_as_typed_content_blocks(self):
        """Typed system content blocks are counted."""
        msg = MagicMock()
        msg.content = "Hi"
        count_no_sys = get_token_count([msg])
        system_blocks = [
            SystemContent(type="text", text="System prompt from typed block")
        ]
        count_with_system = get_token_count([msg], system=system_blocks)
        assert count_with_system > count_no_sys

    def test_tool_use_includes_id(self):
        """Tool use blocks count id field."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search"
        tool_block.input = {"q": "test"}
        tool_block.id = "call_abc123"
        msg = MagicMock()
        msg.content = [tool_block]
        count = get_token_count([msg])
        assert count > 0

    def test_tool_result_includes_tool_use_id(self):
        """Tool result blocks count tool_use_id field."""
        result_block = MagicMock()
        result_block.type = "tool_result"
        result_block.content = "ok"
        result_block.tool_use_id = "call_xyz"
        msg = MagicMock()
        msg.content = [result_block]
        count = get_token_count([msg])
        assert count > 0

    def test_unrecognized_block_type_fallback(self):
        """Unrecognized block types are tokenized via json.dumps fallback."""
        unknown_block = {"type": "custom", "spec": "data"}
        msg = MagicMock()
        msg.content = [unknown_block]
        count = get_token_count([msg])
        assert count > 0

    def test_message_with_image_block(self):
        """Test token count includes image blocks."""
        image_block = MagicMock()
        image_block.type = "image"
        image_block.source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "x" * 3000,
        }
        msg = MagicMock()
        msg.content = [image_block]
        count = get_token_count([msg])
        assert count >= 85

    def test_image_block_with_dict_source(self):
        """Image block with dict-style source is counted."""
        image_block = {"type": "image", "source": {"data": "a" * 10000}}
        msg = MagicMock()
        msg.content = [image_block]
        count = get_token_count([msg])
        assert count >= 85

    def test_image_in_tool_result_costs_the_same_as_a_top_level_image(self):
        """A tool-returned image (Read on a PNG, an MCP screenshot) is priced as an image.

        Serializing it with the rest of the tool result counted its base64 payload as
        text: a 300 KB screenshot came out near 215k tokens instead of about 100.
        """
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "A" * 300_000,
            },
        }
        text = {"type": "text", "text": "Screenshot captured"}

        def user_message(*blocks):
            msg = MagicMock()
            msg.content = list(blocks)
            return msg

        def tool_result(*parts):
            return user_message(
                {"type": "tool_result", "tool_use_id": "t1", "content": list(parts)}
            )

        def image_cost(wrap):
            return get_token_count([wrap(text, image)]) - get_token_count([wrap(text)])

        assert image_cost(tool_result) == image_cost(user_message)
        assert image_cost(tool_result) < 1_000

    @pytest.mark.parametrize("data", [12345, ["a"], {"x": 1}])
    def test_tool_result_image_with_non_string_data_gets_the_flat_cost(self, data):
        """Tool-result content is not validated, so image data can be any JSON value.

        A number used to reach ``len()`` and turn count_tokens into an HTTP 500.
        """
        image = {"type": "image", "source": {"type": "base64", "data": data}}
        msg = MagicMock()
        msg.content = [{"type": "tool_result", "tool_use_id": "t1", "content": [image]}]
        no_data = MagicMock()
        no_data.content = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "image", "source": {"type": "base64"}}],
            }
        ]

        assert get_token_count([msg]) == get_token_count([no_data])

    @pytest.mark.parametrize(
        "parts",
        [
            [{"type": "text", "text": "line one"}, {"type": "text", "text": "two"}],
            [],
        ],
    )
    def test_image_free_tool_result_list_is_counted_as_before(self, parts):
        """Only images leave the serialized payload; image-free results are unchanged."""
        msg = MagicMock()
        msg.content = [{"type": "tool_result", "tool_use_id": "t1", "content": parts}]

        with patch(
            "free_claude_code.core.anthropic.tokens.estimate_text_tokens",
            side_effect=len,
        ):
            count = get_token_count([msg])

        assert count == len(json.dumps(parts)) + len("t1") + 8 + 4

    def test_known_payload_structural_overhead(self):
        """Protocol overhead remains separate from plain-text estimation."""
        system_text = "You are a helpful assistant."
        user_text = "Hello, how are you?"
        msg = MagicMock()
        msg.content = user_text
        with patch(
            "free_claude_code.core.anthropic.tokens.estimate_text_tokens",
            side_effect=len,
        ):
            count = get_token_count([msg], system=system_text)

        assert count == len(system_text) + len(user_text) + 4 + 4


# --- Parametrized Edge Case Tests ---
