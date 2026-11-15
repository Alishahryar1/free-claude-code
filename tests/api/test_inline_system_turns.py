"""Tests for hoisting inline system turns into the top-level ``system`` field."""

from typing import Any

from free_claude_code.api.inline_system_turns import hoist_inline_system_turns
from free_claude_code.core.json_types import JsonObject


def _request(messages: Any, **extra: Any) -> JsonObject:
    return {"model": "claude-3-sonnet", "messages": messages, **extra}


class TestHoistInlineSystemTurns:
    def test_moves_inline_system_turns_to_the_top_level_system(self):
        body = _request(
            [
                {"role": "system", "content": "first reminder"},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "second reminder"},
            ]
        )

        hoisted = hoist_inline_system_turns(body)

        assert hoisted["system"] == "first reminder\n\nsecond reminder"
        assert hoisted["messages"] == [{"role": "user", "content": "Hi"}]

    def test_appends_to_an_existing_string_system(self):
        body = _request(
            [{"role": "system", "content": "reminder"}],
            system="base prompt",
        )

        hoisted = hoist_inline_system_turns(body)

        assert hoisted["system"] == "base prompt\n\nreminder"
        assert hoisted["messages"] == []

    def test_appends_to_an_existing_block_system(self):
        body = _request(
            [{"role": "system", "content": "reminder"}],
            system=[{"type": "text", "text": "base prompt"}],
        )

        hoisted = hoist_inline_system_turns(body)

        assert hoisted["system"] == [
            {"type": "text", "text": "base prompt"},
            {"type": "text", "text": "reminder"},
        ]

    def test_extracts_text_blocks_and_drops_block_metadata(self):
        body = _request(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "cached"},
                        {
                            "type": "text",
                            "text": "tail",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "image", "source": {"type": "base64", "data": "AA=="}},
                    ],
                    "clear_at": "next_user_message",
                },
                {"role": "user", "content": "Hi"},
            ]
        )

        hoisted = hoist_inline_system_turns(body)

        assert hoisted["system"] == "cached\n\ntail"
        assert hoisted["messages"] == [{"role": "user", "content": "Hi"}]

    def test_drops_system_turns_that_carry_no_text(self):
        body = _request(
            [
                {"role": "system", "content": ""},
                {"role": "system", "content": [{"type": "tool_addition"}]},
                {"role": "user", "content": "Hi"},
            ]
        )

        hoisted = hoist_inline_system_turns(body)

        assert "system" not in hoisted
        assert hoisted["messages"] == [{"role": "user", "content": "Hi"}]

    def test_leaves_requests_without_inline_system_untouched(self):
        body = _request([{"role": "user", "content": "Hi"}])

        assert hoist_inline_system_turns(body) is body

    def test_leaves_bodies_without_a_message_list_untouched(self):
        body: JsonObject = {"model": "claude-3-sonnet", "messages": "not-a-list"}

        assert hoist_inline_system_turns(body) is body

    def test_keeps_other_entries_in_order_and_does_not_mutate_the_input(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "reminder"},
            "not-a-message",
            {"role": "assistant", "content": "Hello"},
        ]
        body = _request(list(messages))

        hoisted = hoist_inline_system_turns(body)

        assert hoisted["messages"] == [
            {"role": "user", "content": "Hi"},
            "not-a-message",
            {"role": "assistant", "content": "Hello"},
        ]
        assert body["messages"] == messages
