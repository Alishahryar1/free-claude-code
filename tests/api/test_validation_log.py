"""Tests for validation log summaries (metadata only)."""

from free_claude_code.api.validation_log import summarize_request_validation_body


def test_summarize_lists_block_metadata_without_echoing_string_content() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": "secret user phrase",
            }
        ],
        "tools": [{"name": "web_search", "type": "web_search_20250305"}],
    }
    summary, tool_names = summarize_request_validation_body(body)
    assert summary == [
        {
            "role": "user",
            "content_kind": "str",
            "content_length": 18,
        }
    ]
    assert tool_names == ["web_search"]
    blob = repr(summary) + repr(tool_names)
    assert "secret" not in blob


def test_summarize_handles_non_dict_messages_and_missing_tools() -> None:
    body = {"messages": ["not_a_dict"]}
    summary, tool_names = summarize_request_validation_body(body)
    assert summary == [{"message_kind": "str"}]
    assert tool_names == []


def test_summarize_gemini_contents_without_echoing_part_text() -> None:
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "secret question"}]},
            {
                "role": "model",
                "parts": [{"thought": True, "text": "private reasoning"}],
            },
        ]
    }
    summary, tool_names = summarize_request_validation_body(body)
    assert summary == [
        {
            "role": "user",
            "content_kind": "list",
            "block_types": ["dict"],
            "block_keys": [["text"]],
        },
        {
            "role": "model",
            "content_kind": "list",
            "block_types": ["dict"],
            "block_keys": [["text", "thought"]],
        },
    ]
    assert tool_names == []
    blob = repr(summary) + repr(tool_names)
    assert "secret" not in blob and "private" not in blob
