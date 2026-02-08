"""Tests for providers/nvidia_nim/response.py response conversion."""

import pytest
from unittest.mock import MagicMock

from providers.nvidia_nim.response import convert_response


def _make_response(
    content="Hello",
    finish_reason="stop",
    tool_calls=None,
    reasoning_content=None,
    reasoning_details=None,
    usage=None,
):
    """Helper to build a minimal OpenAI-format response dict."""
    message = {"content": content, "role": "assistant"}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if reasoning_details is not None:
        message["reasoning_details"] = reasoning_details

    return {
        "id": "chatcmpl-test",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _make_request(model="test-model"):
    req = MagicMock()
    req.model = model
    return req


class TestConvertResponse:
    """Tests for convert_response function."""

    def test_simple_text_response(self):
        """Simple text content is preserved."""
        resp = _make_response(content="Hello world")
        result = convert_response(resp, _make_request())
        assert result["content"] == [{"type": "text", "text": "Hello world"}]
        assert result["stop_reason"] == "end_turn"

    def test_empty_content_gets_space(self):
        """Empty content produces a single space text block."""
        resp = _make_response(content="")
        result = convert_response(resp, _make_request())
        assert result["content"] == [{"type": "text", "text": " "}]

    def test_none_content_gets_space(self):
        """None content produces a single space text block."""
        resp = _make_response(content=None)
        result = convert_response(resp, _make_request())
        assert result["content"] == [{"type": "text", "text": " "}]

    def test_reasoning_content_field(self):
        """reasoning_content field is extracted as thinking block."""
        resp = _make_response(content="Answer", reasoning_content="I need to think...")
        result = convert_response(resp, _make_request())
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        assert "text" in types
        thinking = [b for b in result["content"] if b["type"] == "thinking"]
        assert thinking[0]["thinking"] == "I need to think..."

    def test_reasoning_details_list(self):
        """reasoning_details list is joined into thinking block."""
        resp = _make_response(
            content="Answer",
            reasoning_details=[
                {"text": "Step 1"},
                {"text": "Step 2"},
            ],
        )
        result = convert_response(resp, _make_request())
        thinking = [b for b in result["content"] if b["type"] == "thinking"]
        assert len(thinking) == 1
        assert "Step 1" in thinking[0]["thinking"]
        assert "Step 2" in thinking[0]["thinking"]

    def test_content_with_think_tags(self):
        """Think tags in content string are extracted when no reasoning field."""
        resp = _make_response(content="<think>reasoning here</think>The answer is 42")
        result = convert_response(resp, _make_request())
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert any("42" in b["text"] for b in text_blocks)

    def test_think_tags_skipped_when_reasoning_exists(self):
        """When reasoning_content exists, think tags in content are NOT re-extracted."""
        resp = _make_response(
            content="<think>duplicate</think>Answer",
            reasoning_content="Real reasoning",
        )
        result = convert_response(resp, _make_request())
        thinking = [b for b in result["content"] if b["type"] == "thinking"]
        # Only the reasoning_content should be in thinking, not duplicate extraction
        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "Real reasoning"

    def test_content_as_list(self):
        """Content as list of dicts is preserved."""
        resp = _make_response(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        )
        result = convert_response(resp, _make_request())
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 2

    def test_tool_call_valid_json(self):
        """Tool calls with valid JSON arguments are parsed."""
        resp = _make_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ],
        )
        result = convert_response(resp, _make_request())
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["input"] == {"query": "test"}
        assert tool_blocks[0]["name"] == "search"

    def test_tool_call_invalid_json_fallback(self):
        """Tool call with non-JSON arguments falls back to raw value."""
        resp = _make_response(
            content="",
            tool_calls=[
                {
                    "id": "call_2",
                    "function": {
                        "name": "test",
                        "arguments": "not valid json {",
                    },
                }
            ],
        )
        result = convert_response(resp, _make_request())
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["input"] == "not valid json {"

    def test_usage_mapping(self):
        """Usage tokens are mapped from OpenAI to Anthropic format."""
        resp = _make_response(usage={"prompt_tokens": 100, "completion_tokens": 50})
        result = convert_response(resp, _make_request())
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["cache_creation_input_tokens"] == 0
        assert result["usage"]["cache_read_input_tokens"] == 0

    def test_missing_usage(self):
        """Missing usage defaults to zeros."""
        resp = _make_response()
        resp.pop("usage")
        result = convert_response(resp, _make_request())
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    @pytest.mark.parametrize(
        "finish_reason,expected_stop",
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
            (None, "end_turn"),
        ],
        ids=["stop", "length", "tool_calls", "content_filter", "none"],
    )
    def test_stop_reason_mapping(self, finish_reason, expected_stop):
        """Finish reasons map correctly to Anthropic stop_reasons."""
        resp = _make_response(finish_reason=finish_reason)
        result = convert_response(resp, _make_request())
        assert result["stop_reason"] == expected_stop

    def test_model_from_request(self):
        """Response model comes from original request, not provider response."""
        resp = _make_response()
        result = convert_response(resp, _make_request(model="claude-3"))
        assert result["model"] == "claude-3"
