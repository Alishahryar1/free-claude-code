"""Tests for api/models/responses.py Pydantic response models."""

import pytest

from api.models.responses import MessagesResponse, Usage, TokenCountResponse
from api.models.anthropic import (
    ContentBlockText,
    ContentBlockToolUse,
    ContentBlockThinking,
)


class TestUsage:
    """Tests for Usage model."""

    def test_required_fields(self):
        usage = Usage(input_tokens=10, output_tokens=20)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_cache_defaults_zero(self):
        usage = Usage(input_tokens=1, output_tokens=2)
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_cache_fields_set(self):
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=3,
        )
        assert usage.cache_creation_input_tokens == 5
        assert usage.cache_read_input_tokens == 3

    def test_serialization(self):
        usage = Usage(input_tokens=10, output_tokens=20)
        data = usage.model_dump()
        assert data == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }


class TestTokenCountResponse:
    """Tests for TokenCountResponse model."""

    def test_basic(self):
        resp = TokenCountResponse(input_tokens=42)
        assert resp.input_tokens == 42

    def test_serialization(self):
        resp = TokenCountResponse(input_tokens=100)
        data = resp.model_dump()
        assert data == {"input_tokens": 100}


class TestMessagesResponse:
    """Tests for MessagesResponse model."""

    def test_minimum_fields(self):
        resp = MessagesResponse(
            id="msg_001",
            model="test-model",
            content=[ContentBlockText(type="text", text="Hello")],
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        assert resp.id == "msg_001"
        assert resp.model == "test-model"
        assert resp.role == "assistant"
        assert resp.type == "message"
        assert resp.stop_reason is None
        assert resp.stop_sequence is None

    def test_with_text_content(self):
        resp = MessagesResponse(
            id="msg_002",
            model="model",
            content=[ContentBlockText(type="text", text="response")],
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        assert len(resp.content) == 1
        assert resp.content[0].type == "text"
        assert resp.content[0].text == "response"

    def test_with_tool_use_content(self):
        resp = MessagesResponse(
            id="msg_003",
            model="model",
            content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="tool_1",
                    name="Read",
                    input={"path": "test.py"},
                )
            ],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        )
        assert resp.content[0].type == "tool_use"
        assert resp.content[0].name == "Read"
        assert resp.stop_reason == "tool_use"

    def test_with_thinking_content(self):
        resp = MessagesResponse(
            id="msg_004",
            model="model",
            content=[
                ContentBlockThinking(type="thinking", thinking="Let me reason..."),
                ContentBlockText(type="text", text="Answer"),
            ],
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        assert len(resp.content) == 2
        assert resp.content[0].type == "thinking"
        assert resp.content[0].thinking == "Let me reason..."
        assert resp.content[1].type == "text"

    def test_with_all_content_types(self):
        resp = MessagesResponse(
            id="msg_005",
            model="model",
            content=[
                ContentBlockThinking(type="thinking", thinking="hmm"),
                ContentBlockText(type="text", text="result"),
                ContentBlockToolUse(
                    type="tool_use", id="t1", name="Bash", input={"command": "ls"}
                ),
            ],
            usage=Usage(input_tokens=10, output_tokens=20),
            stop_reason="tool_use",
        )
        assert len(resp.content) == 3

    def test_with_dict_content(self):
        """Dict content (unknown block type) should be accepted."""
        resp = MessagesResponse(
            id="msg_006",
            model="model",
            content=[{"type": "custom", "data": "value"}],
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        assert resp.content[0]["type"] == "custom"

    def test_stop_reason_values(self):
        """All valid stop_reason values should be accepted."""
        for reason in ["end_turn", "max_tokens", "stop_sequence", "tool_use"]:
            resp = MessagesResponse(
                id="msg",
                model="model",
                content=[ContentBlockText(type="text", text="x")],
                usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason=reason,
            )
            assert resp.stop_reason == reason

    def test_serialization_round_trip(self):
        resp = MessagesResponse(
            id="msg_rt",
            model="model-v1",
            content=[ContentBlockText(type="text", text="hello")],
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )
        data = resp.model_dump()
        restored = MessagesResponse(**data)
        assert restored.id == resp.id
        assert restored.model == resp.model
        assert restored.stop_reason == resp.stop_reason

    def test_empty_content_list(self):
        resp = MessagesResponse(
            id="msg_empty",
            model="model",
            content=[],
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        assert resp.content == []
