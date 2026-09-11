import json
import unittest
from types import SimpleNamespace

from free_claude_code.core.anthropic.streaming import ToolSchema
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.providers.openai_chat.profiles import OPENAI_CHAT_PROFILES
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
)
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
)
from free_claude_code.providers.openai_chat.transport import (
    _OpenAIChatStreamAssembler,
)

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
    },
    "required": ["file_path"],
    "additionalProperties": False,
}


def identity_tool_names():
    return OpenAIToolNameCodec(
        _original_to_alias={"Read": "Read"},
        _alias_to_original={"Read": "Read"},
        _unchanged_names=frozenset({"Read"}),
    )


def new_output():
    return AnthropicChatStreamOutput(
        message_id="msg_test",
        model="qwen2.5-coder:7b-alfa-6144",
        input_tokens=10,
    )


def new_assembler():
    tool_names = identity_tool_names()

    return _OpenAIChatStreamAssembler(
        output=new_output(),
        profile=OPENAI_CHAT_PROFILES["ollama"],
        provider_name="ollama",
        output_reasoning=False,
        tool_names=tool_names,
        tool_schemas={
            "Read": ToolSchema(
                name="Read",
                input_schema=READ_SCHEMA,
            )
        },
        tool_choice_enabled=True,
        tool_calls=OpenAIToolCallAssembler(
            reserved_tool_ids=(),
        ),
        extra_reasoning_events=lambda delta, output: iter(()),
    )


def chunk(
    *,
    content=None,
    tool_calls=None,
    finish_reason=None,
):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
    )

    choice = SimpleNamespace(
        delta=delta,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        choices=[choice],
        usage=None,
    )


def decode_sse(events):
    decoded = []

    for event in events:
        for line in event.splitlines():
            if not line.startswith("data: "):
                continue

            payload = line[len("data: ") :]

            if payload == "[DONE]":
                continue

            decoded.append(json.loads(payload))

    return decoded


class TerminalJsonAssemblerTests(unittest.TestCase):
    def test_terminal_json_becomes_anthropic_tool_use(self):
        assembler = new_assembler()

        events = list(assembler.start_events())

        raw = '{"name":"Read","arguments":{"file_path":".claude/settings.json"}}'

        events += list(assembler.feed(chunk(content=raw)))

        events += list(
            assembler.feed(
                chunk(
                    content=None,
                    finish_reason="stop",
                )
            )
        )

        events += list(assembler.finish_upstream())

        decoded = decode_sse(events)

        self.assertTrue(assembler.output.has_emitted_tool_block())

        self.assertEqual(
            assembler.output.accumulated_text,
            "",
        )

        serialized = json.dumps(decoded)

        self.assertIn('"type": "tool_use"', serialized)
        self.assertIn('"name": "Read"', serialized)
        self.assertIn(
            ".claude/settings.json",
            serialized,
        )

    def test_invalid_terminal_json_remains_text(self):
        assembler = new_assembler()

        events = list(assembler.start_events())

        raw = '{"name":"Read","arguments":{"wrong_key":"x"}}'

        events += list(assembler.feed(chunk(content=raw)))

        events += list(
            assembler.feed(
                chunk(
                    content=None,
                    finish_reason="stop",
                )
            )
        )

        events += list(assembler.finish_upstream())

        self.assertFalse(assembler.output.has_emitted_tool_block())

        self.assertEqual(
            assembler.output.accumulated_text,
            raw,
        )

    def test_native_tool_call_has_priority_over_terminal_json_candidate(self):
        assembler = new_assembler()

        events = list(assembler.start_events())

        held = '{"name":"Read","arguments":'

        events += list(assembler.feed(chunk(content=held)))

        native_call = SimpleNamespace(
            index=0,
            id="call_native_1",
            function=SimpleNamespace(
                name="Read",
                arguments='{"file_path":"native.txt"}',
            ),
        )

        events += list(
            assembler.feed(
                chunk(
                    content=None,
                    tool_calls=[native_call],
                )
            )
        )

        events += list(
            assembler.feed(
                chunk(
                    content=None,
                    finish_reason="tool_calls",
                )
            )
        )

        events += list(assembler.finish_upstream())

        decoded = decode_sse(events)
        serialized = json.dumps(decoded)

        self.assertTrue(assembler.output.has_emitted_tool_block())

        self.assertIn("native.txt", serialized)

        # The incomplete JSON candidate must be released as text,
        # never recovered as a second tool.
        self.assertIn(held, assembler.output.accumulated_text)

        tool_starts = [
            item
            for item in decoded
            if item.get("type") == "content_block_start"
            and item.get("content_block", {}).get("type") == "tool_use"
        ]

        self.assertEqual(len(tool_starts), 1)


if __name__ == "__main__":
    unittest.main()
