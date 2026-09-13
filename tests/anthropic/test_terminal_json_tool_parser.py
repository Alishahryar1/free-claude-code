import unittest
from unittest.mock import patch

from free_claude_code.core.anthropic.tools import TerminalJsonToolParser
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec


def identity_tool_names() -> OpenAIToolNameCodec:
    return OpenAIToolNameCodec(
        _original_to_alias={"Read": "Read"},
        _alias_to_original={"Read": "Read"},
        _unchanged_names=frozenset({"Read"}),
    )


READ_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
    },
    "required": ["file_path"],
    "additionalProperties": False,
}


def new_parser():
    return TerminalJsonToolParser.from_schemas(
        tool_names=identity_tool_names(),
        schemas={"Read": READ_SCHEMA},
        enabled=True,
    )


class TerminalJsonToolParserTests(unittest.TestCase):
    def test_valid_terminal_json_recovers_tool_use(self):
        parser = new_parser()

        self.assertEqual(
            parser.feed(
                '{"name":"Read","arguments":{"file_path":".claude/settings.json"}}'
            ),
            "",
        )

        fallback, tools = parser.finish()

        self.assertEqual(fallback, "")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "tool_use")
        self.assertEqual(tools[0]["name"], "Read")
        self.assertEqual(
            tools[0]["input"],
            {"file_path": ".claude/settings.json"},
        )
        self.assertTrue(tools[0]["id"].startswith("toolu_terminal_json_"))

    def test_valid_json_can_arrive_across_chunks(self):
        parser = new_parser()

        self.assertEqual(parser.feed('{"name":"'), "")
        self.assertEqual(parser.feed('Read","arguments":'), "")
        self.assertEqual(
            parser.feed('{"file_path":".claude/settings.json"}}'),
            "",
        )

        fallback, tools = parser.finish()

        self.assertEqual(fallback, "")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Read")

    def test_unknown_tool_is_released_as_text(self):
        parser = new_parser()

        raw = '{"name":"DeleteEverything","arguments":{}}'

        self.assertEqual(parser.feed(raw), "")

        fallback, tools = parser.finish()

        self.assertEqual(fallback, raw)
        self.assertEqual(tools, ())

    def test_invalid_arguments_are_released_as_text(self):
        parser = new_parser()

        raw = '{"name":"Read","arguments":{"wrong_key":"x"}}'

        self.assertEqual(parser.feed(raw), "")

        fallback, tools = parser.finish()

        self.assertEqual(fallback, raw)
        self.assertEqual(tools, ())

    def test_surrounding_text_is_never_recovered(self):
        parser = new_parser()

        raw = (
            "prefix "
            '{"name":"Read","arguments":{"file_path":".claude/settings.json"}} '
            "suffix"
        )

        self.assertEqual(parser.feed(raw), raw)

        fallback, tools = parser.finish()

        self.assertEqual(fallback, "")
        self.assertEqual(tools, ())

    def test_json_with_extra_key_is_released_as_text(self):
        parser = new_parser()

        raw = (
            '{"name":"Read",'
            '"arguments":{"file_path":".claude/settings.json"},'
            '"extra":true}'
        )

        self.assertEqual(parser.feed(raw), "")

        fallback, tools = parser.finish()

        self.assertEqual(fallback, raw)
        self.assertEqual(tools, ())

    def test_non_tool_json_is_released_as_text(self):
        parser = new_parser()

        raw = '{"message":"hello"}'

        self.assertEqual(parser.feed(raw), "")

        fallback, tools = parser.finish()

        self.assertEqual(fallback, raw)
        self.assertEqual(tools, ())

    def test_arguments_must_be_object(self):
        parser = new_parser()

        raw = '{"name":"Read","arguments":"not-an-object"}'

        self.assertEqual(parser.feed(raw), "")

        fallback, tools = parser.finish()

        self.assertEqual(fallback, raw)
        self.assertEqual(tools, ())

    def test_leading_whitespace_valid_json_is_atomic(self):
        parser = new_parser()

        self.assertEqual(parser.feed("  \n"), "")
        self.assertEqual(
            parser.feed('{"name":"Read","arguments":{"file_path":"x"}}'),
            "",
        )

        fallback, tools = parser.finish()

        self.assertEqual(fallback, "")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["input"], {"file_path": "x"})

    def test_non_standard_json_constants_are_released_as_text(self):
        number_schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "required": ["value"],
            "additionalProperties": False,
        }

        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                parser = TerminalJsonToolParser.from_schemas(
                    tool_names=identity_tool_names(),
                    schemas={"Read": number_schema},
                    enabled=True,
                )
                raw = '{"name":"Read","arguments":{"value":' + constant + "}}"

                self.assertEqual(parser.feed(raw), "")

                fallback, tools = parser.finish()

                self.assertEqual(fallback, raw)
                self.assertEqual(tools, ())

    def test_decoder_recursion_error_is_released_as_text(self):
        parser = new_parser()
        raw = '{"name":"Read","arguments":{"file_path":"x"}}'

        self.assertEqual(parser.feed(raw), "")

        with patch(
            "free_claude_code.core.anthropic.tools.json.loads",
            side_effect=RecursionError,
        ):
            fallback, tools = parser.finish()

        self.assertEqual(fallback, raw)
        self.assertEqual(tools, ())

    def test_normal_text_is_released_without_waiting_for_finish(self):
        parser = new_parser()

        self.assertEqual(parser.feed("ordinary response"), "ordinary response")

        fallback, tools = parser.finish()

        self.assertEqual(fallback, "")
        self.assertEqual(tools, ())

    def test_disable_releases_held_candidate_unchanged(self):
        parser = new_parser()

        first = '{"name":"Read",'
        second = '"arguments":{"file_path":"x"}}'

        self.assertEqual(parser.feed(first), "")
        self.assertEqual(parser.disable(), first)
        self.assertEqual(parser.feed(second), second)
        self.assertEqual(parser.finish(), ("", ()))

    def test_disabled_parser_never_intercepts_json(self):
        parser = TerminalJsonToolParser.from_schemas(
            tool_names=identity_tool_names(),
            schemas={"Read": READ_SCHEMA},
            enabled=False,
        )

        raw = '{"name":"Read","arguments":{"file_path":"x"}}'

        self.assertEqual(parser.feed(raw), raw)
        self.assertEqual(parser.finish(), ("", ()))


if __name__ == "__main__":
    unittest.main()
