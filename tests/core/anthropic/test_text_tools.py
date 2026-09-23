import pytest

from free_claude_code.core.anthropic import FunctionTagToolParser
from free_claude_code.core.anthropic import tools as tool_parsers
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from tests.providers.request_factory import make_messages_request


def _request(*tools: dict):
    return make_messages_request(
        "test-model",
        messages=[],
        max_tokens=256,
        tools=list(tools),
    )


def _tool(name: str, properties: dict, required: list[str] | None = None) -> dict:
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required is not None:
        schema["required"] = required
    return {"name": name, "input_schema": schema}


def _call(name: str, **arguments: str) -> str:
    parameters = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def _parse(parser: FunctionTagToolParser, *chunks: str):
    visible = "".join(parser.feed(chunk) for chunk in chunks)
    tail, calls = parser.finish()
    return visible + tail, calls


def test_function_tag_parser_accepts_exact_reserved_response() -> None:
    parser = FunctionTagToolParser(
        _request(_tool("Bash", {"command": {"type": "string"}}, ["command"]))
    )

    visible, calls = _parse(parser, _call("Bash", command="printf ok"))

    assert visible == ""
    assert len(calls) == 1
    assert calls[0]["id"].startswith("toolu_function_tag_")
    assert calls[0]["name"] == "Bash"
    assert calls[0]["input"] == {"command": "printf ok"}


def test_function_tag_parser_handles_every_two_chunk_split() -> None:
    raw = _call("Read", file_path="README.md")
    request = _request(_tool("Read", {"file_path": {"type": "string"}}, ["file_path"]))

    for split in range(len(raw) + 1):
        visible, calls = _parse(
            FunctionTagToolParser(request),
            raw[:split],
            raw[split:],
        )
        assert visible == "", split
        assert [call["input"] for call in calls] == [{"file_path": "README.md"}], split


def test_function_tag_parser_streams_prefix_and_parses_terminal_call() -> None:
    request = _request(_tool("Bash", {"command": {"type": "string"}}, ["command"]))
    prefix = "I will run the requested command.\n"
    raw_call = _call("Bash", command="printf ok")
    parser = FunctionTagToolParser(request)

    first_visible = parser.feed(prefix + raw_call[:5])
    second_visible = parser.feed(raw_call[5:])
    tail, calls = parser.finish()

    assert first_visible == prefix
    assert second_visible == ""
    assert tail == ""
    assert [call["input"] for call in calls] == [{"command": "printf ok"}]


def test_function_tag_parser_accepts_multiple_typed_calls_atomically() -> None:
    request = _request(
        _tool(
            "Configure",
            {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "nothing": {"type": "null"},
                "items": {"type": "array", "items": {"type": "integer"}},
                "metadata": {"type": "object"},
                "mode": {"enum": ["fast", "safe"]},
                "fixed": {"const": 7},
            },
            [
                "count",
                "ratio",
                "enabled",
                "nothing",
                "items",
                "metadata",
                "mode",
                "fixed",
            ],
        ),
        _tool("Bash", {"command": {"type": "string"}}, ["command"]),
    )
    raw = "\n".join(
        (
            _call(
                "Configure",
                count="2",
                ratio="1.5",
                enabled="true",
                nothing="null",
                items="[1,2]",
                metadata='{"key":"value"}',
                mode="safe",
                fixed="7",
            ),
            _call("Bash", command="printf first\nprintf second"),
        )
    )

    visible, calls = _parse(FunctionTagToolParser(request), raw)

    assert visible == ""
    assert [call["name"] for call in calls] == ["Configure", "Bash"]
    assert calls[0]["input"] == {
        "count": 2,
        "ratio": 1.5,
        "enabled": True,
        "nothing": None,
        "items": [1, 2],
        "metadata": {"key": "value"},
        "mode": "safe",
        "fixed": 7,
    }
    assert calls[1]["input"] == {"command": "printf first\nprintf second"}


def test_function_tag_parser_decodes_openai_tool_alias() -> None:
    original = "mcp__function_tag__" + "x" * 80
    request = _request(_tool(original, {}, []))
    alias = OpenAIToolNameCodec.from_request(request).encode(original)

    visible, calls = _parse(FunctionTagToolParser(request), _call(alias))

    assert visible == ""
    assert [call["name"] for call in calls] == [original]


@pytest.mark.parametrize(
    "raw",
    [
        "Example: {call}\nThat was only an example.",
        "{call}\nDone.",
        "```xml\n{call}\n```",
        "<tool_call><function=Bash></function>",
        (
            "<tool_call><function=Bash>"
            "<parameter=command>x</parameter>"
            "<parameter=command>y</parameter>"
            "</function></tool_call>"
        ),
        "<tool_call><function=Unknown></function></tool_call>",
        "<tool_call><function=Bash></function></tool_call>",
        "{wrong_type}",
        "{extra_parameter}",
    ],
)
def test_function_tag_parser_preserves_non_control_lookalikes(raw: str) -> None:
    valid_call = _call("Bash", command="printf ok")
    wrong_type = _call("Count", value="not-an-integer")
    extra_parameter = _call("Bash", command="printf ok", extra="no")
    raw = raw.format(
        call=valid_call,
        wrong_type=wrong_type,
        extra_parameter=extra_parameter,
    )
    request = _request(
        _tool("Bash", {"command": {"type": "string"}}, ["command"]),
        _tool("Count", {"value": {"type": "integer"}}, ["value"]),
    )

    visible, calls = _parse(FunctionTagToolParser(request), raw)

    assert visible == raw
    assert calls == ()


def test_function_tag_parser_preserves_whole_multi_call_candidate_on_failure() -> None:
    request = _request(_tool("Bash", {"command": {"type": "string"}}, ["command"]))
    raw = _call("Bash", command="printf ok") + _call("Unknown")

    visible, calls = _parse(FunctionTagToolParser(request), raw)

    assert visible == raw
    assert calls == ()


def test_function_tag_parser_without_tools_is_immediate_passthrough() -> None:
    raw = _call("Bash", command="printf ok")
    parser = FunctionTagToolParser(_request())

    assert parser.feed(raw) == raw
    assert parser.finish() == ("", ())


def test_function_tag_parser_respects_explicit_none_choice() -> None:
    raw = _call("Bash", command="printf ok")
    request = make_messages_request(
        "test-model",
        messages=[],
        max_tokens=256,
        tools=[_tool("Bash", {"command": {"type": "string"}}, ["command"])],
        tool_choice={"type": "none"},
    )
    parser = FunctionTagToolParser(request)

    assert parser.feed(raw) == raw
    assert parser.finish() == ("", ())


def test_function_tag_parser_disable_releases_candidate_once() -> None:
    raw = _call("Bash", command="printf ok")
    parser = FunctionTagToolParser(
        _request(_tool("Bash", {"command": {"type": "string"}}, ["command"]))
    )

    assert parser.feed(raw[:20]) == ""
    assert parser.disable() == raw[:20]
    assert parser.disable() == ""
    assert parser.feed(raw[20:]) == raw[20:]
    assert parser.finish() == ("", ())


def test_function_tag_parser_releases_oversized_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _call("Bash", command="printf ok")
    monkeypatch.setattr(
        tool_parsers,
        "_MAX_FUNCTION_TAG_CANDIDATE_CHARS",
        len(raw) - 1,
    )
    parser = FunctionTagToolParser(
        _request(_tool("Bash", {"command": {"type": "string"}}, ["command"]))
    )

    assert parser.feed(raw) == raw
    assert parser.finish() == ("", ())
