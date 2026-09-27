import pytest

from free_claude_code.core.anthropic import tools as tool_parsers
from free_claude_code.core.anthropic.tools import (
    FunctionTagToolParser,
    HeuristicToolParser,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec


def _parser(schema, *, name="Task", enabled=True):
    return HeuristicToolParser(
        tool_names=OpenAIToolNameCodec.from_names([name]),
        schemas={name: schema},
        enabled=enabled,
    )


def _parts(parser, chunks):
    parts = []
    for chunk in chunks:
        parts.extend(parser.feed(chunk))
    parts.extend(parser.flush())
    return parts


@pytest.mark.parametrize(
    "property_schema,value,expected",
    [
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "string"}, "true", "true"),
        ({"type": "string"}, "123", "123"),
        ({"type": "string"}, "[1,2]", "[1,2]"),
        ({"type": "integer"}, "42", 42),
        ({"type": "number"}, "0.5", 0.5),
        ({"type": "array"}, '[1,"x"]', [1, "x"]),
        ({"type": "object"}, '{"x":1}', {"x": 1}),
        ({"type": "null"}, "null", None),
        ({"type": ["boolean", "null"]}, "true", True),
        ({"type": ["boolean", "string"]}, "true", "true"),
        ({"enum": [1, 2]}, "2", 2),
        ({"const": 2}, "2", 2),
    ],
)
def test_text_parameter_decoding_matches_both_parsers(property_schema, value, expected):
    schema = {
        "type": "object",
        "properties": {"value": property_schema},
        "required": ["value"],
    }
    function = f"<function=Task><parameter=value>{value}</parameter>"
    parts = _parts(_parser(schema), [f"● {function}"])
    assert len(parts) == 1
    assert parts[0]["input"] == {"value": expected}
    parser = FunctionTagToolParser.from_schemas(
        tool_names=OpenAIToolNameCodec.from_names(["Task"]),
        schemas={"Task": schema},
        enabled=True,
    )
    assert parser.feed(f"<tool_call>{function}</function></tool_call>") == ""
    text, tools = parser.finish()
    assert text == ""
    assert tools[0]["input"] == {"value": expected}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": {"type": "boolean"}}},
        {"type": "object", "required": ["missing"]},
        {"type": "invalid"},
        {"$ref": "#/missing"},
        {"$ref": "https://example.invalid/never-fetch-this-schema"},
    ],
)
def test_rejected_candidates_survive_every_chunk_boundary(schema):
    text = "Before ● <function=Task><parameter=value>invalid</parameter> After"
    for split in range(len(text) + 1):
        parts = _parts(_parser(schema), [text[:split], text[split:]])
        assert all(isinstance(part, str) for part in parts)
        assert "".join(parts) == text


@pytest.mark.parametrize("name,enabled", [("Unknown", True), ("Task", False)])
def test_undeclared_or_disabled_candidate_is_text(name, enabled):
    text = f"● <function={name}><parameter=value>true</parameter>"
    assert "".join(_parts(_parser({"type": "object"}, enabled=enabled), [text])) == text


def test_mixed_candidates_keep_order_and_rejected_text():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "boolean"}},
        "required": ["value"],
    }
    bad = "● <function=Task><parameter=value>bad</parameter>"
    good = "● <function=Task><parameter=value>true</parameter>"
    text = f"before {bad} between {good} after"
    for split in range(len(text) + 1):
        parts = _parts(_parser(schema), [text[:split], text[split:]])
        normalized = "".join(
            part if isinstance(part, str) else "[tool]" for part in parts
        )
        assert normalized == f"before {bad} between [tool] after"
        assert [p["input"] for p in parts if isinstance(p, dict)] == [{"value": True}]


def test_accepted_call_survives_parameter_and_whitespace_splits():
    schema = {
        "type": "object",
        "properties": {"one": {"type": "boolean"}, "two": {"type": "string"}},
        "required": ["one", "two"],
    }
    text = "● <function=Task><parameter=one>true</parameter>\n<parameter=two>true</parameter>"
    parts = _parts(_parser(schema), list(text))
    assert [p["input"] for p in parts if isinstance(p, dict)] == [
        {"one": True, "two": "true"}
    ]


def test_web_json_preserves_types_and_surrounding_text():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "flag": {"type": "boolean"}},
        "required": ["query", "flag"],
    }
    invalid = 'Use WebSearch {"query":"test","flag":"true"}'
    valid = 'Use WebSearch {"query":"test","flag":true}'
    text = f"before {invalid} between {valid} after"
    parts = _parts(_parser(schema, name="WebSearch"), [text])
    assert (
        "".join(p if isinstance(p, str) else "[tool]" for p in parts)
        == f"before {invalid} between [tool] after"
    )
    assert [p["input"] for p in parts if isinstance(p, dict)] == [
        {"query": "test", "flag": True}
    ]


def test_overlarge_candidate_is_released_once_and_disables_inference(monkeypatch):
    monkeypatch.setattr(tool_parsers, "_MAX_FUNCTION_TAG_CANDIDATE_CHARS", 80)
    text = "● <function=Task><parameter=value>" + "x" * 90
    later = "● <function=Task><parameter=value>y</parameter>"
    parser = _parser({"type": "object"})
    parts = _parts(parser, [text[:40], text[40:], later])
    assert all(isinstance(p, str) for p in parts)
    assert "".join(parts) == text + later


def test_nonfinite_text_parameter_stays_text():
    schema = {"type": "object", "properties": {"value": {"type": "number"}}}
    text = "● <function=Task><parameter=value>NaN</parameter>"
    assert "".join(_parts(_parser(schema), [text])) == text


def test_aliased_tool_name_resolves_before_schema_lookup():
    name = "worker with spaces"
    schema = {"type": "object", "properties": {"value": {"type": "boolean"}}}
    codec = OpenAIToolNameCodec.from_names([name])
    text = f"● <function={codec.encode(name)}><parameter=value>true</parameter>"
    parts = _parts(_parser(schema, name=name), [text])
    assert parts[0]["name"] == name
    assert parts[0]["input"] == {"value": True}
