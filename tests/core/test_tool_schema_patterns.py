"""Regressions for issue #1730 at the shared Claude conversion boundary."""

import re
import unicodedata
from copy import deepcopy

import pytest

from free_claude_code.core.anthropic.conversion import build_base_request_body
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.tool_schema_patterns import translate_tool_schema_patterns

ARTIFACT_PATTERN = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'


@pytest.mark.parametrize("wire_format", ["chat", "responses"])
@pytest.mark.parametrize(
    "pattern", [r"^[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}]{1,200}$", ARTIFACT_PATTERN]
)
def test_claude_converters_preserve_artifact_rules_in_portable_pattern(
    wire_format, pattern
):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "pattern": pattern}},
        "required": ["name"],
    }
    request = MessagesRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call Artifact with demo"}],
            "tools": [
                {"name": "Artifact", "description": "Save", "input_schema": schema}
            ],
        }
    )
    original = deepcopy(request.model_dump())
    if wire_format == "chat":
        body = build_base_request_body(request)
        parameters = body["tools"][0]["function"]["parameters"]
    else:
        body = build_responses_provider_request(
            request, reasoning=ReasoningPolicy.provider_default()
        )
        parameters = body["tools"][0]["parameters"]
    translated = parameters["properties"]["name"]["pattern"]
    assert r"\p{" not in translated
    regex = re.compile(translated)
    assert regex.fullmatch("demo")
    assert not regex.fullmatch("bad\u2028name")
    assert not regex.fullmatch("bad\U000e0001name")
    assert not regex.fullmatch("a" * 201)
    assert not regex.fullmatch("")
    assert parameters["required"] == ["name"]
    assert request.model_dump() == original


def translated_pattern(pattern):
    return translate_tool_schema_patterns({"pattern": pattern})["pattern"]


@pytest.mark.parametrize("category", ["Cc", "Cf", "Zl", "Zp"])
def test_category_expansion_preserves_every_unicode_scalar(category):
    regex = re.compile(translated_pattern(r"[^\p{" + category + "}]+"))
    for point in range(0x110000):
        if 0xD800 <= point <= 0xDFFF:
            continue
        char = chr(point)
        assert bool(regex.fullmatch(char)) == (
            unicodedata.category(char) != category
        ), f"U+{point:04X}, category {category}, Unicode {unicodedata.unidata_version}"


@pytest.mark.parametrize(
    "name,accepted",
    [
        ("demo", True),
        ("é名字😀", True),
        ("__name", True),
        ("name__", True),
        ("-name-", True),
        ("a", True),
        ("a" * 200, True),
        ("", False),
        ("a" * 201, False),
        ("__name__", False),
        *[("x" + char + "y", False) for char in r'"\/.[]'],
        *[
            ("x" + char + "y", False)
            for char in [
                "\x00",
                "\x1f",
                "\x7f",
                "\x9f",
                "\u200d",
                "\u2028",
                "\u2029",
                "\U000110bd",
                "\U000e0001",
                "\U000e0020",
                "\U000e007f",
            ]
        ],
    ],
)
def test_artifact_restrictions_survive_translation(name, accepted):
    assert bool(re.fullmatch(translated_pattern(ARTIFACT_PATTERN), name)) == accepted


@pytest.mark.parametrize(
    "pattern",
    [
        r"^[a-z]+$",
        r"[^\\p{Cc}]",
        r"\\p{Cc}",
        r"[\p{Cc}]",
        r"\p{Cc}",
        r"[^\P{Cc}]",
        r"[^\p{L}]",
        r"[^\p{cc}]",
        r"[^\p{Cc}\p{L}]",
        r"[^\p{Cc}]\P{Cf}",
        r"[^\p{Cc}]\p{Cf}",
        r"[^\p{Cc}\p{General_Category=Format}]",
        r"[^\p{Cc}\p{Cf]",
        r"[^\p{Cc}",
        r"[^\p{Cc}]" + "\\",
        r"([^\p{Cc}]",
        r"[^\p{Cc}-z]",
        r"[^a-\p{Cc}]",
        r"[^\p{Cc}--a]",
        r"[^\p{Cc}&&a]",
        r"[^\p{Cc}~~a]",
        r"[^\p{Cc}||a]",
        r"[^\p{Cc}[a]]",
        r"\[^\p{Cc}]",
    ],
)
def test_unsupported_or_literal_patterns_remain_unchanged(pattern):
    schema = {"type": "string", "pattern": pattern}
    assert translate_tool_schema_patterns(schema) is schema


@pytest.mark.parametrize(
    "pattern,excluded,accepted",
    [
        (r"[^\\\p{Cc}]", "\\\x00", "p{}C"),
        (r"[^\p{Cc}\-]", "-\x00", "az"),
        (r"[^\]\p{Cc}]", "]\x00", "["),
        (r"[^\p{Cc}[\]]", "[]\x00", "az"),
        (r"[^\p{Zl}][^\p{Zp}]", "\u2028x", "ab"),
    ],
)
def test_class_escaping_is_preserved(pattern, excluded, accepted):
    translated = translated_pattern(pattern)
    assert translated != pattern
    regex = re.compile(translated)
    if pattern == r"[^\p{Zl}][^\p{Zp}]":
        assert regex.fullmatch(accepted)
        assert not regex.fullmatch(excluded)
    else:
        assert all(not regex.fullmatch(char) for char in excluded)
        assert all(regex.fullmatch(char) for char in accepted)


@pytest.mark.parametrize(
    "container",
    [
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
        "dependencies",
    ],
)
def test_schema_maps_visit_values_and_preserve_names(container):
    child = {"type": "string", "pattern": r"[^\p{Cc}]"}
    schema = {container: {"pattern": child, r"\p{Cc}": True, "required": ["name"]}}
    original = deepcopy(schema)
    converted = translate_tool_schema_patterns(schema)
    assert converted[container]["pattern"]["pattern"] != child["pattern"]
    assert converted[container][r"\p{Cc}"] is True
    assert converted[container]["required"] == ["name"]
    assert schema == original


@pytest.mark.parametrize(
    "keyword",
    [
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "items",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
        "contentSchema",
    ],
)
def test_single_schema_keywords_are_translated(keyword):
    schema = {keyword: {"pattern": r"[^\p{Cf}]"}}
    converted = translate_tool_schema_patterns(schema)
    assert converted[keyword]["pattern"] != schema[keyword]["pattern"]


@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf", "prefixItems", "items"])
def test_schema_arrays_preserve_boolean_and_unchanged_children(keyword):
    child = {"type": "string", "pattern": r"[a-z]"}
    schema = {keyword: [child, False, {"pattern": r"[^\p{Cf}]"}]}
    converted = translate_tool_schema_patterns(schema)
    assert converted[keyword][0] is child
    assert converted[keyword][1] is False
    assert converted[keyword][2]["pattern"] != r"[^\p{Cf}]"


def test_instance_data_is_untouched_and_translation_is_idempotent():
    literal = {"pattern": r"[^\p{Cc}]"}
    schema = {
        "type": "object",
        "properties": {"name": literal},
        "default": literal,
        "const": literal,
        "enum": [literal],
        "examples": [literal],
        "x-custom": literal,
        "description": r"\p{Cc}",
    }
    before = deepcopy(schema)
    converted = translate_tool_schema_patterns(schema)
    for key in ("default", "const", "enum", "examples", "x-custom", "description"):
        assert converted[key] is schema[key]
    assert schema == before
    assert translate_tool_schema_patterns(converted) is converted


@pytest.mark.parametrize(
    "schema", [True, False, {}, {"pattern": 123}, {"type": "string"}]
)
def test_non_patterns_remain_unchanged(schema):
    assert translate_tool_schema_patterns(schema) is schema
