from free_claude_code.core.anthropic.tool_schema import sanitize_tool_schema_patterns

# Claude Code's Artifact tool sends a JSON Schema pattern using Unicode property
# escapes plus a negative lookahead; OpenAI-compatible validators reject it.
ARTIFACT_PATTERN = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}'


def test_sanitize_returns_schema_without_patterns_unchanged() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    assert sanitize_tool_schema_patterns(schema) is schema


def test_sanitize_preserves_supported_pattern_verbatim() -> None:
    schema = {"type": "string", "pattern": r"^[a-z]+$"}

    assert sanitize_tool_schema_patterns(schema) is schema


def test_sanitize_drops_narrowing_pattern_instead_of_stripping() -> None:
    # Stripping \p{L} from [\p{L}\d]+ would leave [\d]+, which rejects values
    # the original accepted — the whole pattern must go.
    schema = {"type": "string", "pattern": r"[\p{L}\d]+"}

    assert sanitize_tool_schema_patterns(schema) == {"type": "string"}


def test_sanitize_walks_nested_subschema_locations() -> None:
    schema = {
        "type": "object",
        "$defs": {"ref": {"type": "string", "pattern": r"\p{Lu}\d*"}},
        "properties": {
            "name": {"type": "string", "pattern": ARTIFACT_PATTERN},
            "tags": {
                "type": "array",
                "items": {"type": "string", "pattern": r"\p{Lu}\d*"},
            },
            "either": {
                "anyOf": [
                    {"type": "string", "pattern": r"\p{N}"},
                    {"type": "null"},
                ]
            },
            "extra": {
                "type": "object",
                "additionalProperties": {"type": "string", "pattern": r"\p{L}"},
            },
            "plain": {"type": "string", "pattern": r"^[a-z-]+$"},
        },
    }

    sanitized = sanitize_tool_schema_patterns(schema)

    assert sanitized["$defs"] == {"ref": {"type": "string"}}
    assert sanitized["properties"]["name"] == {"type": "string"}
    assert sanitized["properties"]["tags"]["items"] == {"type": "string"}
    assert sanitized["properties"]["either"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert sanitized["properties"]["extra"]["additionalProperties"] == {
        "type": "string"
    }
    assert sanitized["properties"]["plain"] == {
        "type": "string",
        "pattern": r"^[a-z-]+$",
    }
    assert schema["properties"]["name"]["pattern"] == ARTIFACT_PATTERN


def test_sanitize_preserves_property_named_pattern() -> None:
    schema = {"properties": {"pattern": {"type": "string", "pattern": r"\p{L}+"}}}

    sanitized = sanitize_tool_schema_patterns(schema)

    assert sanitized == {"properties": {"pattern": {"type": "string"}}}


def test_sanitize_walks_single_subschema_keywords() -> None:
    schema = {
        "type": "object",
        "propertyNames": {"type": "string", "pattern": r"\p{L}+"},
        "unevaluatedProperties": {"type": "string", "pattern": r"\p{N}+"},
        "patternProperties": {
            r"^[a-z]+$": {"type": "string", "pattern": r"\p{L}+"},
        },
    }

    sanitized = sanitize_tool_schema_patterns(schema)

    assert sanitized["propertyNames"] == {"type": "string"}
    assert sanitized["unevaluatedProperties"] == {"type": "string"}
    assert sanitized["patternProperties"] == {r"^[a-z]+$": {"type": "string"}}


def test_sanitize_walks_additional_items_and_content_schema() -> None:
    schema = {
        "type": "array",
        "prefixItems": [{"type": "string"}],
        "additionalItems": {"type": "string", "pattern": r"\p{L}+"},
        "contentMediaType": "text/plain",
        "contentSchema": {"type": "string", "pattern": r"\p{N}+"},
    }

    sanitized = sanitize_tool_schema_patterns(schema)

    assert sanitized["additionalItems"] == {"type": "string"}
    assert sanitized["contentSchema"] == {"type": "string"}
    assert sanitized["contentMediaType"] == "text/plain"


def test_sanitize_walks_schema_valued_dependencies_only() -> None:
    schema = {
        "type": "object",
        "dependencies": {
            # Property-name arrays are data, not subschemas, and must survive.
            "billing": ["shipping"],
            "credit_card": {"type": "object", "pattern": r"\p{L}+"},
        },
    }

    sanitized = sanitize_tool_schema_patterns(schema)

    assert sanitized["dependencies"]["billing"] == ["shipping"]
    assert sanitized["dependencies"]["credit_card"] == {"type": "object"}


def test_sanitize_leaves_literal_value_keywords_untouched() -> None:
    schema = {
        "type": "object",
        "enum": [{"pattern": r"\p{L}"}, {"nested": {"pattern": r"\p{L}"}}],
        "const": {"pattern": r"\p{L}"},
        "default": {"pattern": r"\p{L}"},
        "examples": [{"pattern": r"\p{L}"}],
    }

    assert sanitize_tool_schema_patterns(schema) is schema


def test_sanitize_does_not_mutate_input() -> None:
    schema = {"type": "string", "pattern": ARTIFACT_PATTERN}

    sanitize_tool_schema_patterns(schema)

    assert schema == {"type": "string", "pattern": ARTIFACT_PATTERN}


def test_sanitize_leaves_non_string_pattern_untouched() -> None:
    schema = {"type": "string", "pattern": 123}

    assert sanitize_tool_schema_patterns(schema) is schema
