"""Portable Unicode exclusions for Claude tool parameter schemas."""

import re
import unicodedata
from functools import cache
from typing import Any

_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_SCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
        "dependencies",
    }
)
_SCHEMA_KEYS = frozenset(
    {
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
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
)


def translate_tool_schema_patterns(schema: Any) -> Any:
    """Copy changed schema paths only; instance data and unsupported regexes pass through."""
    if isinstance(schema, list):
        items = [translate_tool_schema_patterns(item) for item in schema]
        return (
            items
            if any(a is not b for a, b in zip(items, schema, strict=True))
            else schema
        )
    if not isinstance(schema, dict):
        return schema
    result = {}
    changed = False
    for key, value in schema.items():
        replacement = value
        if key == "pattern" and isinstance(value, str):
            replacement = _translate_pattern(value)
        elif key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            mapped = {
                name: translate_tool_schema_patterns(child)
                for name, child in value.items()
            }
            if any(mapped[name] is not child for name, child in value.items()):
                replacement = mapped
        elif key in _SCHEMA_KEYS:
            replacement = translate_tool_schema_patterns(value)
        result[key] = replacement
        changed |= replacement is not value
    return result if changed else schema


def _translate_pattern(pattern: str) -> str:
    # A bounded scanner for four standalone atoms in ordinary negated classes,
    # not a general regex transpiler. Never partially translate unsupported input.
    if r"\p{" not in pattern:
        return pattern
    replacements: list[tuple[int, int, str]] = []
    in_class = False
    negated = False
    previous = ""
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            if index + 1 == len(pattern):
                return pattern
            if pattern[index + 1] in "pP":
                end = pattern.find("}", index + 3)
                category = pattern[index + 3 : end]
                if (
                    not pattern.startswith(r"\p{", index)
                    or end == -1
                    or category not in _CATEGORIES
                    or not in_class
                    or not negated
                    or previous == "-"
                    or pattern[end + 1 : end + 2] == "-"
                ):
                    return pattern
                replacements.append((index, end + 1, category))
                previous = pattern[index : end + 1]
                index = end + 1
                continue
            previous = pattern[index : index + 2]
            index += 2
            continue
        if char == "[":
            if in_class:
                # Artifact's ordinary class includes a literal '[' before '\]'.
                # Other nested forms may be Unicode set syntax; leave them alone.
                if not pattern.startswith(r"[\]]", index):
                    return pattern
            else:
                in_class = True
                negated = pattern[index + 1 : index + 2] == "^"
        elif char == "]":
            if not in_class:
                return pattern
            in_class = False
        elif in_class and pattern[index : index + 2] in {"&&", "--", "~~", "||"}:
            return pattern
        previous = char
        index += 1
    if in_class or not replacements:
        return pattern
    ranges = _category_ranges()
    parts = []
    offset = 0
    for start, end, category in replacements:
        parts.extend((pattern[offset:start], ranges[category]))
        offset = end
    parts.append(pattern[offset:])
    translated = "".join(parts)
    try:
        re.compile(translated)
    except re.error:
        return pattern
    return translated


@cache
def _category_ranges() -> dict[str, str]:
    """Build the four finite tables lazily, using Python's Unicode database."""
    ranges: dict[str, list[tuple[int, int]]] = {key: [] for key in _CATEGORIES}
    for point in range(0x110000):
        category = unicodedata.category(chr(point))
        if category not in ranges:
            continue
        spans = ranges[category]
        if spans and spans[-1][1] == point - 1:
            spans[-1] = (spans[-1][0], point)
        else:
            spans.append((point, point))
    return {
        category: "".join(
            _codepoint(start)
            if start == end
            else f"{_codepoint(start)}-{_codepoint(end)}"
            for start, end in spans
        )
        for category, spans in ranges.items()
    }


def _codepoint(point: int) -> str:
    # Actual astral scalars work in JS Unicode mode and Python. Surrogate ranges
    # would change the exclusion; \U and \u{...} escapes are dialect-specific.
    return f"\\u{point:04X}" if point <= 0xFFFF else chr(point)
