"""Compare emitted regexes with the client's JavaScript Unicode semantics."""

import unicodedata

from free_claude_code.core.tool_schema_patterns import translate_tool_schema_patterns
from tests.core.test_tool_schema_patterns import ARTIFACT_PATTERN


def test_artifact_pattern_matches_javascript_reference(page):
    converted = translate_tool_schema_patterns({"pattern": ARTIFACT_PATTERN})["pattern"]
    samples = [
        "",
        "demo",
        "é名字😀",
        "__name__",
        "__name",
        "name__",
        "-name-",
        "a" * 200,
        "a" * 201,
        *["x" + char + "y" for char in r'"\/.[]'],
        *[
            chr(point)
            for point in range(0x110000)
            if unicodedata.category(chr(point)) in {"Cc", "Cf", "Zl", "Zp"}
        ],
        "\U000110bc",
        "\U000110be",
        "\U000e0000",
        "\U000e0080",
    ]
    mismatches = page.evaluate(
        """({original, converted, samples}) => {
            const before = new RegExp(original, 'u');
            const after = new RegExp(converted, 'u');
            const mismatches = samples.filter(text => before.test(text) !== after.test(text));
            for (let point = 0; point < 0x110000; point++) {
                if (point >= 0xD800 && point <= 0xDFFF) continue;
                const text = String.fromCodePoint(point);
                if (before.test(text) !== after.test(text)) {
                    mismatches.push(`U+${point.toString(16)}`);
                    if (mismatches.length >= 10) break;
                }
            }
            return mismatches;
        }""",
        {"original": ARTIFACT_PATTERN, "converted": converted, "samples": samples},
    )
    assert mismatches == [], f"Unicode {unicodedata.unidata_version}: {mismatches!r}"
