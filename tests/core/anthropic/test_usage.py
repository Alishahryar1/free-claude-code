from typing import cast

import pytest

from free_claude_code.core.anthropic.usage import anthropic_input_usage_fields


@pytest.mark.parametrize(
    ("total_tokens", "cache_read_tokens", "cache_creation_tokens", "expected"),
    [
        (
            30,
            10,
            5,
            {
                "input_tokens": 15,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            },
        ),
        (30, 10, None, {"input_tokens": 20, "cache_read_input_tokens": 10}),
        (
            30,
            None,
            5,
            {"input_tokens": 25, "cache_creation_input_tokens": 5},
        ),
        (
            30,
            0,
            0,
            {
                "input_tokens": 30,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ),
        (30, None, None, {}),
        (None, 10, 5, {}),
        (-1, 0, 0, {}),
        (True, 0, 0, {}),
        (30.0, 10, 5, {}),
        (
            30,
            -1,
            5,
            {"input_tokens": 25, "cache_creation_input_tokens": 5},
        ),
        (
            30,
            True,
            5,
            {"input_tokens": 25, "cache_creation_input_tokens": 5},
        ),
        (
            30,
            "10",
            5,
            {"input_tokens": 25, "cache_creation_input_tokens": 5},
        ),
        (
            30,
            31,
            5,
            {"input_tokens": 25, "cache_creation_input_tokens": 5},
        ),
        (30, 10, -1, {"input_tokens": 20, "cache_read_input_tokens": 10}),
        (30, 10, True, {"input_tokens": 20, "cache_read_input_tokens": 10}),
        (30, 10, "5", {"input_tokens": 20, "cache_read_input_tokens": 10}),
        (30, 10, 21, {"input_tokens": 20, "cache_read_input_tokens": 10}),
    ],
    ids=[
        "read-and-creation",
        "read-only",
        "creation-only",
        "explicit-zeroes",
        "no-cache-details",
        "missing-total",
        "negative-total",
        "boolean-total",
        "float-total",
        "negative-read",
        "boolean-read",
        "string-read",
        "read-over-total",
        "negative-creation",
        "boolean-creation",
        "string-creation",
        "creation-over-remaining-total",
    ],
)
def test_anthropic_input_usage_fields_preserve_only_trustworthy_partitions(
    total_tokens: object,
    cache_read_tokens: object,
    cache_creation_tokens: object,
    expected: dict[str, int],
) -> None:
    assert (
        anthropic_input_usage_fields(
            cast(int | None, total_tokens),
            cache_read_tokens=cast(int | None, cache_read_tokens),
            cache_creation_tokens=cast(int | None, cache_creation_tokens),
        )
        == expected
    )
