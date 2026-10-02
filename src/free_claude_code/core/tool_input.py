"""Structural tool-input checks that never rewrite the original argument bytes."""

import json


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def complete_json_object(arguments: str) -> bool:
    """Accept one JSON object without interpreting numeric values or its schema."""
    try:
        value = json.loads(
            arguments,
            parse_float=str,
            parse_int=str,
            parse_constant=_reject_constant,
        )
    except ValueError, RecursionError:
        return False
    return isinstance(value, dict)
