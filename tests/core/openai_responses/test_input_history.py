from collections.abc import Iterator, Mapping
from typing import get_args, get_type_hints

from openai.types.responses.response_input_item_param import ResponseInputItemParam

from free_claude_code.core.openai_responses import (
    ResponsesConversionError,
    read_history,
)


def literal_strings(annotation: object) -> Iterator[str]:
    arguments: tuple[object, ...] = get_args(annotation)
    for value in arguments:
        if isinstance(value, str):
            yield value
        else:
            yield from literal_strings(value)


def test_every_standard_input_item_is_translated_or_rejected() -> None:
    kinds: set[str] = set()
    variants: tuple[object, ...] = get_args(ResponseInputItemParam)
    for model in variants:
        assert isinstance(model, type)
        fields: Mapping[str, object] = get_type_hints(model)
        kinds.update(literal_strings(fields["type"]))
    assert "function_call" in kinds
    skipped = []
    for kind in sorted(kinds):
        try:
            items = list(read_history([{"type": kind}]))
        except ResponsesConversionError:
            continue
        if not items:
            skipped.append(kind)
    assert not skipped, (
        f"Standard Responses items fell through as optional extensions: {skipped}"
    )
