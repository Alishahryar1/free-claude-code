"""Best-effort token estimates for native Responses requests."""

import json
from collections.abc import Iterator, Mapping, Sequence
from io import StringIO
from typing import NamedTuple

from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.token_estimation import estimate_text_tokens

from .models import OpenAIResponsesRequest

_MAX_ESTIMATE_CHARS_PER_FIELD = 1_000_000


class _ValueStep(NamedTuple):
    value: JsonValue


class _TextStep(NamedTuple):
    text: str


class _QuotedStep(NamedTuple):
    text: str


class _MappingStep(NamedTuple):
    items: Iterator[tuple[str, JsonValue]]
    first: bool


class _SequenceStep(NamedTuple):
    items: Iterator[JsonValue]
    first: bool


type _TraversalStep = (
    _ValueStep | _TextStep | _QuotedStep | _MappingStep | _SequenceStep
)


def estimate_responses_input_tokens(request: OpenAIResponsesRequest) -> int:
    """Estimate only request fields that contribute model input tokens."""

    values: tuple[object, ...] = (
        request.instructions,
        request.input,
        request.tools,
    )
    total = 0
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value[:_MAX_ESTIMATE_CHARS_PER_FIELD]
        else:
            text = _bounded_json_text(value, limit=_MAX_ESTIMATE_CHARS_PER_FIELD)
        total += estimate_text_tokens(text)
    return total


def _bounded_json_text(value: JsonValue, *, limit: int) -> str:
    """Serialize at most ``limit`` JSON-like characters without copying the rest."""

    output = StringIO()
    stack: list[_TraversalStep] = [_ValueStep(value)]
    while stack and output.tell() < limit:
        step = stack.pop()
        if isinstance(step, _TextStep):
            _write_bounded(output, step.text, limit=limit)
            continue
        if isinstance(step, _QuotedStep):
            remaining = limit - output.tell()
            if remaining > 0:
                encoded = json.dumps(step.text[:remaining], ensure_ascii=False)
                _write_bounded(output, encoded, limit=limit)
            continue
        if isinstance(step, _MappingStep):
            try:
                key, item = next(step.items)
            except StopIteration:
                _write_bounded(output, "}", limit=limit)
            else:
                stack.extend(
                    (
                        _MappingStep(step.items, False),
                        _ValueStep(item),
                        _TextStep(":"),
                        _QuotedStep(key),
                    )
                )
                if not step.first:
                    stack.append(_TextStep(","))
            continue
        if isinstance(step, _SequenceStep):
            try:
                item = next(step.items)
            except StopIteration:
                _write_bounded(output, "]", limit=limit)
            else:
                stack.extend(
                    (
                        _SequenceStep(step.items, False),
                        _ValueStep(item),
                    )
                )
                if not step.first:
                    stack.append(_TextStep(","))
            continue

        item = step.value
        if isinstance(item, str):
            stack.append(_QuotedStep(item))
        elif isinstance(item, Mapping):
            _write_bounded(output, "{", limit=limit)
            stack.append(_MappingStep(iter(item.items()), True))
        elif isinstance(item, Sequence):
            _write_bounded(output, "[", limit=limit)
            stack.append(_SequenceStep(iter(item), True))
        else:
            _write_bounded(
                output,
                json.dumps(item, ensure_ascii=False),
                limit=limit,
            )
    return output.getvalue()


def _write_bounded(output: StringIO, text: str, *, limit: int) -> None:
    remaining = limit - output.tell()
    if remaining > 0:
        output.write(text[:remaining])
