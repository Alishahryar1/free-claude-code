"""Typed views of arbitrary mock call arguments."""

from collections.abc import Mapping
from unittest.mock import _Call


def positional_arguments(call: _Call) -> tuple[object, ...]:
    return call.args


def keyword_arguments(call: _Call) -> Mapping[str, object]:
    return call.kwargs
