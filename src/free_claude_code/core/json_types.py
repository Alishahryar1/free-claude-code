"""Shared JSON value vocabulary for wire and persistence boundaries."""

from collections.abc import Mapping, Sequence

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | Sequence[JsonValue] | Mapping[str, JsonValue]
type JsonObject = dict[str, JsonValue]
