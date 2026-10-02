"""Tool identities and restoration used by decoded source observations."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from .json_types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class ResponsesToolIdentity:
    kind: Literal["function", "custom"]
    name: str
    namespace: str | None = None


class ResponsesToolEvents(Protocol):
    def feed(
        self, event_type: str, payload: JsonObject
    ) -> Iterable[tuple[str, JsonObject]]: ...


class ResponsesToolMapping(Protocol):
    def event_adapter(self) -> ResponsesToolEvents | None: ...
    def restore_item(self, value: JsonValue) -> JsonValue: ...
