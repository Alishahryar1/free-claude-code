"""Which completed client calls a translated provider request can authorize."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .failures import ExecutionFailure, FailureKind
from .json_types import JsonValue

type _Wire = Literal["chat", "messages", "responses"]
type _Identity = tuple[str, str, str | None]


def has_tool_selection(tools: object, choice: object) -> bool:
    mode = choice.get("type") if isinstance(choice, dict) else choice
    return bool(tools) or mode not in (None, "auto")


def _identity(tool: Mapping[str, JsonValue], wire: _Wire) -> _Identity | None:
    kind = tool.get("type", "function")
    if wire == "messages":
        kind = "function"
    if kind not in ("function", "custom"):
        return None
    nested = tool.get(str(kind))
    source = nested if isinstance(nested, dict) else tool
    name, namespace = source.get("name"), tool.get("namespace")
    if not isinstance(name, str):
        return None
    return str(kind), name, namespace if isinstance(namespace, str) else None


@dataclass(frozen=True, slots=True)
class ToolCallScope:
    identities: frozenset[_Identity]

    @classmethod
    def from_body(cls, body: Mapping[str, JsonValue], wire: _Wire) -> ToolCallScope:
        choice = body.get("tool_choice")
        mode = choice.get("type") if isinstance(choice, dict) else choice
        tools = body.get("tools")
        if mode == "none" or not isinstance(tools, list):
            return cls(frozenset())
        identities = {
            identity
            for tool in tools
            if isinstance(tool, dict)
            and (identity := _identity(tool, wire)) is not None
        }
        if isinstance(choice, dict) and mode in ("tool", "function", "custom"):
            selected = _identity(choice, wire)
            identities = {name for name in identities if name == selected}
        return cls(frozenset(identities))

    def require(self, kind: str, name: object, namespace: object = None) -> None:
        if (
            not isinstance(name, str)
            or not (namespace is None or isinstance(namespace, str))
            or (kind, name, namespace) not in self.identities
        ):
            raise ExecutionFailure(
                FailureKind.UPSTREAM,
                502,
                f"Upstream returned an undeclared or excluded tool for this turn: {name}",
                False,
            )
