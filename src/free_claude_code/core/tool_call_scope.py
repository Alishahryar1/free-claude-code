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


def has_chat_tool_selection(
    tools: object, choice: object, fields: Mapping[str, object]
) -> bool:
    return (
        has_tool_selection(tools, choice)
        or bool(fields.get("functions"))
        or fields.get("function_call") not in (None, "auto")
    )


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


def _identities(tools: JsonValue, wire: _Wire) -> set[_Identity]:
    if not isinstance(tools, list):
        return set()
    return {
        identity
        for tool in tools
        if isinstance(tool, dict) and (identity := _identity(tool, wire)) is not None
    }


@dataclass(frozen=True, slots=True)
class ToolCallScope:
    # An open translated request can return an ordinary, unqualified function.
    identities: frozenset[_Identity] | None

    @classmethod
    def from_body(cls, body: Mapping[str, JsonValue], wire: _Wire) -> ToolCallScope:
        choice = body.get("tool_choice")
        mode = choice.get("type") if isinstance(choice, dict) else choice
        if mode == "none" or (wire == "chat" and body.get("function_call") == "none"):
            return cls(frozenset())
        identities = _identities(body.get("tools"), wire)
        if isinstance(choice, dict) and mode in ("tool", "function", "custom"):
            selected = _identity(choice, wire)
            identities = {name for name in identities if name == selected}
        elif wire == "chat" and isinstance(choice, dict) and mode == "allowed_tools":
            allowed = choice.get("allowed_tools")
            entries = allowed.get("tools") if isinstance(allowed, dict) else None
            identities.intersection_update(_identities(entries, wire))
        if wire == "chat" and isinstance(legacy := body.get("function_call"), dict):
            identities = {
                identity
                for identity in identities
                if identity == ("function", legacy.get("name"), None)
            }
        return cls(frozenset(identities))

    def require(self, kind: str, name: object, namespace: object = None) -> None:
        if (
            isinstance(name, str)
            and name
            and (namespace is None or isinstance(namespace, str))
        ):
            if self.identities is None:
                if kind == "function" and namespace is None:
                    return
            elif (kind, name, namespace) in self.identities:
                return
        raise ExecutionFailure(
            FailureKind.UPSTREAM,
            502,
            f"Upstream returned an undeclared or excluded tool for this turn: {name}",
            False,
        )
