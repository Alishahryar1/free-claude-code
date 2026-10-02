"""Validation of complete client tool inputs."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..models import MessagesRequest


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Tool schema resolved from the original Anthropic request."""

    name: str
    input_schema: dict[str, Any]


def tool_schemas_by_name(request: MessagesRequest) -> dict[str, ToolSchema]:
    """Return Anthropic tool input schemas keyed by tool name."""
    schemas: dict[str, ToolSchema] = {}
    tools = request.tools
    if not tools:
        return schemas

    for tool in tools:
        name = tool.name
        if not name:
            continue
        schema = tool.input_schema
        if schema is None:
            schema = {"type": "object"}
        schemas[name] = ToolSchema(name=name, input_schema=deepcopy(schema))
    return schemas
