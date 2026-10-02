"""Validation of complete client tool inputs."""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jsonschema
from loguru import logger

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


def validate_tool_input(
    tool_name: str, parsed_input: dict[str, Any], schemas: dict[str, ToolSchema]
) -> bool:
    tool_schema = schemas.get(tool_name)
    if tool_schema is None:
        return True
    try:
        validator_cls = jsonschema.validators.validator_for(tool_schema.input_schema)
        validator_cls.check_schema(tool_schema.input_schema)
        validator_cls(tool_schema.input_schema).validate(parsed_input)
    except jsonschema.exceptions.SchemaError as exc:
        logger.warning("Skipping invalid tool schema for {}: {}", tool_name, exc)
        return True
    except jsonschema.exceptions.ValidationError:
        return False
    return True


def parse_complete_tool_input(
    raw_json: str, tool_name: str, schemas: dict[str, ToolSchema]
) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not validate_tool_input(tool_name, parsed, schemas):
        return None
    return parsed
