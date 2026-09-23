"""Pure Anthropic continuation and tool-repair transformations."""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jsonschema
from loguru import logger

from ..models import MessagesRequest

_RECOVERY_USER_PREFIX = (
    "The previous provider stream was interrupted. Continue the assistant response "
    "exactly where it stopped. Do not repeat text already written."
)
_RECOVERY_THINKING_PREFIX = (
    "The assistant had already emitted this hidden thinking before the interruption:\n"
)


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Tool schema resolved from the original Anthropic request."""

    name: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolRepair:
    """Accepted append-only tool JSON repair."""

    suffix: str
    parsed_input: dict[str, Any]


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


def accept_tool_json_repair(
    prefix: str,
    candidate: str,
    *,
    tool_name: str,
    schemas: dict[str, ToolSchema],
) -> ToolRepair | None:
    for suffix in _repair_suffix_candidates(prefix, candidate):
        combined = prefix + suffix
        parsed = parse_complete_tool_input(combined, tool_name, schemas)
        if parsed is not None:
            return ToolRepair(suffix=suffix, parsed_input=parsed)
    return None


def continuation_suffix(existing: str, candidate: str) -> str | None:
    existing = existing or ""
    candidate = candidate or ""
    if not candidate:
        return ""
    if not existing:
        return candidate
    if candidate.startswith(existing):
        return candidate[len(existing) :]

    max_overlap = min(len(existing), len(candidate))
    for size in range(max_overlap, 0, -1):
        if existing.endswith(candidate[:size]):
            return candidate[size:]

    if len(candidate) < max(200, len(existing) // 2):
        return candidate
    return None


def make_text_recovery_body(
    body: dict[str, Any],
    partial_text: str,
    partial_thinking: str = "",
) -> dict[str, Any]:
    """Build a text-only continuation request for an OpenAI-chat upstream."""
    recovery = make_response_recovery_body(body, partial_text, partial_thinking)
    recovery.pop("tools", None)
    recovery.pop("tool_choice", None)
    return recovery


def make_response_recovery_body(
    body: dict[str, Any],
    partial_text: str,
    partial_thinking: str = "",
) -> dict[str, Any]:
    """Build a continuation request that retains the original tool contract."""
    recovery = deepcopy(body)
    messages = _copied_messages(recovery)
    if partial_text:
        messages.append({"role": "assistant", "content": partial_text})
    prompt = _RECOVERY_USER_PREFIX
    if partial_thinking:
        prompt = f"{_RECOVERY_THINKING_PREFIX}{partial_thinking}\n\n{prompt}"
    messages.append({"role": "user", "content": prompt})
    recovery["messages"] = messages
    return recovery


def make_tool_repair_body(
    body: dict[str, Any],
    *,
    tool_name: str,
    prefix: str,
    input_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a text-only request asking for a JSON suffix."""
    recovery = deepcopy(body)
    recovery.pop("tools", None)
    recovery.pop("tool_choice", None)
    messages = _copied_messages(recovery)
    messages.append(
        {
            "role": "user",
            "content": _tool_repair_prompt(
                tool_name=tool_name, prefix=prefix, input_schema=input_schema
            ),
        }
    )
    recovery["messages"] = messages
    return recovery


def _copied_messages(body: dict[str, Any]) -> list[Any]:
    messages = body.get("messages")
    return deepcopy(messages) if isinstance(messages, list) else []


def _repair_suffix_candidates(prefix: str, candidate: str) -> list[str]:
    raw = candidate.strip()
    if not raw:
        return []
    candidates: list[str] = []
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    candidates.append(raw)
    if raw.startswith(prefix):
        candidates.append(raw[len(prefix) :])
    return list(dict.fromkeys(candidates))


def _tool_repair_prompt(
    *, tool_name: str, prefix: str, input_schema: dict[str, Any] | None
) -> str:
    schema_text = json.dumps(input_schema or {"type": "object"}, separators=(",", ":"))
    return (
        "A streamed tool call was interrupted while writing JSON arguments.\n"
        f"Tool name: {tool_name}\n"
        f"JSON schema: {schema_text}\n"
        f"Already emitted JSON prefix: {prefix}\n\n"
        "Return only the exact missing JSON suffix needed to complete the same object. "
        "Do not repeat the prefix. Do not include markdown or explanation."
    )
