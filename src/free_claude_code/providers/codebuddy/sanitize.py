"""Upstream request sanitization for CodeBuddy's WAF and schema validator.

CodeBuddy rejects requests whose system text carries another product's
channel fingerprints (``x-anthropic-billing-header``, ``cc_version=``,
``Claude Agent SDK``, ``You are Codex``) with HTTP 400 "Illegal API
invocation from an unapproved channel", and rejects tool JSON Schemas that
use ``$ref`` into ``allOf``-heavy ``$defs`` (or dangle) with HTTP 400
"Invalid request parameters". Both guards are implemented here so every
ingress protocol shares one neutralization pass.
"""

import re
from typing import Any

NEUTRAL_SYSTEM_PROMPT = "You are a helpful coding assistant running in a terminal."

_CHANNEL_FINGERPRINTS = re.compile(
    r"x-anthropic-billing-header|cc_version=|Claude Agent SDK|You are Codex",
    re.IGNORECASE,
)


def sanitize_system_text(text: str) -> str:
    """Replace a system prompt carrying foreign channel fingerprints."""

    if _CHANNEL_FINGERPRINTS.search(text):
        return NEUTRAL_SYSTEM_PROMPT
    return text


_SCHEMA_DROP_KEYS = frozenset(
    {
        "$defs",
        "$schema",
        "$id",
        "if",
        "then",
        "else",
        "not",
        "unevaluatedProperties",
        "dependentSchemas",
        "patternProperties",
        "contains",
        "propertyNames",
    }
)
_SCHEMA_MAX_DEPTH = 25


def sanitize_tool_parameters(parameters: Any) -> Any:
    """Inline local ``#/$defs/...`` refs, flatten ``allOf``, drop 2020-12 keys."""

    if not isinstance(parameters, dict):
        return parameters
    defs = parameters.get("$defs")
    defs = defs if isinstance(defs, dict) else None
    return _sanitize_schema_node(parameters, defs, 0)


def _sanitize_schema_node(node: Any, defs: dict[str, Any] | None, depth: int) -> Any:
    if isinstance(node, list):
        return [_sanitize_schema_node(item, defs, depth + 1) for item in node]
    if not isinstance(node, dict) or depth > _SCHEMA_MAX_DEPTH:
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _SCHEMA_DROP_KEYS:
            continue
        if key == "$ref":
            target = _local_def_target(value, defs)
            if target is not None:
                out = _merge_schema(out, _sanitize_schema_node(target, defs, depth + 1))
            # Dangling refs are dropped rather than rejected upstream.
            continue
        if key == "allOf" and isinstance(value, list):
            for sub in value:
                sanitized = _sanitize_schema_node(sub, defs, depth + 1)
                if isinstance(sanitized, dict):
                    out = _merge_schema(out, sanitized)
            continue
        out[key] = _sanitize_schema_node(value, defs, depth + 1)
    return out


def _local_def_target(ref: Any, defs: dict[str, Any] | None) -> Any | None:
    """Resolve a local ``#/$defs/<name>`` ref, returning None when dangling."""

    if not isinstance(ref, str) or defs is None:
        return None
    match = re.fullmatch(r"#/\$defs/(.+)", ref)
    if match is None:
        return None
    target = defs.get(match.group(1))
    return target if isinstance(target, dict) else None


def _merge_schema(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = {**base, **override}
    if "properties" in base or "properties" in override:
        out["properties"] = {
            **base.get("properties", {}),
            **override.get("properties", {}),
        }
    if "required" in base or "required" in override:
        out["required"] = sorted(
            {*base.get("required", []), *override.get("required", [])}
        )
    return out


def normalize_tool_choice(body: dict[str, Any]) -> None:
    """Coerce ``tool_choice`` to the plain string the upstream accepts.

    ``"none"`` also drops the tool declarations, matching the upstream
    proxy behavior for models that reject tool metadata entirely.
    """

    if "tool_choice" not in body:
        return
    tool_choice = body["tool_choice"]

    def drop_tools() -> None:
        body.pop("tool_choice", None)
        body.pop("tools", None)

    if isinstance(tool_choice, str):
        if tool_choice.lower() == "none":
            drop_tools()
        return
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type == "none":
            drop_tools()
            return
        if choice_type in {"auto", "required"}:
            body["tool_choice"] = choice_type
            return
        if choice_type == "function":
            function = tool_choice.get("function")
            name = (
                function.get("name")
                if isinstance(function, dict)
                else tool_choice.get("name")
            )
            body["tool_choice"] = str(name) if name else "auto"
            return
    body.pop("tool_choice", None)
