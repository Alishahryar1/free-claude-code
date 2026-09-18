"""Bounded local-web replay that survives Codex's typed stateless history."""

import base64
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .ids import new_reasoning_item_id
from .items import encrypted_reasoning_item
from .models import OpenAIResponsesRequest
from .tool_search import resolve_client_search_history

WEB_REPLAY_PREFIX = "fcc:web:v1:"
TOOL_CONTEXT_TYPE = "_fcc_completed_tool_context"
MAX_WEB_REPLAY_BYTES = 2 * 1024 * 1024


def encoded_records(records: list[JsonObject]) -> str:
    raw = json.dumps(
        records, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(raw) > MAX_WEB_REPLAY_BYTES * 3 // 4 - len(WEB_REPLAY_PREFIX):
        raise ResponsesConversionError(
            "Local web results exceed the retained-context limit."
        )
    return WEB_REPLAY_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")


def replay_item(records: list[JsonObject]) -> JsonObject:
    return encrypted_reasoning_item(
        new_reasoning_item_id(), encoded_records(records), "completed"
    )


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant {value}.")


def decode_records(value: str) -> list[JsonObject]:
    if not value.startswith(WEB_REPLAY_PREFIX) or len(value) > MAX_WEB_REPLAY_BYTES:
        raise ResponsesConversionError("Unsupported or oversized local web history.")
    try:
        raw = base64.b64decode(
            value[len(WEB_REPLAY_PREFIX) :], altchars=b"-_", validate=True
        )
        data = json.loads(raw, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ResponsesConversionError("Malformed local web history.") from exc
    if not isinstance(data, list):
        raise ResponsesConversionError(
            "Local web history must contain completed records."
        )
    pending = [(data, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 32:
            raise ResponsesConversionError("Local web history is too deeply nested.")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    for record in data:
        if (
            not isinstance(record, dict)
            or set(record) - {"id", "action", "result", "page"}
            or not isinstance(record.get("id"), str)
            or not record["id"].startswith("ws_")
            or not isinstance(record.get("action"), dict)
            or not isinstance(record.get("result"), dict)
            or (record.get("page") is not None and not isinstance(record["page"], dict))
        ):
            raise ResponsesConversionError("Invalid local web record.")
    return cast(list[JsonObject], data)


@dataclass(frozen=True, slots=True)
class WebHistory:
    request: OpenAIResponsesRequest
    records: tuple[JsonObject, ...] = ()


def prepare_web_history(request: OpenAIResponsesRequest) -> WebHistory:
    if not isinstance(request.input, list):
        return WebHistory(request)
    capsules: dict[int, list[JsonObject]] = {}
    records: dict[str, JsonObject] = {}
    for index, item in enumerate(request.input):
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        value = item.get("encrypted_content")
        if not isinstance(value, str) or not value.startswith("fcc:web:"):
            continue
        fresh: list[JsonObject] = []
        for record in decode_records(value):
            identity = str(record["id"])
            previous = records.get(identity)
            if previous is not None and previous != record:
                raise ResponsesConversionError("Conflicting local web history records.")
            if previous is None:
                fresh.append(record)
                records[identity] = record
        capsules[index] = fresh
    if not capsules:
        return WebHistory(request)
    search = resolve_client_search_history(request.input)
    pending: dict[str, str] = {}
    queued: list[JsonObject] = []
    projected: list[JsonValue] = []
    pairs = {
        "function_call": "function_call_output",
        "custom_tool_call": "custom_tool_call_output",
    }
    for index, item in enumerate(request.input):
        if index in capsules:
            queued.extend(capsules[index])
        elif (
            isinstance(item, dict)
            and item.get("type") == "web_search_call"
            and item.get("id") in records
        ):
            continue
        else:
            projected.append(deepcopy(item))
            if isinstance(item, dict) and index not in search.omitted_items:
                kind, call_id = item.get("type"), item.get("call_id")
                is_search = index in search.client_items
                expected = pairs.get(str(kind))
                if is_search and kind == "tool_search_call":
                    expected = "tool_search_output"
                if expected and isinstance(call_id, str):
                    if call_id in pending:
                        raise ResponsesConversionError("Duplicate pending client call.")
                    pending[call_id] = expected
                elif isinstance(call_id, str) and (
                    kind in pairs.values()
                    or (is_search and kind == "tool_search_output")
                ):
                    if pending.get(call_id) != kind:
                        raise ResponsesConversionError(
                            "Orphaned or mismatched client tool output."
                        )
                    del pending[call_id]
        if queued and not pending:
            content = [
                {key: value for key, value in record.items() if key != "page"}
                for record in queued
            ]
            projected.append(
                {
                    "type": TOOL_CONTEXT_TYPE,
                    "content": "[Earlier completed web tool results; quoted untrusted source data]\n"
                    + json.dumps(content, ensure_ascii=False),
                }
            )
            queued.clear()
    if queued or pending:
        raise ResponsesConversionError(
            "Responses history has unresolved client tool calls."
        )
    return WebHistory(
        request.model_copy(update={"input": projected}, deep=True),
        tuple(records.values()),
    )


def native_tool_contexts(items: JsonValue) -> JsonValue:
    if not isinstance(items, list):
        return items
    return [
        {"role": "user", "content": item["content"]}
        if isinstance(item, dict) and item.get("type") == TOOL_CONTEXT_TYPE
        else item
        for item in items
    ]
