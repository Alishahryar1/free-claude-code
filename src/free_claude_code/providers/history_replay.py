"""Provider connection provenance and precise historical request corrections."""

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.history_replay import (
    HistoryProtocol,
    HistoryReplayError,
    ReplayOrigin,
    decode_replay,
    is_replay,
    resolve_messages_replay,
)
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.recovery import CandidateIncompatible

from .endpoint_types import HttpEndpoint

if TYPE_CHECKING:
    from openai import AsyncOpenAI


_NATIVE_INPUT_FIELDS = {
    None: {"role", "content", "name", "url", "detail"},
    "message": {"role", "content", "status", "phase"},
    "text": {"text", "citations", "media_type", "data"},
    "input_text": {"text"},
    "output_text": {"text", "annotations", "logprobs"},
    "refusal": {"refusal"},
    "image": {"source"},
    "input_image": {"image_url", "file_id", "detail"},
    "image_url": {"image_url", "detail"},
    "document": {"source", "title", "context", "citations"},
    "input_file": {"file_id", "file_data", "filename", "file_url"},
    "file": {"file_data", "filename"},
    "thinking": {"thinking", "signature"},
    "redacted_thinking": {"data"},
    "reasoning": {"summary", "content", "encrypted_content", "status"},
    "summary_text": {"text"},
    "reasoning_text": {"text"},
    "tool_use": {"name", "input"},
    "tool_result": {"tool_use_id", "content", "is_error"},
    "function_call": {"call_id", "name", "arguments", "status", "namespace"},
    "function_call_output": {"call_id", "output", "status"},
    "custom_tool_call": {"call_id", "name", "input", "status", "namespace"},
    "custom_tool_call_output": {"call_id", "output", "status"},
    "tool_search_call": {"call_id", "execution", "arguments", "status"},
    "tool_search_output": {"call_id", "execution", "tools", "status"},
    "function": {"name", "arguments"},
    "base64": {"media_type", "data"},
    "url": {"url"},
}


def validate_history(body: Mapping[str, Any]) -> None:
    """Reject malformed persisted carriers before credentials or inference."""
    try:
        for protocol in ("responses", "messages", "chat"):
            for _, item in _reasoning_records(body, protocol):
                for key in ("encrypted_content", "signature", "data"):
                    value = item.get(key)
                    if isinstance(value, str) and is_replay(value):
                        decode_replay(value)
    except HistoryReplayError as error:
        raise InvalidRequestError(str(error)) from error


def requires_native_origin(body: Mapping[str, Any], protocol: HistoryProtocol) -> bool:
    """Retain the origin of opaque input, including state used before first output."""
    if protocol != "chat":
        fields = (
            set(OpenAIResponsesRequest.model_fields)
            | {
                "text",
                "include",
                "truncation",
                "background",
                "max_tool_calls",
                "service_tier",
                "safety_identifier",
                "user",
                "top_logprobs",
                "prompt_cache_key",
                "prompt_cache_retention",
                "stream_options",
            }
            if protocol == "responses"
            else set(MessagesRequest.model_fields) | {"service_tier"}
        )
        if any(key not in fields and value is not None for key, value in body.items()):
            return True
    if any(
        body.get(key)
        for key in ("previous_response_id", "conversation", "container", "prompt")
    ):
        return True
    for _, record in _reasoning_records(body, protocol):
        if any(record.get(key) for key in ("encrypted_content", "signature", "data")):
            return True
    pending: list[object] = [
        body.get("input" if protocol == "responses" else "messages")
    ]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, Mapping):
            kind = value.get("type")
            if kind not in _NATIVE_INPUT_FIELDS:
                return True
            if protocol != "chat" and set(value) - (
                _NATIVE_INPUT_FIELDS[kind] | {"type", "id", "cache_control"}
            ):
                return True
            if (
                any(
                    value.get(key)
                    for key in ("file_id", "container_id", "thought_signature")
                )
                or value.get("type") == "item_reference"
            ):
                return True
            # Tool arguments are user data. Their field names cannot establish
            # provider ownership. Follow only protocol-defined content paths.
            for key in (
                "content",
                "summary",
                "source",
                "image_url",
                "file",
                "tool_calls",
                "extra_content",
            ):
                nested = value.get(key)
                if isinstance(nested, list | Mapping):
                    pending.append(nested)
            if kind in {
                "function_call_output",
                "custom_tool_call_output",
            } and isinstance(value.get("output"), list):
                pending.append(value["output"])
    return False


def require_original_origin(
    body: Mapping[str, Any], protocol: HistoryProtocol, destination: ReplayOrigin
) -> None:
    """Check the accepted input before a converter can project native state away."""
    validate_history(body)
    for _, item in _reasoning_records(body, protocol):
        for key in ("encrypted_content", "signature", "data"):
            value = item.get(key)
            if not value:
                continue
            if isinstance(value, str) and is_replay(value):
                record = decode_replay(value)
                if not destination.accepts(record.origin):
                    raise CandidateIncompatible(
                        "Original native history belongs to another origin."
                    )
            elif destination.protocol != protocol:
                raise CandidateIncompatible(
                    "Original native history requires its source protocol."
                )


def normalize_messages_history(request: MessagesRequest) -> MessagesRequest:
    """Resolve persisted associations before any lossy protocol conversion."""
    validate_history(request.model_dump(mode="json"))
    messages = []
    try:
        for message in request.messages:
            if message.role != "assistant" or isinstance(message.content, str):
                messages.append(message)
                continue
            body = message.model_dump(mode="json")
            body["content"] = resolve_messages_replay(
                cast(list[JsonValue], body["content"])
            )
            messages.append(Message.model_validate(body))
    except HistoryReplayError as error:
        raise InvalidRequestError(str(error)) from error
    return request.model_copy(update={"messages": messages})


def replay_origin(
    provider: str,
    protocol: HistoryProtocol,
    model: str,
    *,
    client: AsyncOpenAI | None = None,
    endpoint: HttpEndpoint | None = None,
) -> ReplayOrigin:
    """Identify the actual connection without storing its credentials."""
    base_url = (
        endpoint.base_url
        if endpoint is not None
        else str(client.base_url)
        if client is not None
        else ""
    )
    headers: Mapping[str, object] = (
        endpoint.headers
        if endpoint is not None
        else client.default_headers
        if client is not None
        else {}
    )
    normalized_headers = {
        key.lower(): value for key, value in headers.items() if isinstance(value, str)
    }
    key = (
        endpoint.api_key
        if endpoint is not None
        else client.api_key
        if client is not None
        else None
    )
    account = endpoint.account_id if endpoint is not None else None
    credential = (
        "account:" + account
        if account
        else "key:" + key
        if isinstance(key, str) and key
        else "auth:"
        + normalized_headers.get(
            "authorization", normalized_headers.get("x-api-key", "")
        )
    )
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host += f":{parsed.port}"
    address = urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    return ReplayOrigin(
        provider,
        protocol,
        address,
        hashlib.sha256(credential.encode()).hexdigest(),
        model,
    )


def history_retry_body(
    error: Exception,
    body: Mapping[str, Any],
    protocol: HistoryProtocol,
    *,
    normalized_ids: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Normalize an explicitly rejected identifier without discarding history."""
    detail = extract_upstream_error_detail(error)
    if detail.status_code not in {None, 200, 400, 422}:
        return None
    try:
        payload = json.loads(detail.body_text or "")
    except ValueError:
        payload = {
            "message": detail.exception_text or str(error),
            "code": getattr(error, "code", None),
        }
    candidates = list(_reasoning_records(body, protocol))
    for record in _error_records(payload):
        message = str(record.get("message", ""))
        invalid_id = bool(
            re.search(
                r"invalid.+(?:input|messages).+\.id.+(?:letters|characters|ID)",
                message,
                re.I,
            )
        )
        if not invalid_id:
            continue
        matched = _rejected_record(record, candidates)
        if matched is None:
            continue
        path, original = matched
        result = deepcopy(dict(body))
        parent: Any = result
        for key in path[:-1]:
            parent = parent[key]
        value = original.get("id")
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value):
            continue
        corrected = "rs_" + hashlib.sha256(value.encode()).hexdigest()[:24]
        if normalized_ids is not None:
            normalized_ids[value] = corrected
        parent[path[-1]]["id"] = corrected
        for item in result.get("input", []):
            if (
                isinstance(item, dict)
                and item.get("type") == "item_reference"
                and item.get("id") == value
            ):
                item["id"] = corrected
        return result
    return None


def reapply_history_ids(
    body: dict[str, Any], protocol: HistoryProtocol, identifiers: Mapping[str, str]
) -> None:
    """Retain accepted identifier normalization on a fresh continuation copy."""
    for _, record in _reasoning_records(body, protocol):
        value = record.get("id")
        if isinstance(record, dict) and isinstance(value, str) and value in identifiers:
            record["id"] = identifiers[value]
    for item in body.get("input", []):
        if isinstance(item, dict) and item.get("type") == "item_reference":
            value = item.get("id")
            if isinstance(value, str) and value in identifiers:
                item["id"] = identifiers[value]


def _reasoning_records(
    body: Mapping[str, Any], protocol: HistoryProtocol
) -> Iterator[tuple[tuple[str | int, ...], Mapping[str, Any]]]:
    if protocol == "responses":
        items = body.get("input")
        if isinstance(items, Mapping):
            items = [items]
        for index, item in enumerate(items if isinstance(items, list) else []):
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                yield ("input", index), item
        return
    for index, message in enumerate(body.get("messages", [])):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        field = "content" if protocol == "messages" else "reasoning_details"
        blocks = message.get(field)
        if not isinstance(blocks, list):
            continue
        for block_index, block in enumerate(blocks):
            if isinstance(block, Mapping) and (
                protocol == "chat"
                or block.get("type") in {"thinking", "redacted_thinking"}
            ):
                yield ("messages", index, field, block_index), block


def _error_records(payload: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("message"), str):
            yield payload
        for key in ("error", "errors", "metadata", "raw", "previous_errors"):
            nested = payload.get(key)
            if key == "raw" and isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except ValueError:
                    continue
            yield from _error_records(nested)
    elif isinstance(payload, list):
        for item in payload:
            yield from _error_records(item)


def _rejected_record(
    error: Mapping[str, Any],
    candidates: list[tuple[tuple[str | int, ...], Mapping[str, Any]]],
) -> tuple[tuple[str | int, ...], Mapping[str, Any]] | None:
    location = str(error.get("param") or error.get("loc") or error.get("message") or "")
    match = re.search(
        r"(?:input|messages)(?:\[\d+\]|\.\d+)(?:(?:\.content|\.reasoning_details)(?:\[\d+\]|\.\d+))?",
        location,
    )
    if match:
        path = tuple(
            int(part) if part.isdigit() else part
            for part in re.findall(r"[a-z_]+|\d+", match[0])
        )
        return next((entry for entry in candidates if entry[0] == path), None)
    message = str(error.get("message", ""))
    identified = [
        entry
        for entry in candidates
        if isinstance(entry[1].get("id"), str) and f"'{entry[1]['id']}'" in message
    ]
    if len(identified) == 1:
        return identified[0]
    opaque = [
        entry
        for entry in candidates
        if any(entry[1].get(key) for key in ("encrypted_content", "signature", "data"))
    ]
    if len(opaque) == 1:
        return opaque[0]
    return None
