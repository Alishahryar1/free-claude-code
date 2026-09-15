"""Provider connection provenance and precise historical request corrections."""

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.history_replay import (
    HistoryProtocol,
    HistoryReplayError,
    ReplayOrigin,
    decode_replay,
    is_replay,
    readable_reasoning,
    reasoning_context,
)

from .endpoint_types import HttpEndpoint

if TYPE_CHECKING:
    from openai import AsyncOpenAI


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
    error: Exception, body: Mapping[str, Any], protocol: HistoryProtocol
) -> dict[str, Any] | None:
    """Repair one explicitly rejected historical record on an independent copy."""
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
        code = record.get("code", record.get("type"))
        invalid_id = bool(
            re.search(
                r"invalid.+(?:input|messages).+\.id.+(?:letters|characters|ID)",
                message,
                re.I,
            )
        )
        invalid_native = code in {
            "invalid_encrypted_content",
            "invalid_signature",
        } or bool(
            re.search(
                r"invalid signature in thinking block|referenced reasoning item .+ (?:not found|expired)",
                message,
                re.I,
            )
        )
        if not (invalid_id or invalid_native):
            continue
        matched = _rejected_record(record, candidates)
        if matched is None:
            continue
        path, original = matched
        result = deepcopy(dict(body))
        parent: Any = result
        for key in path[:-1]:
            parent = parent[key]
        if invalid_id:
            value = original.get("id")
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value):
                continue
            corrected = "rs_" + hashlib.sha256(value.encode()).hexdigest()[:24]
            parent[path[-1]]["id"] = corrected
            for item in result.get("input", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "item_reference"
                    and item.get("id") == value
                ):
                    item["id"] = corrected
        else:
            context = "\n\n".join(
                reasoning_context(text, summary=summary)
                for text, summary in readable_reasoning(
                    original
                    if protocol != "chat"
                    else {"reasoning_details": [original]}
                )
            )
            if protocol == "responses" and context:
                parent[path[-1]] = {"role": "assistant", "content": context}
            elif protocol == "messages" and context:
                parent[path[-1]] = {"type": "text", "text": context}
            else:
                del parent[path[-1]]
                if protocol == "chat":
                    assistant = result["messages"][path[1]]
                    if context:
                        assistant["content"] = (
                            (assistant.get("content") or "") + "\n\n" + context
                        )
                    if not parent:
                        assistant.pop("reasoning_details", None)
        return result
    return None


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
