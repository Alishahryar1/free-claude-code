"""Pure, request-local projection of reasoning retained by the native harness."""

import base64
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Literal, cast

from .json_types import JsonObject, JsonValue
from .openai_chat import ChatToolResultImages

type HistoryProtocol = Literal["responses", "messages", "chat"]
_PREFIX = "fcc:history:v1:"
_LEGACY_PREFIX = "fcc:anthropic-reasoning:v1:"


class HistoryReplayError(ValueError):
    """A persisted replay record cannot be safely decoded or projected."""


class HistoryScope(StrEnum):
    """Documented readable reasoning reuse, separate from its wire encoding."""

    UNKNOWN = "unknown"
    TOOL_CONTINUATION = "tool_continuation"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class ReplayOrigin:
    provider: str
    protocol: HistoryProtocol
    endpoint: str
    connection: str
    model: str

    def accepts(self, source: ReplayOrigin) -> bool:
        # The upstream owns model/key/prefix compatibility within this domain.
        return (
            self.provider == source.provider
            and self.protocol == source.protocol
            and (not source.endpoint or self.endpoint == source.endpoint)
            and (not source.connection or self.connection == source.connection)
        )


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    origin: ReplayOrigin
    native: JsonObject


def _json_copy(value: object) -> JsonObject:
    if not isinstance(value, Mapping):
        raise HistoryReplayError("History replay must contain a native object.")
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 128:
            raise HistoryReplayError("History replay is too deeply nested.")
        if isinstance(item, Mapping):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    try:
        return cast(
            JsonObject,
            json.loads(
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            ),
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise HistoryReplayError(
            "History replay requires finite, representable JSON."
        ) from exc


def is_replay(value: object) -> bool:
    return isinstance(value, str) and value.startswith(
        ("fcc:history:", "fcc:anthropic-reasoning:")
    )


def encode_replay(record: ReplayRecord) -> str:
    """Preserve native state, without claiming to encrypt or authenticate it."""
    payload = _json_copy({"origin": asdict(record.origin), "native": record.native})
    _validate_record(payload)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _validate_record(payload: JsonObject) -> ReplayRecord:
    origin, native = payload.get("origin"), payload.get("native")
    if (
        set(payload) != {"origin", "native"}
        or not isinstance(origin, dict)
        or set(origin) != {"provider", "protocol", "endpoint", "connection", "model"}
        or not all(isinstance(value, str) for value in origin.values())
        or not origin["provider"]
        or not origin["model"]
        or origin["protocol"] not in {"responses", "messages", "chat"}
        or not isinstance(native, dict)
    ):
        raise HistoryReplayError(
            "History replay has invalid provenance or native data."
        )
    protocol = cast(HistoryProtocol, origin["protocol"])
    if protocol == "responses" and native.get("type") != "reasoning":
        raise HistoryReplayError("Responses replay requires a reasoning item.")
    if protocol == "messages":
        kind = native.get("type")
        if not (
            (
                kind == "thinking"
                and isinstance(native.get("thinking"), str)
                and isinstance(native.get("signature"), str)
                and native["signature"]
            )
            or (
                kind == "redacted_thinking"
                and isinstance(native.get("data"), str)
                and native["data"]
            )
        ):
            raise HistoryReplayError(
                "Messages replay requires signed thinking or redacted data."
            )
    if protocol == "chat" and not isinstance(native.get("reasoning_details"), list):
        raise HistoryReplayError("Chat replay requires structured reasoning details.")
    return ReplayRecord(
        ReplayOrigin(
            str(origin["provider"]),
            protocol,
            str(origin["endpoint"]),
            str(origin["connection"]),
            str(origin["model"]),
        ),
        native,
    )


def decode_replay(value: str) -> ReplayRecord:
    prefix = next(
        (prefix for prefix in (_PREFIX, _LEGACY_PREFIX) if value.startswith(prefix)),
        None,
    )
    if prefix is None:
        raise HistoryReplayError("Unsupported history replay version.")
    try:
        encoded = value[len(prefix) :]
        payload = _json_copy(
            json.loads(
                base64.b64decode(
                    encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
                ).decode("utf-8")
            )
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HistoryReplayError("Malformed history replay.") from exc
    if prefix == _LEGACY_PREFIX:
        if (
            set(payload) != {"protocol", "replay_scope", "source_model", "block"}
            or payload.get("protocol") != "anthropic_messages"
        ):
            raise HistoryReplayError("Malformed legacy Messages reasoning replay.")
        payload = {
            "origin": {
                "provider": payload["replay_scope"],
                "protocol": "messages",
                "endpoint": "",
                "connection": "",
                "model": payload["source_model"],
            },
            "native": payload["block"],
        }
    return _validate_record(payload)


def readable_reasoning(item: Mapping[str, JsonValue]) -> list[tuple[str, bool]]:
    """Keep distinct readable representations and their summary/full-text meaning."""
    if item.get("type") == "thinking":
        text = str(item.get("thinking") or "")
        return [(text, False)] if text else []
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, list):
            text = "\n\n".join(
                str(part["text"])
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            if text:
                return [(text, key == "summary")]
    details = item.get("reasoning_details")
    parts: list[tuple[str, bool]] = []
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            kind = detail.get("type")
            if kind not in {"reasoning.text", "reasoning.summary"}:
                continue
            summary = kind == "reasoning.summary"
            text = (
                detail.get("summary", detail.get("text"))
                if summary
                else detail.get("text")
            )
            if isinstance(text, str) and text:
                parts.append((text, summary))
    # The native string and structured details can be alternate copies. Preserve
    # the typed version in that case, especially when the text is a summary.
    represented = {text for text, _ in parts}
    represented.add("".join(text for text, _ in parts))
    represented.add("\n\n".join(text for text, _ in parts))
    native_parts: list[tuple[str, bool]] = []
    for key in ("reasoning_content", "reasoning"):
        text = item.get(key)
        if isinstance(text, str) and text and text not in represented:
            native_parts.append((text, False))
            represented.add(text)
    return [*native_parts, *parts]


def has_readable_replay(value: object) -> bool:
    return (
        isinstance(value, str)
        and is_replay(value)
        and bool(readable_reasoning(decode_replay(value).native))
    )


def _replay_readable(
    item: Mapping[str, JsonValue], record: ReplayRecord | None
) -> list[tuple[str, bool]]:
    native = readable_reasoning(record.native) if record else []
    return native or readable_reasoning(item)


def reasoning_context(text: str, *, summary: bool = False) -> str:
    label = "Earlier reasoning summary" if summary else "Earlier reasoning"
    return f"[{label}]\n{text}" if text else ""


def tool_history_context(item: Mapping[str, JsonValue]) -> str:
    """Keep available tool data as quoted history, never as new instructions."""
    return "[Earlier tool record]\n" + json.dumps(
        dict(item), ensure_ascii=False, separators=(",", ":")
    )


def validate_hosted_tool_history(blocks: list[JsonObject]) -> None:
    """A foreign harness cannot finish an upstream-owned pending tool call."""
    results = {
        block.get("tool_use_id")
        for block in blocks
        if str(block.get("type", "")).endswith("_tool_result")
    }
    if any(
        block.get("type") == "server_tool_use" and block.get("id") not in results
        for block in blocks
    ):
        raise HistoryReplayError(
            "An active server tool cannot continue through a different protocol."
        )


def responses_replay_item(item: JsonObject, origin: ReplayOrigin) -> JsonObject:
    result = deepcopy(item)
    result["encrypted_content"] = encode_replay(ReplayRecord(origin, item))
    return result


def preserve_responses_reasoning(
    payload: JsonObject, origin: ReplayOrigin
) -> JsonObject:
    """Wrap complete native reasoning before a presenter rewrites public metadata."""
    result = deepcopy(payload)
    item = result.get("item")
    if isinstance(item, dict) and item.get("type") == "reasoning":
        result["item"] = responses_replay_item(item, origin)
    response = result.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            response["output"] = [
                responses_replay_item(item, origin)
                if isinstance(item, dict) and item.get("type") == "reasoning"
                else item
                for item in output
            ]
    return result


def reasoning_detail(value: str) -> list[JsonValue]:
    """Decode older Chat detail strings while retaining unknown opaque state."""
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return [{"type": "reasoning.encrypted", "data": value}]


def _record(item: Mapping[str, JsonValue], *keys: str) -> ReplayRecord | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and is_replay(value):
            return decode_replay(value)
    return None


def _context_item(text: str) -> JsonObject:
    return {"role": "assistant", "content": text}


def responses_reasoning_blocks(item: Mapping[str, JsonValue]) -> list[JsonValue]:
    """Carry Responses reasoning into Messages until destination preparation."""
    blocks: list[JsonValue] = []
    carrier = item.get("encrypted_content")
    if isinstance(carrier, str) and carrier:
        record = _record(item, "encrypted_content")
        if (
            record is not None
            and record.origin.protocol == "messages"
            and record.native.get("type") == "thinking"
        ):
            return [
                {
                    "type": "thinking",
                    "thinking": record.native["thinking"],
                    "signature": carrier,
                }
            ]
        blocks.append({"type": "redacted_thinking", "data": carrier})
        if record is not None and readable_reasoning(record.native):
            return blocks
    blocks.extend(
        {"type": "text", "text": reasoning_context(text, summary=summary)}
        for text, summary in readable_reasoning(item)
    )
    return blocks


def prepare_history(
    body: Mapping[str, JsonValue],
    destination: ReplayOrigin,
    *,
    scope: HistoryScope = HistoryScope.TOOL_CONTINUATION,
    reasoning_field: str = "reasoning_content",
    structured_details: bool = True,
) -> JsonObject:
    """Project a fresh wire copy. The caller's native transcript stays intact."""
    result = deepcopy(dict(body))
    if destination.protocol == "responses":
        items = result.get("input")
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return result
        projected: list[JsonValue] = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                projected.append(item)
                continue
            record = _record(item, "encrypted_content")
            if record is not None and destination.accepts(record.origin):
                projected.append(record.native)
                if not readable_reasoning(record.native):
                    projected.extend(
                        _context_item(reasoning_context(text, summary=summary))
                        for text, summary in readable_reasoning(item)
                    )
                continue
            if record is None and item.get("encrypted_content"):
                projected.append(
                    item
                )  # Older history: try native before precise recovery.
                continue
            for text, summary in _replay_readable(item, record):
                projected.append(
                    _context_item(reasoning_context(text, summary=summary))
                )
        result["input"] = projected
        return result
    messages = result.get("messages")
    if not isinstance(messages, list):
        return result
    last_user = max(
        (
            index
            for index, msg in enumerate(messages)
            if isinstance(msg, dict)
            and msg.get("role") == "user"
            and not _tool_result_message(msg)
        ),
        default=-1,
    )
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if destination.protocol == "messages":
            _prepare_messages_content(message, destination)
        else:
            _prepare_chat_content(
                message,
                destination,
                native_text=scope == HistoryScope.ALL
                or (
                    scope == HistoryScope.TOOL_CONTINUATION
                    and index > last_user
                    and any(
                        isinstance(later, dict) and later.get("role") == "tool"
                        for later in messages[index + 1 :]
                    )
                ),
                reasoning_field=reasoning_field,
                structured_details=structured_details,
            )
    result["messages"] = [
        message
        for message in messages
        if not (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and message.get("content") == []
        )
    ]
    return result


def _tool_result_message(message: JsonObject) -> bool:
    if isinstance(message, ChatToolResultImages):
        return True
    content = message.get("content")
    return (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
    )


def _prepare_messages_content(message: JsonObject, destination: ReplayOrigin) -> None:
    content = message.get("content")
    if not isinstance(content, list):
        return
    blocks: list[JsonValue] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in {
            "thinking",
            "redacted_thinking",
        }:
            blocks.append(block)
            continue
        record = _record(block, "signature", "data")
        if record is not None and destination.accepts(record.origin):
            blocks.append(record.native)
            if not readable_reasoning(record.native):
                blocks.extend(
                    {"type": "text", "text": reasoning_context(text, summary=summary)}
                    for text, summary in readable_reasoning(block)
                )
        elif record is None and (block.get("signature") or block.get("data")):
            blocks.append(block)
        else:
            for text, summary in _replay_readable(block, record):
                blocks.append(
                    {"type": "text", "text": reasoning_context(text, summary=summary)}
                )
    message["content"] = blocks


def _append_context(message: JsonObject, text: str) -> None:
    if not text:
        return
    content = message.get("content")
    if isinstance(content, list):
        content.append({"type": "text", "text": text})
    else:
        message["content"] = (
            f"{content}\n\n{text}"
            if isinstance(content, str) and content.strip()
            else text
        )


def _prepare_chat_content(
    message: JsonObject,
    destination: ReplayOrigin,
    *,
    native_text: bool,
    reasoning_field: str,
    structured_details: bool,
) -> None:
    had_reasoning_content = "reasoning_content" in message
    details = message.get("reasoning_details")
    restored: list[JsonValue] = []
    original_reasoning = message.get("reasoning_content", message.get("reasoning"))
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            record = _record(detail, "data", "signature")
            if record is None and structured_details:
                restored.append(detail)
                continue
            elif (
                record is not None
                and destination.accepts(record.origin)
                and structured_details
            ):
                native = record.native.get("reasoning_details")
                if isinstance(native, list):
                    restored.extend(native)
                parts = readable_reasoning(record.native)
                native_parts = readable_reasoning({"reasoning_details": native})
                parts = [part for part in parts if part not in native_parts]
            else:
                parts = readable_reasoning(
                    record.native if record else {"reasoning_details": [detail]}
                )
            for text, summary in parts:
                existing = message.get("reasoning_content", message.get("reasoning"))
                if text != original_reasoning:
                    if (
                        native_text
                        and not summary
                        and reasoning_field in {"reasoning_content", "reasoning"}
                    ):
                        message[reasoning_field] = (
                            f"{existing}\n{text}"
                            if isinstance(existing, str) and existing
                            else text
                        )
                    else:
                        _append_context(
                            message, reasoning_context(text, summary=summary)
                        )
    if restored:
        message["reasoning_details"] = restored
    else:
        message.pop("reasoning_details", None)
    if not native_text:
        for key in ("reasoning_content", "reasoning"):
            text = message.pop(key, None)
            if isinstance(text, str) and text:
                _append_context(message, reasoning_context(text))
    if (
        had_reasoning_content
        and reasoning_field == "reasoning_content"
        and message.get("tool_calls")
    ):
        message.setdefault("reasoning_content", "")
