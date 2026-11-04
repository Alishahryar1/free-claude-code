"""Pure, request-local projection of reasoning retained by the native harness."""

import base64
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Literal, cast

from .json_types import JsonObject, JsonValue
from .openai_chat import ChatToolResultImages

type HistoryProtocol = Literal["responses", "messages", "chat"]
_PREFIX = "fcc:history:v1:"
_ASSOCIATED_PREFIX = "fcc:history:v2:"
_LEGACY_PREFIX = "fcc:anthropic-reasoning:v1:"


class HistoryReplayError(ValueError):
    """A persisted replay record cannot be safely decoded or projected."""


class HistoryMode(StrEnum):
    """Acceptance-tested retention of independently editable reasoning."""

    ALL = "all"
    CURRENT = "current"
    NONE = "none"

    def includes(self, index: int, last_user: int) -> bool:
        return self is HistoryMode.ALL or (
            self is HistoryMode.CURRENT and index > last_user
        )


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


@dataclass(frozen=True, slots=True)
class AssociatedReplayRecord(ReplayRecord):
    group_id: str
    part: Literal["anchor", "final"]


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
    associated = isinstance(record, AssociatedReplayRecord)
    if associated:
        payload.update(group_id=record.group_id, part=record.part)
    _validate_record(payload, associated=associated)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    prefix = _ASSOCIATED_PREFIX if associated else _PREFIX
    return prefix + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _validate_record(payload: JsonObject, *, associated: bool = False) -> ReplayRecord:
    origin, native = payload.get("origin"), payload.get("native")
    fields = (
        {"origin", "native", "group_id", "part"} if associated else {"origin", "native"}
    )
    if (
        set(payload) != fields
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
    if protocol == "chat":
        fields = {"reasoning_details": list, "reasoning_content": str, "reasoning": str}
        if not any(key in native for key in fields) or any(
            key in native and not isinstance(native[key], kind)
            for key, kind in fields.items()
        ):
            raise HistoryReplayError("Chat replay requires native reasoning fields.")
    record = ReplayRecord(
        ReplayOrigin(
            str(origin["provider"]),
            protocol,
            str(origin["endpoint"]),
            str(origin["connection"]),
            str(origin["model"]),
        ),
        native,
    )
    if associated:
        group_id, part = payload.get("group_id"), payload.get("part")
        if (
            protocol != "chat"
            or not isinstance(group_id, str)
            or not group_id
            or not isinstance(part, str)
            or part not in {"anchor", "final"}
        ):
            raise HistoryReplayError("Invalid reasoning replay association.")
        return AssociatedReplayRecord(
            record.origin,
            record.native,
            group_id,
            part,
        )
    return record


def decode_replay(value: str) -> ReplayRecord:
    prefix = next(
        (
            prefix
            for prefix in (_PREFIX, _ASSOCIATED_PREFIX, _LEGACY_PREFIX)
            if value.startswith(prefix)
        ),
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
    return _validate_record(payload, associated=prefix == _ASSOCIATED_PREFIX)


def resolve_messages_replay(content: Sequence[JsonValue]) -> list[JsonValue]:
    """Join late metadata within one assistant message on a fresh copy."""
    blocks = deepcopy(list(content))
    groups: dict[str, dict[str, tuple[int, str, AssociatedReplayRecord]]] = {}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        key = (
            "signature"
            if block.get("type") == "thinking"
            else "data"
            if block.get("type") == "redacted_thinking"
            else None
        )
        if key is None or not is_replay(value := block.get(key)):
            continue
        record = decode_replay(cast(str, value))
        if not isinstance(record, AssociatedReplayRecord):
            continue
        group = groups.setdefault(record.group_id, {})
        if record.part in group or any(
            previous.origin != record.origin for _, _, previous in group.values()
        ):
            raise HistoryReplayError("Conflicting reasoning replay association.")
        if record.part == "final" and key != "data":
            raise HistoryReplayError(
                "Final reasoning replay requires a redacted carrier."
            )
        group[record.part] = (index, key, record)
    removed: set[int] = set()
    for group in groups.values():
        anchor, final = group.get("anchor"), group.get("final")
        if anchor and final and anchor[0] >= final[0]:
            raise HistoryReplayError("Reasoning replay final precedes its anchor.")
        target = anchor or final
        assert target is not None
        index, key, record = target
        if final:
            record = final[2]
        block = blocks[index]
        assert isinstance(block, dict)
        block[key] = encode_replay(ReplayRecord(record.origin, record.native))
        if anchor and final:
            removed.add(final[0])
    return [block for index, block in enumerate(blocks) if index not in removed]


def readable_reasoning(item: Mapping[str, JsonValue]) -> list[tuple[str, bool]]:
    """Keep distinct readable representations and their summary/full-text meaning."""
    if item.get("type") == "thinking":
        text = str(item.get("thinking") or "")
        return [(text, False)] if text else []
    fields: list[tuple[str, bool]] = []
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, list):
            text = "\n\n".join(
                str(part["text"])
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            if text:
                fields.append((text, key == "summary"))
    if fields:
        return fields
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
    for summary in (None, False, True):
        texts = [
            text
            for text, is_summary in parts
            if summary is None or summary == is_summary
        ]
        represented.update(("".join(texts), "\n\n".join(texts)))
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


def unprotected_replay(value: object) -> ReplayRecord | None:
    """Identify local carriers whose native records contain editable reasoning."""
    if not isinstance(value, str) or not is_replay(value):
        return None
    try:
        record = decode_replay(value)
    except HistoryReplayError:
        return None
    if protected_reasoning(record.native):
        return None
    return record


def _replay_readable(
    item: Mapping[str, JsonValue], record: ReplayRecord | None
) -> list[tuple[str, bool]]:
    native = readable_reasoning(record.native) if record else []
    return native or readable_reasoning(item)


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
        if record is None:
            readable = readable_reasoning(item)
            if readable or item.get("content"):
                return [
                    {
                        "type": "thinking",
                        "thinking": "\n\n".join(text for text, _ in readable),
                        "signature": carrier,
                    }
                ]
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
        {"type": "thinking", "thinking": text}
        for text, summary in readable_reasoning(item)
        if not summary
    )
    return blocks


type HistoryPath = tuple[str | int, ...]


def protected_reasoning(item: Mapping[str, JsonValue]) -> bool:
    """Local carriers do not turn unsigned readable data into protected state."""
    record = _record(item, "encrypted_content", "signature", "data")
    if record is not None:
        return protected_reasoning(record.native)
    if item.get("encrypted_content") or item.get("signature"):
        return True
    if item.get("type") in {"redacted_thinking", "reasoning.encrypted"} and item.get(
        "data"
    ):
        return True
    if item.get("extra_content"):
        return True
    details = item.get("reasoning_details")
    return isinstance(details, list) and any(
        isinstance(detail, dict) and protected_reasoning(detail) for detail in details
    )


def normalize_history_source(
    body: Mapping[str, JsonValue], protocol: HistoryProtocol
) -> JsonObject:
    """Own the input shape used by both projection paths and corrections."""
    result = deepcopy(dict(body))
    if protocol == "responses" and isinstance(result.get("input"), dict):
        result["input"] = [result["input"]]
    return result


def prepare_history(
    body: Mapping[str, JsonValue],
    destination: ReplayOrigin,
    *,
    scope: HistoryMode = HistoryMode.ALL,
    reasoning_field: str = "reasoning_content",
    structured_details: bool = True,
    source_paths: dict[HistoryPath, HistoryPath] | None = None,
    continuation: bool = False,
) -> JsonObject:
    """Restore native records and project readable history on a fresh wire copy.

    Source paths identify original records for precise corrections after filtering.
    They are request-local metadata, never fields in the outgoing JSON.
    """
    result = normalize_history_source(body, destination.protocol)
    paths = source_paths if source_paths is not None else {}
    paths.clear()
    if destination.protocol == "responses":
        _prepare_response_history(result, destination, scope, paths, continuation)
        return result
    messages = result.get("messages")
    if not isinstance(messages, list):
        return result
    last_user = _last_user(messages[:-1] if continuation else messages)
    projected: list[JsonValue] = []
    for index, message in enumerate(messages):
        output_index = len(projected)
        source_path = ("messages", index)
        output_path = ("messages", output_index)
        if not isinstance(message, dict) or message.get("role") != "assistant":
            projected.append(message)
            paths[output_path] = source_path
            continue
        include = scope.includes(index, last_user)
        if destination.protocol == "messages":
            _prepare_messages_content(
                message, destination, include, output_path, source_path, paths
            )
        else:
            _prepare_chat_content(
                message,
                destination,
                include,
                reasoning_field,
                structured_details,
                output_path,
                source_path,
                paths,
            )
        if (
            message.get("content") == []
            and not any(
                message.get(key)
                for key in ("tool_calls", "reasoning_details", "extra_content")
            )
            and not any(key in message for key in ("reasoning_content", "reasoning"))
        ):
            continue
        paths[output_path] = source_path
        projected.append(message)
    result["messages"] = projected
    return result


def _last_user(messages: Sequence[JsonValue]) -> int:
    return max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict)
            and message.get("role") == "user"
            and not _tool_result_message(message)
        ),
        default=-1,
    )


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


def _unsigned_response(parts: list[tuple[str, bool]]) -> JsonObject:
    return {
        "type": "reasoning",
        "content": [
            {"type": "reasoning_text", "text": text}
            for text, summary in parts
            if not summary
        ],
        "summary": [
            {"type": "summary_text", "text": text} for text, summary in parts if summary
        ],
    }


def _response_assistant(item: JsonValue) -> bool:
    return isinstance(item, dict) and (
        item.get("role") == "assistant"
        or item.get("type") in {"reasoning", "function_call", "custom_tool_call"}
    )


def _prepare_response_history(
    body: JsonObject,
    destination: ReplayOrigin,
    scope: HistoryMode,
    paths: dict[HistoryPath, HistoryPath],
    continuation: bool,
) -> None:
    items = body.get("input")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return
    last_user = _last_user(items[:-1] if continuation else items)
    projected: list[JsonValue] = []
    index = 0
    while index < len(items):
        if not _response_assistant(items[index]):
            paths[("input", len(projected))] = ("input", index)
            projected.append(items[index])
            index += 1
            continue
        end = index + 1
        while end < len(items) and _response_assistant(items[end]):
            end += 1
        group = items[index:end]
        has_reasoning = any(
            isinstance(item, dict) and item.get("type") == "reasoning" for item in group
        )
        include = scope.includes(index, last_user)
        if include and not has_reasoning:
            projected.append(_unsigned_response([("", False)]))
        for source_index in range(index, end):
            item = items[source_index]
            assert isinstance(item, dict)
            values: list[JsonValue] = []
            if item.get("type") != "reasoning":
                values.append(item)
            else:
                record = _record(item, "encrypted_content")
                protected = protected_reasoning(item)
                if protected:
                    # Preserve the pre-existing opaque compatibility contract.
                    if record is not None and destination.accepts(record.origin):
                        values.append(record.native)
                        if not readable_reasoning(record.native):
                            values.extend(
                                _context_item(text)
                                for text, _ in readable_reasoning(item)
                            )
                    elif record is None:
                        values.append(item)
                    else:
                        values.extend(
                            _context_item(text)
                            for text, _ in _replay_readable(item, record)
                        )
                elif include:
                    if record is None:
                        values.append(item)
                    elif record.origin.protocol == "responses":
                        values.append(record.native)
                    else:
                        values.append(
                            _unsigned_response(readable_reasoning(record.native))
                        )
            for value in values:
                paths[("input", len(projected))] = ("input", source_index)
                projected.append(value)
        index = end
    body["input"] = projected


def _prepare_messages_content(
    message: JsonObject,
    destination: ReplayOrigin,
    include: bool,
    output_path: HistoryPath,
    source_path: HistoryPath,
    paths: dict[HistoryPath, HistoryPath],
) -> None:
    content = message.get("content")
    if isinstance(content, str):
        if include:
            message["content"] = [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": content},
            ]
        return
    if not isinstance(content, list):
        return
    blocks: list[JsonValue] = []
    has_reasoning = False
    for index, block in enumerate(content):
        values: list[JsonValue] = []
        if not isinstance(block, dict) or block.get("type") not in {
            "thinking",
            "redacted_thinking",
        }:
            values.append(block)
        else:
            record = _record(block, "signature", "data")
            if protected_reasoning(block):
                has_reasoning = True
                if record is not None and destination.accepts(record.origin):
                    values.append(record.native)
                    if not readable_reasoning(record.native):
                        values.extend(
                            {"type": "text", "text": text}
                            for text, _ in readable_reasoning(block)
                        )
                elif record is None:
                    values.append(block)
                else:
                    values.extend(
                        {"type": "text", "text": text}
                        for text, _ in _replay_readable(block, record)
                    )
            elif include:
                if record is None and block.get("type") == "thinking":
                    values.append(block)
                else:
                    values.extend(
                        {"type": "thinking", "thinking": text}
                        for text, summary in _replay_readable(block, record)
                        if not summary
                    )
                has_reasoning |= bool(values)
        for value in values:
            paths[(*output_path, "content", len(blocks))] = (
                *source_path,
                "content",
                index,
            )
            blocks.append(value)
    if include and not has_reasoning:
        # Append so existing block paths are stable. No fabricated signature.
        blocks.append({"type": "thinking", "thinking": ""})
    message["content"] = blocks


def _append_context(message: JsonObject, text: str) -> None:
    """Legacy fallback only for protected foreign native state."""
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
    include: bool,
    reasoning_field: str,
    structured_details: bool,
    output_path: HistoryPath,
    source_path: HistoryPath,
    paths: dict[HistoryPath, HistoryPath],
) -> None:
    scalar = [
        text
        for key in ("reasoning_content", "reasoning")
        if isinstance(text := message.pop(key, None), str)
    ]
    details = message.pop("reasoning_details", None)
    restored: list[JsonValue] = []
    full: list[str] = []
    represented: set[str] = set()
    has_protected = False
    if isinstance(details, list):
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                continue
            record = _record(detail, "data", "signature")
            native = record.native if record else {"reasoning_details": [detail]}
            parts = readable_reasoning(native)
            protected = protected_reasoning(detail)
            has_protected |= protected
            if protected:
                represented.update(text for text, _ in parts)
                represented.update(
                    (
                        "".join(text for text, _ in parts),
                        "\n\n".join(text for text, _ in parts),
                    )
                )
            if record is not None and destination.accepts(record.origin):
                extra = record.native.get("extra_content")
                if isinstance(extra, dict):
                    message["extra_content"] = deepcopy(extra)
            values: list[JsonValue] = []
            if protected:
                if structured_details and (
                    record is None or destination.accepts(record.origin)
                ):
                    native_details = native.get("reasoning_details")
                    if isinstance(native_details, list):
                        values.extend(native_details)
                    native_parts = readable_reasoning(
                        {"reasoning_details": native_details}
                    )
                    for text, summary in parts:
                        if (text, summary) not in native_parts:
                            if summary:
                                _append_context(message, text)
                            else:
                                full.append(text)
                else:
                    # Native foreign reasoning retains its existing fallback.
                    for text, _ in parts:
                        _append_context(message, text)
            elif include:
                if structured_details:
                    native_details = native.get("reasoning_details")
                    if isinstance(native_details, list):
                        values.extend(native_details)
                    else:
                        values.extend(
                            {"type": "reasoning.summary", "summary": text}
                            for text, summary in parts
                            if summary
                        )
                    encoded = readable_reasoning({"reasoning_details": values})
                    full.extend(
                        text
                        for text, summary in parts
                        if not summary and (text, summary) not in encoded
                    )
                else:
                    full.extend(text for text, summary in parts if not summary)
            for value in values:
                paths[(*output_path, "reasoning_details", len(restored))] = (
                    *source_path,
                    "reasoning_details",
                    index,
                )
                restored.append(value)
    if include:
        full = [
            *dict.fromkeys(text for text in scalar if text and text not in represented),
            *full,
        ]
    if restored:
        message["reasoning_details"] = restored
    if full or (include and not has_protected and not restored):
        message[reasoning_field] = "\n".join(full)
