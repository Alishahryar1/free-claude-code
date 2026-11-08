"""Pure, request-local projection of reasoning retained by the native harness."""

import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal, cast

from .json_types import JsonObject, JsonValue
from .openai_chat import ChatToolResultImages
from .recovery_limits import RECOVERY_RETENTION_BYTES

type HistoryProtocol = Literal["responses", "messages", "chat", "gemini"]
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
class ContinuationReplay:
    context: tuple[JsonObject, ...]
    # None is an internal draft, sealed only after public overlap processing.
    echo: tuple[JsonObject, ...] | None = None


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    origin: ReplayOrigin
    native: JsonObject
    tool_defaults: JsonObject | None = field(default=None, kw_only=True)
    continuation: ContinuationReplay | None = field(default=None, kw_only=True)


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
    payload: JsonObject = {"origin": asdict(record.origin), "native": record.native}
    if record.tool_defaults is not None:
        payload["tool_defaults"] = record.tool_defaults
    if record.continuation is not None:
        payload["continuation"] = {
            "context": list(record.continuation.context),
            "echo": list(record.continuation.echo)
            if record.continuation.echo is not None
            else None,
        }
    associated = isinstance(record, AssociatedReplayRecord)
    if associated:
        payload.update(group_id=record.group_id, part=record.part)
    payload = _json_copy(payload)
    _validate_record(payload, associated=associated)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    prefix = _ASSOCIATED_PREFIX if associated else _PREFIX
    encoded = prefix + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    if record.continuation is not None:
        _check_continuation_size(encoded)
    return encoded


def _check_continuation_size(encoded: str) -> None:
    if len(encoded.encode("utf-8")) > RECOVERY_RETENTION_BYTES:
        raise HistoryReplayError("Continuation replay exceeded the retention limit.")


def encode_gemini_replay(record: ReplayRecord) -> str:
    """Encode the existing record as Gemini's base64 byte field."""
    encoded = base64.b64encode(encode_replay(record).encode("utf-8")).decode("ascii")
    _check_continuation_size(encoded)
    return encoded


def decode_gemini_replay(signature: str) -> ReplayRecord | None:
    """Recognize an FCC record while leaving ordinary native signatures alone."""
    try:
        raw = base64.b64decode(
            signature + "=" * (-len(signature) % 4), altchars=b"-_", validate=True
        )
    except ValueError:
        return None
    if not raw.startswith((b"fcc:history:", b"fcc:anthropic-reasoning:")):
        return None
    _check_continuation_size(signature)
    try:
        return decode_replay(raw.decode("utf-8"))
    except UnicodeError as error:
        raise HistoryReplayError("Malformed native Gemini replay.") from error


def _continuation_replay(value: JsonValue) -> ContinuationReplay:
    from .gemini.models import Content, Part

    if not isinstance(value, dict) or set(value) != {"context", "echo"}:
        raise HistoryReplayError("Invalid continuation replay fields.")
    context, echo = value["context"], value["echo"]
    if not isinstance(context, list) or len(context) not in {1, 2}:
        raise HistoryReplayError(
            "Continuation replay requires its instruction and optional model prefix."
        )
    try:
        contents = [Content.model_validate(item) for item in context]
        if [item.role for item in contents] not in (["user"], ["model", "user"]) or any(
            not item.parts
            or any(
                part.text is None or part.thought_signature or part.function_call
                for part in item.parts
            )
            for item in contents
        ):
            raise ValueError("Invalid readable context")
        if echo is not None:
            if not isinstance(echo, list) or not echo:
                raise ValueError("Invalid public echo")
            for item in echo:
                part = Part.model_validate(item)
                if part.thought_signature or (
                    part.text is None and part.function_call is None
                ):
                    raise ValueError("Invalid public echo part")
    except ValueError as error:
        raise HistoryReplayError("Invalid continuation replay content.") from error
    return ContinuationReplay(
        tuple(cast(JsonObject, item) for item in context),
        tuple(cast(JsonObject, item) for item in echo) if echo is not None else None,
    )


def _validate_record(payload: JsonObject, *, associated: bool = False) -> ReplayRecord:
    origin, native = payload.get("origin"), payload.get("native")
    fields = (
        {"origin", "native", "group_id", "part"} if associated else {"origin", "native"}
    )
    defaults = payload.get("tool_defaults")
    if "continuation" in payload:
        fields.add("continuation")
    if "tool_defaults" in payload:
        fields.add("tool_defaults")
        if not isinstance(defaults, dict) or any(
            not isinstance(value, dict) for value in defaults.values()
        ):
            raise HistoryReplayError("Invalid tool replay defaults.")
    if (
        set(payload) != fields
        or not isinstance(origin, dict)
        or set(origin) != {"provider", "protocol", "endpoint", "connection", "model"}
        or not all(isinstance(value, str) for value in origin.values())
        or not origin["provider"]
        or not origin["model"]
        or origin["protocol"] not in {"responses", "messages", "chat", "gemini"}
        or not isinstance(native, dict)
    ):
        raise HistoryReplayError(
            "History replay has invalid provenance or native data."
        )
    protocol = cast(HistoryProtocol, origin["protocol"])
    continuation = None
    if "continuation" in payload:
        if protocol != "gemini" or associated:
            raise HistoryReplayError(
                "Continuation replay requires native Gemini state."
            )
        continuation = _continuation_replay(payload["continuation"])
    if defaults is not None and (protocol != "gemini" or associated):
        raise HistoryReplayError("Tool replay defaults require native Gemini state.")
    if protocol == "gemini":
        from .gemini.models import Content

        try:
            content = Content.model_validate(native)
        except ValueError as error:
            raise HistoryReplayError(
                "Gemini replay requires native model content."
            ) from error
        if content.role != "model" or not any(
            part.thought_signature for part in content.parts
        ):
            raise HistoryReplayError("Gemini replay requires signed model content.")
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
        tool_defaults=cast(JsonObject | None, defaults),
        continuation=continuation,
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
    record = _validate_record(payload, associated=prefix == _ASSOCIATED_PREFIX)
    if record.continuation is not None:
        _check_continuation_size(value)
    return record


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
    if isinstance(native_parts := item.get("parts"), list):
        return [
            (str(part["text"]), True)
            for part in native_parts
            if isinstance(part, dict)
            and part.get("thought") is True
            and isinstance(part.get("text"), str)
        ]
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


def _record(
    item: Mapping[str, JsonValue], *keys: str, destination: ReplayOrigin | None = None
) -> ReplayRecord | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and is_replay(value):
            record = decode_replay(value)
            if (
                destination is not None
                and record.origin.protocol == "gemini"
                and destination.protocol != "gemini"
            ):
                raise HistoryReplayError(
                    "Protected native Gemini state cannot be replayed on another upstream protocol"
                )
            return record
    return None


def _context_item(text: str) -> JsonObject:
    return {"role": "assistant", "content": text}


def responses_reasoning_blocks(item: Mapping[str, JsonValue]) -> list[JsonValue]:
    """Carry Responses reasoning into Messages until destination preparation."""
    blocks: list[JsonValue] = []
    carrier = item.get("encrypted_content")
    if isinstance(carrier, str) and carrier:
        record = _record(item, "encrypted_content")
        if record is not None and not protected_reasoning(record.native):
            return [
                {"type": "thinking", "thinking": text}
                for text, _ in _replay_readable(item, record)
            ]
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
        {"type": "thinking", "thinking": text} for text, _ in readable_reasoning(item)
    )
    return blocks


type HistoryPath = tuple[str | int, ...]


def protected_reasoning(item: Mapping[str, JsonValue]) -> bool:
    """Local carriers do not turn unsigned readable data into protected state."""
    record = _record(item, "encrypted_content", "signature", "data")
    if record is not None:
        return protected_reasoning(record.native)
    if (
        item.get("encrypted_content")
        or item.get("signature")
        or item.get("thoughtSignature")
    ):
        return True
    if isinstance(parts := item.get("parts"), list) and any(
        isinstance(part, dict) and protected_reasoning(part) for part in parts
    ):
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
    if protocol == "messages" and isinstance(messages := result.get("messages"), list):
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and isinstance(content := message.get("content"), list)
            ):
                blocks: list[JsonValue] = []
                for block in resolve_messages_replay(content):
                    record = (
                        _record(block, "signature", "data")
                        if isinstance(block, dict)
                        else None
                    )
                    if (
                        isinstance(block, dict)
                        and record is not None
                        and not protected_reasoning(record.native)
                    ):
                        blocks.extend(
                            {"type": "thinking", "thinking": text}
                            for text, _ in _replay_readable(block, record)
                        )
                    else:
                        blocks.append(block)
                message["content"] = blocks
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
    native_chat: bool = False,
) -> JsonObject:
    """Restore native records and project readable history on a fresh wire copy.

    Source paths identify original records for precise corrections after filtering.
    They are request-local metadata, never fields in the outgoing JSON.
    """
    result = normalize_history_source(body, destination.protocol)
    paths = source_paths if source_paths is not None else {}
    paths.clear()
    if destination.protocol == "gemini":
        _prepare_gemini_history(
            result, destination, scope, paths, continuation=continuation
        )
        return result
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
        elif native_chat and not (
            isinstance(details := message.get("reasoning_details"), list)
            and any(
                isinstance(detail, dict)
                and _record(detail, "data", "signature") is not None
                for detail in details
            )
        ):
            if not include:
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
                if isinstance(details := message.get("reasoning_details"), list):
                    retained = [
                        detail
                        for detail in details
                        if isinstance(detail, dict) and protected_reasoning(detail)
                    ]
                    if retained:
                        message["reasoning_details"] = retained
                    else:
                        message.pop("reasoning_details", None)
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
        "content": [{"type": "reasoning_text", "text": text} for text, _ in parts],
        "summary": [],
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
                record = _record(item, "encrypted_content", destination=destination)
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
            record = _record(block, "signature", "data", destination=destination)
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
                        for text, _ in _replay_readable(block, record)
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
    native_full: list[str] = []
    if isinstance(details, list):
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                continue
            record = _record(detail, "data", "signature", destination=destination)
            native = record.native if record else {"reasoning_details": [detail]}
            parts = readable_reasoning(native)
            if record is None:
                native_full.extend(text for text, _ in parts)
            protected = protected_reasoning(detail)
            if protected and record is None:
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
                full.extend(text for text, _ in parts)
            for value in values:
                paths[(*output_path, "reasoning_details", len(restored))] = (
                    *source_path,
                    "reasoning_details",
                    index,
                )
                restored.append(value)
    if include:
        # Raw Chat details and scalar fields are alternate forms of this message.
        # A replay carrier may instead represent an independent source record.
        represented.update(native_full)
        represented.update(
            separator.join(native_full) for separator in ("", "\n", "\n\n")
        )
        full = [
            *dict.fromkeys(text for text in scalar if text and text not in represented),
            *full,
        ]
    if restored:
        message["reasoning_details"] = restored
    if full or (
        include and ("" in scalar or not (restored or message.get("extra_content")))
    ):
        message[reasoning_field] = "\n".join(full)


def _visible_gemini_parts(
    parts: list[dict[str, Any]], *, thoughts: bool
) -> list[dict[str, Any]]:
    from .gemini.models import Part

    result: list[dict[str, Any]] = []
    # Chat groups thinking, answer text, and calls into separate fields. Compare
    # each channel in order, then restore the original native ordering verbatim.
    ordered = sorted(
        parts, key=lambda part: 0 if part.get("thought") else 1 if "text" in part else 2
    )
    for part in ordered:
        if not any(
            key in part
            for key in ("text", "functionCall", "inlineData", "functionResponse")
        ):
            continue
        # ProtoJSON nulls and omitted optional fields have the same meaning.
        # Normalize only this comparison copy, never the signed native record.
        value = Part.model_validate(part).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        value.pop("thoughtSignature", None)
        if not value.get("thought"):
            value.pop("thought", None)
        if not any(
            key in value
            for key in ("text", "functionCall", "inlineData", "functionResponse")
        ) or (value.get("thought") and not thoughts):
            continue
        if value.get("text") == "":
            continue
        if (
            result
            and isinstance(value.get("text"), str)
            and isinstance(result[-1].get("text"), str)
            and bool(result[-1].get("thought")) == bool(value.get("thought"))
        ):
            result[-1]["text"] += value["text"]
        else:
            result.append(value)
    return result


def _gemini_replay_record(signature: object) -> ReplayRecord | None:
    if not isinstance(signature, str):
        return None
    return (
        decode_replay(signature)
        if is_replay(signature)
        else decode_gemini_replay(signature)
    )


def _gemini_replay_turns(
    contents: list[JsonValue], *, continuation: bool
) -> Iterator[tuple[int, JsonObject, list[HistoryPath]]]:
    """Join SDK chunks of a recovered turn, retaining caller row coordinates."""
    index = 0
    while index < len(contents):
        content = contents[index]
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            raise HistoryReplayError("Invalid native Gemini content")
        end = index + 1
        if content.get("role") == "model":
            while end < len(contents):
                following = contents[end]
                if (
                    not isinstance(following, dict)
                    or following.get("role") != "model"
                    or not isinstance(following.get("parts"), list)
                    or (continuation and end == len(contents) - 2)
                ):
                    break
                end += 1
        rows = cast(list[JsonObject], contents[index:end])
        if end > index + 1 and any(
            record is not None and record.continuation is not None
            for row in rows
            for part in cast(list[JsonObject], row["parts"])
            for record in [_gemini_replay_record(part.get("thoughtSignature"))]
        ):
            parts = [
                part for row in rows for part in cast(list[JsonObject], row["parts"])
            ]
            paths: list[HistoryPath] = [
                ("contents", row_index, "parts", part_index)
                for row_index, row in enumerate(rows, index)
                for part_index, _ in enumerate(cast(list[JsonObject], row["parts"]))
            ]
            yield index, {"role": "model", "parts": parts}, paths
        else:
            for row_index, row in enumerate(rows, index):
                yield (
                    row_index,
                    row,
                    [
                        ("contents", row_index, "parts", part_index)
                        for part_index, _ in enumerate(
                            cast(list[JsonObject], row["parts"])
                        )
                    ],
                )
        index = end


def _restore_gemini_parts(
    parts: list[dict[str, Any]],
    origin: ReplayOrigin,
    call_aliases: set[str],
) -> tuple[list[dict[str, Any]], int | None, tuple[JsonObject, ...]]:
    records = []
    echo = []
    carrier_index = None
    for index, part in enumerate(parts):
        signature = part.get("thoughtSignature")
        record = _gemini_replay_record(signature)
        if record is not None:
            records.append(record)
            carrier_index = index
            stripped = {
                key: value for key, value in part.items() if key != "thoughtSignature"
            }
            if stripped:
                echo.append(stripped)
        else:
            if isinstance(signature, str):
                try:
                    decoded = base64.b64decode(signature, validate=True)
                except ValueError:
                    decoded = b""
                if decoded.startswith(b"fcc:gemini-replay:"):
                    raise HistoryReplayError(
                        "Saved compatibility signatures require a fresh Gemini turn"
                    )
            echo.append(part)
    if not records:
        return parts, None, ()
    native = [record for record in records if record.origin.protocol == "gemini"]
    if not native:
        # A compatibility signature is bound to the old Chat representation.
        # Google does not document a lossless cross-API signature migration.
        raise HistoryReplayError(
            "Saved non-native reasoning cannot be replayed on the native Gemini connection; start a fresh turn"
        )
    if len(native) != 1 or len(records) != 1:
        raise HistoryReplayError("Conflicting native Gemini replay records")
    record = native[0]
    if not origin.accepts(record.origin):
        raise HistoryReplayError(
            "Native Gemini state belongs to a different connection"
        )
    tool_defaults = cast(dict[str, dict[str, Any]], record.tool_defaults or {})
    restored = cast(list[dict[str, Any]], deepcopy(record.native["parts"]))
    continuation = record.continuation
    if continuation is not None and continuation.echo is None:
        raise HistoryReplayError(
            "Unfinished continuation replay cannot be used as history."
        )
    expected = (
        cast(list[dict[str, Any]], list(continuation.echo))
        if continuation is not None and continuation.echo is not None
        else restored
    )
    thoughts = any(part.get("thought") and part.get("text") for part in echo)
    visible_echo = _visible_gemini_parts(echo, thoughts=thoughts)
    visible_native = _visible_gemini_parts(expected, thoughts=thoughts)
    native_calls = iter(
        part["functionCall"] for part in restored if "functionCall" in part
    )
    for echoed, native_part in zip(visible_echo, visible_native, strict=False):
        expected_call = native_part.get("functionCall")
        call = next(native_calls, {}) if isinstance(expected_call, dict) else None
        echo_call = echoed.get("functionCall")
        if isinstance(call, dict) and isinstance(echo_call, dict):
            original_args = call.get("args") or {}
            echo_args = echo_call.get("args", {})
            # Harnesses can materialize declared defaults before echoing a call.
            # Ignore only added default-valued fields during comparison. The
            # signed native arguments are restored exactly, never rewritten.
            for key, value in tool_defaults.get(call.get("name"), {}).items():
                if (
                    key not in original_args
                    and key in echo_args
                    and echo_args[key] == value
                ):
                    echo_args.pop(key)
                    if isinstance(expected_call, dict):
                        expected_call.get("args", {}).pop(key, None)

        # Other harnesses require an ID. Google's signed Part may omit it.
        if (
            isinstance(call, dict)
            and not call.get("id")
            and isinstance(echo_call, dict)
            and isinstance(alias := echo_call.pop("id", None), str)
        ):
            call_aliases.add(alias)
            if isinstance(expected_call, dict):
                expected_call.pop("id", None)
    if visible_echo != visible_native:
        raise HistoryReplayError("Signed Gemini content was edited after generation")
    return (
        restored,
        carrier_index,
        deepcopy(continuation.context) if continuation else (),
    )


def _prepare_gemini_history(
    body: JsonObject,
    origin: ReplayOrigin,
    scope: HistoryMode,
    paths: dict[HistoryPath, HistoryPath],
    *,
    continuation: bool,
) -> None:
    contents = body.get("contents")
    if not isinstance(contents, list):
        return
    last_user = max(
        (
            index
            for index, content in enumerate(contents[:-1] if continuation else contents)
            if isinstance(content, dict)
            and content.get("role", "user") == "user"
            and any(
                isinstance(part, dict) and "functionResponse" not in part
                for part in content.get("parts", [])
            )
        ),
        default=-1,
    )
    projected = []
    call_aliases: set[str] = set()
    for index, content, part_sources in _gemini_replay_turns(
        contents, continuation=continuation
    ):
        source = ("contents", index)
        target = ("contents", len(projected))
        paths[target] = source
        if content.get("role") != "model":
            for part in cast(list[dict[str, Any]], content["parts"]):
                if (
                    isinstance(part, dict)
                    and isinstance(result := part.get("functionResponse"), dict)
                    and isinstance(alias := result.get("id"), str)
                    and alias in call_aliases
                ):
                    call_aliases.remove(alias)
                    result.pop("id")
            projected.append(content)
            continue
        parts, carrier_index, continuation_context = _restore_gemini_parts(
            cast(list[dict[str, Any]], content["parts"]),
            origin,
            call_aliases,
        )
        if carrier_index is not None:
            source = part_sources[carrier_index][:-2]
            paths[target] = source
        if continuation_context:
            for context in continuation_context:
                paths[("contents", len(projected))] = source
                projected.append(context)
            target = ("contents", len(projected))
            paths[target] = source
        include = scope.includes(index, last_user)
        kept = []
        for part_index, part in enumerate(parts):
            if (
                not continuation_context
                and part.get("thought")
                and not part.get("thoughtSignature")
                and not include
            ):
                continue
            paths[(*target, "parts", len(kept))] = part_sources[
                carrier_index if carrier_index is not None else part_index
            ]
            kept.append(part)
        if (
            not continuation_context
            and include
            and not any(
                part.get("thought") or part.get("thoughtSignature") for part in kept
            )
        ):
            kept.append({"text": "", "thought": True})
        if kept:
            content["parts"] = kept
            projected.append(content)
    body["contents"] = projected
