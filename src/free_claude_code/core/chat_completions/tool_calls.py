"""Choice-scoped Chat call accumulation; only source termination seals calls."""

import json
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..failures import ExecutionFailure, FailureKind
from ..openai_tool_names import OpenAIToolNameCodec
from ..stream_recovery import SourceRecoverySnapshot, StreamFailureContext
from ..tool_call_buffer import OrderedToolCallBuffer, RetentionBudget
from .response import chat_failure


class AmbiguousToolCallIdentityError(ExecutionFailure):
    """No unique upstream call owns a fragment."""

    def __init__(
        self,
        message: str,
        *,
        kind: FailureKind = FailureKind.UPSTREAM,
        status_code: int = 502,
        retryable: bool = False,
        stream_context: StreamFailureContext | None = None,
    ) -> None:
        super().__init__(
            kind,
            status_code,
            message,
            retryable,
            stream_context
            or StreamFailureContext(
                SourceRecoverySnapshot(unsafe_reason="ambiguous_tool_identity")
            ),
        )


@dataclass(slots=True)
class _Identity:
    index: int | None = None
    tool_id: str | None = None


class ToolCallIdentityResolver:
    def __init__(self, budget: RetentionBudget | None = None) -> None:
        self._calls: list[_Identity] = []
        self._budget = budget

    def resolve(self, raw_index: object, raw_id: object) -> tuple[int, str | None]:
        index = (
            raw_index
            if isinstance(raw_index, int)
            and not isinstance(raw_index, bool)
            and raw_index >= 0
            else None
        )
        tool_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else None
        if index is None and tool_id is None:
            raise AmbiguousToolCallIdentityError(
                "Tool fragment has neither a usable index nor an ID."
            )
        matches = [
            slot
            for slot, call in enumerate(self._calls)
            if (index is None or call.index in (None, index))
            and (tool_id is None or call.tool_id in (None, tool_id))
        ]
        if not any(
            (index is not None and self._calls[slot].index == index)
            or (tool_id is not None and self._calls[slot].tool_id == tool_id)
            for slot in matches
        ):
            matches = []
        if len(matches) > 1:
            raise AmbiguousToolCallIdentityError(
                "Tool fragment has ambiguous identity."
            )
        if matches:
            slot = matches[0]
        else:
            slot = len(self._calls)
            if self._budget is not None:
                self._budget.add(32)
            self._calls.append(_Identity())
        call = self._calls[slot]
        if index is not None:
            call.index = index
        if tool_id is not None:
            if self._budget is not None and call.tool_id is None:
                self._budget.add(len(tool_id.encode("utf-8")), 0)
            call.tool_id = tool_id
        return slot, call.tool_id


@dataclass(frozen=True, slots=True)
class CompletedChatCall:
    choice: int
    index: int
    body: dict[str, Any]
    envelope: dict[str, Any]
    extensions: tuple[dict[str, Any], ...] = ()


@dataclass(slots=True)
class _PendingCall:
    choice: int
    index: int
    envelope: dict[str, Any]
    body: dict[str, Any] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)
    extensions: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "function"
    completed: CompletedChatCall | None = None

    def add(self, fragment: Mapping[str, Any], *, tool_id: str | None) -> None:
        if isinstance(fragment.get("custom"), Mapping):
            self.kind = "custom"
        function = fragment.get(self.kind) or {}
        if name := function.get("name"):
            if not isinstance(name, str):
                raise chat_failure("Tool name fragment must be a string")
            self.names.append(name)
        arguments = function.get("input" if self.kind == "custom" else "arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise chat_failure("Tool argument fragment must be a string")
            self.arguments.append(arguments)
        if tool_id is not None:
            self.body["id"] = tool_id
        if fragment.get("type") is not None:
            self.body["type"] = fragment["type"]
        extra = {
            key: deepcopy(value)
            for key, value in fragment.items()
            if key not in {"index", "id", "type", "function", "custom"}
            and value is not None
        }
        function_extra = {
            key: deepcopy(value)
            for key, value in function.items()
            if key not in {"name", "arguments", "input"} and value is not None
        }
        if function_extra:
            extra[self.kind] = function_extra
        if extra:
            self.extensions.append(extra)

    def seal(self) -> None:
        name = "".join(self.names)
        if not name.strip():
            raise chat_failure("Completed tool call has no name")
        body = {
            **self.body,
            "type": self.body.get("type", self.kind),
            self.kind: {
                "name": name,
                "input" if self.kind == "custom" else "arguments": "".join(
                    self.arguments
                ),
            },
        }
        # Keep the ordinary single metadata snapshot readily available to presenters.
        # Distinct fragment-associated snapshots also remain available to native output.
        for extra in self.extensions:
            for key, value in extra.items():
                if key == self.kind:
                    body[self.kind].update(value)
                else:
                    body[key] = value
        self.completed = CompletedChatCall(
            self.choice, self.index, body, self.envelope, tuple(self.extensions)
        )


type ChatSourceOutput = dict[str, Any] | CompletedChatCall


@dataclass(slots=True)
class _NativeGroup:
    frames: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompletedChatBatch:
    frames: tuple[dict[str, Any], ...]


class ChatToolCallBuffer:
    def __init__(
        self, *, budget: RetentionBudget | None = None, native: bool = False
    ) -> None:
        self._buffer = OrderedToolCallBuffer[
            dict[str, Any] | _PendingCall | _NativeGroup
        ](budget)
        self._identities: dict[int, ToolCallIdentityResolver] = {}
        self._calls: dict[tuple[int, int], _PendingCall] = {}
        self._finished: set[int] = set()
        self._native = native
        self._native_group: _NativeGroup | None = None

    def feed(
        self, payload: Mapping[str, Any]
    ) -> list[ChatSourceOutput | CompletedChatBatch]:
        for choice in payload.get("choices") or []:
            index = choice.get("index", 0)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise chat_failure("Invalid Chat choice index")
            if index in self._finished:
                raise chat_failure("Chat choice changed after completion")
        if self._native:
            return [
                output
                for part in _native_parts(payload)
                for output in self._feed_native(part)
            ]
        envelope = {
            key: deepcopy(value) for key, value in payload.items() if key != "choices"
        }
        choices = payload.get("choices") or []
        if not choices:
            self._append(dict(payload))
        for position, choice in enumerate(choices):
            choice_index = choice.get("index", 0)
            if choice_index in self._finished:
                raise chat_failure("Chat choice changed after completion")
            delta = dict(choice.get("delta") or {})
            calls = delta.pop("tool_calls", None) or []
            finish = choice.get("finish_reason")
            ordinary = {**choice, "delta": delta, "finish_reason": None}
            if delta or any(
                key not in {"index", "delta", "finish_reason"} and value is not None
                for key, value in choice.items()
            ):
                self._append({**envelope, "choices": [ordinary]})
            elif position == 0 and any(
                key
                not in {
                    "id",
                    "object",
                    "created",
                    "model",
                    "system_fingerprint",
                    "service_tier",
                    "usage",
                }
                for key in envelope
            ):
                self._append({**envelope, "choices": []})
            for fragment in calls:
                resolver = self._identities.setdefault(
                    choice_index, ToolCallIdentityResolver(self._buffer.budget)
                )
                slot, tool_id = resolver.resolve(
                    fragment.get("index"), fragment.get("id")
                )
                key = (choice_index, slot)
                if key not in self._calls:
                    base = {
                        key: value
                        for key, value in envelope.items()
                        if key
                        in {
                            "id",
                            "object",
                            "created",
                            "model",
                            "system_fingerprint",
                            "service_tier",
                        }
                    }
                    call = _PendingCall(choice_index, slot, base)
                    self._buffer.reserve(key, call)
                    self._calls[key] = call
                self._buffer.grow(key, _size(fragment))
                self._calls[key].add(fragment, tool_id=tool_id)
            if finish is not None:
                self._finish_choice(choice_index, finish)
                self._append(
                    {
                        **envelope,
                        "choices": [{**choice, "delta": {}, "finish_reason": finish}],
                    }
                )
        return self._drain()

    def _feed_native(
        self, payload: Mapping[str, Any]
    ) -> list[ChatSourceOutput | CompletedChatBatch]:
        data = deepcopy(dict(payload))
        has_calls = any(
            (choice.get("delta") or {}).get("tool_calls")
            for choice in data.get("choices", [])
        )
        if self._native_group is None and not has_calls:
            for choice in data.get("choices", []):
                if choice.get("finish_reason") is not None:
                    self._remember_finish(choice.get("index", 0))
            return [data]
        if self._native_group is None:
            self._native_group = _NativeGroup()
            self._buffer.reserve("native", self._native_group)
        self._buffer.grow("native", _size(data))
        self._native_group.frames.append(data)
        for choice in data.get("choices", []):
            choice_index = choice.get("index", 0)
            if choice_index in self._finished:
                raise chat_failure("Chat choice changed after completion")
            for fragment in (choice.get("delta") or {}).get("tool_calls") or []:
                resolver = self._identities.setdefault(
                    choice_index, ToolCallIdentityResolver(self._buffer.budget)
                )
                slot, tool_id = resolver.resolve(
                    fragment.get("index"), fragment.get("id")
                )
                key = (choice_index, slot)
                if key not in self._calls:
                    self._calls[key] = _PendingCall(choice_index, slot, {})
                self._buffer.grow("native", 0)
                self._calls[key].add(fragment, tool_id=tool_id)
                fragment["index"] = slot
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason in {"stop", "tool_calls"}:
                    for (owner, _), call in self._calls.items():
                        if owner == choice_index:
                            call.seal()
                self._remember_finish(choice_index)
        if all(choice in self._finished for choice, _ in self._calls):
            self._complete_native_group()
        return self._drain()

    def _complete_native_group(self) -> None:
        group = self._native_group
        if group is None:
            return
        identities: set[tuple[int, str]] = set()
        for call in self._calls.values():
            if call.completed is None:
                continue
            identity = call.completed.body.get("id")
            if not isinstance(identity, str) or not identity.strip():
                raise AmbiguousToolCallIdentityError(
                    "Completed native Chat call requires an ID"
                )
            key = (call.choice, identity)
            if key in identities:
                raise AmbiguousToolCallIdentityError(
                    "Native Chat calls share an execution ID"
                )
            identities.add(key)
        seen: set[tuple[int, int]] = set()
        for data in group.frames:
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                fragments = delta.get("tool_calls")
                if fragments is None:
                    continue
                retained = []
                dropped = any(
                    self._calls[(choice.get("index", 0), fragment["index"])].completed
                    is None
                    for fragment in fragments
                )
                opaque_envelope = any(
                    key
                    not in {
                        "id",
                        "object",
                        "created",
                        "model",
                        "system_fingerprint",
                        "service_tier",
                        "usage",
                        "choices",
                    }
                    for key in data
                )
                opaque_envelope |= any(
                    key not in {"index", "delta", "finish_reason", "logprobs"}
                    for key in choice
                )
                opaque_envelope |= any(
                    key
                    not in {
                        "role",
                        "content",
                        "reasoning",
                        "reasoning_content",
                        "reasoning_details",
                        "refusal",
                        "tool_calls",
                    }
                    for key in delta
                )
                if dropped and opaque_envelope:
                    raise AmbiguousToolCallIdentityError(
                        "Cannot separate unfinished calls from an opaque native Chat envelope"
                    )
                for fragment in fragments:
                    key = (choice.get("index", 0), fragment["index"])
                    call = self._calls[key]
                    if call.completed is None:
                        continue
                    function = fragment.get(call.kind)
                    if key not in seen:
                        if not isinstance(function, dict):
                            function = fragment[call.kind] = {}
                        function["name"] = call.completed.body[call.kind]["name"]
                        if call.completed.body.get("id"):
                            fragment["id"] = call.completed.body["id"]
                        seen.add(key)
                    elif isinstance(function, dict):
                        function.pop("name", None)
                    if fragment.get("id") != call.completed.body["id"]:
                        fragment.pop("id", None)
                    retained.append(fragment)
                if retained:
                    delta["tool_calls"] = retained
                else:
                    delta.pop("tool_calls", None)
        self._buffer.complete("native")
        self._native_group = None
        self._calls.clear()

    def _append(self, value: dict[str, Any]) -> None:
        self._buffer.append(deepcopy(value), size=_size(value))

    def _finish_choice(self, choice: int, reason: object) -> None:
        keys = [key for key in self._calls if key[0] == choice]
        if reason in {"stop", "tool_calls"}:
            for key in keys:
                self._calls[key].seal()
            for key in keys:
                self._buffer.complete(key)
        else:
            for key in keys:
                self._buffer.discard(key)
                del self._calls[key]
        self._remember_finish(choice)

    def _remember_finish(self, choice: int) -> None:
        self._buffer.budget.add(32)
        self._finished.add(choice)

    def _drain(self) -> list[ChatSourceOutput | CompletedChatBatch]:
        result: list[ChatSourceOutput | CompletedChatBatch] = []
        for item in self._buffer.drain():
            if isinstance(item, _PendingCall):
                assert item.completed is not None
                result.append(item.completed)
                del self._calls[(item.choice, item.index)]
            elif isinstance(item, _NativeGroup):
                result.append(CompletedChatBatch(tuple(item.frames)))
            else:
                result.append(item)
        return result

    def abandon(self) -> list[ChatSourceOutput | CompletedChatBatch]:
        if self._native:
            self._complete_native_group()
            return self._drain()
        self._buffer.discard_pending()
        result = self._drain()
        self._calls.clear()
        return result


def _native_parts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Separate standard content from calls without guessing opaque associations."""
    data = dict(payload)
    choices = data.get("choices") or []
    if not any((choice.get("delta") or {}).get("tool_calls") for choice in choices):
        return [data]
    if any(
        key
        not in {
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "service_tier",
            "usage",
            "choices",
        }
        for key in data
    ) or any(
        any(
            key not in {"index", "delta", "finish_reason", "logprobs"} for key in choice
        )
        or any(
            key
            not in {
                "role",
                "content",
                "reasoning",
                "reasoning_content",
                "reasoning_details",
                "refusal",
                "tool_calls",
            }
            for key in (choice.get("delta") or {})
        )
        for choice in choices
    ):
        return [data]
    envelope = {key: value for key, value in data.items() if key != "choices"}
    base = {
        key: value
        for key, value in envelope.items()
        if key in {"id", "object", "created", "model"}
    }
    result: list[dict[str, Any]] = []
    for choice in choices:
        delta = dict(choice.get("delta") or {})
        calls = delta.pop("tool_calls", None)
        if not calls:
            result.append({**(base if result else envelope), "choices": [choice]})
            continue
        if delta or choice.get("logprobs") is not None:
            ordinary = {**choice, "delta": delta, "finish_reason": None}
            result.append({**(base if result else envelope), "choices": [ordinary]})
        tool_choice = {key: value for key, value in choice.items() if key != "logprobs"}
        tool_choice["delta"] = {"tool_calls": calls}
        result.append({**(base if result else envelope), "choices": [tool_choice]})
    return result


def finalize_chat_call(
    call: CompletedChatCall,
    *,
    codec: OpenAIToolNameCodec,
    aliases: Mapping[str, dict[str, str]],
    reserved_ids: set[str],
) -> CompletedChatCall:
    metadata = [
        extra["extra_content"] for extra in call.extensions if "extra_content" in extra
    ]
    if metadata and any(value != metadata[0] for value in metadata[1:]):
        raise AmbiguousToolCallIdentityError(
            "This destination cannot preserve multiple distinct call metadata fragments"
        )
    body = deepcopy(call.body)
    kind = "custom" if "custom" in body else "function"
    function = body[kind]
    name = codec.decode(function["name"])
    function["name"] = name
    if kind == "function" and (mapping := aliases.get(name)):
        restored = restore_tool_argument_aliases(function["arguments"], mapping)
        if restored is not None:
            function["arguments"] = restored
    tool_id = body.get("id")
    if not isinstance(tool_id, str) or not tool_id.strip() or tool_id in reserved_ids:
        tool_id = f"tool_{uuid.uuid4()}"
        while tool_id in reserved_ids:
            tool_id = f"tool_{uuid.uuid4()}"
    body["id"] = tool_id
    reserved_ids.add(tool_id)
    return CompletedChatCall(
        call.choice, call.index, body, call.envelope, call.extensions
    )


def restore_tool_argument_aliases(
    argument_json: str, aliases: Mapping[str, str]
) -> str | None:
    try:
        parsed = json.loads(argument_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return argument_json

    def restore(value: Any) -> Any:
        if isinstance(value, dict):
            return {aliases.get(key, key): restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return json.dumps(restore(parsed))


def _size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
