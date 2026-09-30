"""OpenRouter-format structured reasoning replay and stream conversion."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from uuid import uuid4

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    readable_reasoning,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .stream_output import ChatStreamOutput


@dataclass(slots=True)
class _ReasoningGroup:
    origin: ReplayOrigin
    id: str = field(default_factory=lambda: uuid4().hex)
    details: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[tuple[object, object], int] = field(default_factory=dict)
    native_parts: list[str] = field(default_factory=list)
    after_content: bool = False

    def snapshot(self) -> ReplayRecord:
        native: JsonObject = {
            "reasoning_details": cast(list[JsonValue], deepcopy(self.details))
        }
        if self.native_parts:
            native["reasoning_content"] = "".join(self.native_parts)
        return ReplayRecord(self.origin, native)


class StructuredReasoningStream:
    """Collect replay groups independently of their visible wire blocks."""

    def __init__(self) -> None:
        self._text_source: Literal["native", "details"] | None = None
        self._group: _ReasoningGroup | None = None

    @property
    def active(self) -> bool:
        return self._group is not None

    def before_reasoning(self, output: ChatStreamOutput) -> Iterator[str]:
        if self._group is not None and self._group.after_content:
            yield from self.finish(output)

    def before_content(self, output: ChatStreamOutput) -> Iterator[str]:
        if self._group is not None and not self._group.after_content:
            self._group.after_content = True
            yield from output.pause_reasoning_record(
                self._group.id, self._group.snapshot()
            )

    def events(
        self,
        delta: Any,
        output: ChatStreamOutput,
        *,
        native_reasoning: str | None,
    ) -> Iterator[str]:
        details = _reasoning_details(delta)
        if self._text_source is None:
            if native_reasoning:
                self._text_source = "native"
            elif any(_reasoning_detail_text(detail) for detail in details):
                self._text_source = "details"
        visible = (
            [native_reasoning]
            if self._text_source == "native" and native_reasoning
            else [
                text for detail in details if (text := _reasoning_detail_text(detail))
            ]
            if self._text_source == "details"
            else []
        )
        if visible:
            yield from self.before_reasoning(output)
        if self._group is None and (native_reasoning is not None or details):
            if output.replay_origin is None:
                raise AssertionError(
                    "Structured reasoning requires attempt provenance."
                )
            self._group = _ReasoningGroup(output.replay_origin)
            yield from output.begin_reasoning_record()
        group = self._group
        if group is None:
            return
        if native_reasoning is not None:
            group.native_parts.append(native_reasoning)
        for detail in details:
            if isinstance(detail, Mapping):
                value = deepcopy(dict(detail))
                identity = value.get("index", value.get("id"))
                key = (identity, value.get("type"))
                if isinstance(identity, str | int) and key in group.slots:
                    current = group.details[group.slots[key]]
                    for name, part in value.items():
                        if (
                            name
                            in {
                                "text",
                                "summary",
                                "content",
                                "reasoning",
                                "data",
                                "signature",
                            }
                            and isinstance(part, str)
                            and isinstance(current.get(name), str)
                        ):
                            current[name] += part
                        else:
                            current[name] = part
                else:
                    if isinstance(identity, str | int):
                        group.slots[key] = len(group.details)
                    group.details.append(value)
        for text in visible:
            yield from output.ensure_reasoning_block()
            yield output.emit_reasoning_delta(text)

    def finish(self, output: ChatStreamOutput) -> Iterator[str]:
        group, self._group = self._group, None
        if group is not None:
            yield from output.complete_reasoning_record(group.id, group.snapshot())


def _reasoning_details(delta: Any) -> Sequence[Any]:
    details = _field(delta, "reasoning_details")
    if details is None:
        extra = _field(delta, "model_extra")
        if isinstance(extra, Mapping):
            details = extra.get("reasoning_details")
    return details if _is_sequence(details) else ()


def _reasoning_detail_text(detail: Any) -> str | None:
    if isinstance(detail, Mapping) and detail.get("type") == "reasoning.summary":
        return (
            "".join(
                text
                for text, _ in readable_reasoning({"reasoning_details": [dict(detail)]})
            )
            or None
        )
    kind = str(_field(detail, "type") or "").lower()
    if "encrypted" in kind or "redacted" in kind:
        return None
    for key in ("text", "content", "reasoning"):
        value = _field(detail, key)
        if isinstance(value, str) and value:
            return value
    return None


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
