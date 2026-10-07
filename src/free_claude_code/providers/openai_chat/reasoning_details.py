"""OpenRouter-format structured reasoning replay and stream conversion."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from free_claude_code.core.history_replay import readable_reasoning

from .stream_output import ChatStreamOutput


@dataclass(slots=True)
class _ReasoningGroup:
    details: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[tuple[object, object], int] = field(default_factory=dict)
    native_parts: list[str] = field(default_factory=list)
    visible_parts: dict[tuple[str, int], str] = field(default_factory=dict)
    output_reasoning: bool = True
    after_content: bool = False


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
            yield from self._remaining_readable(output, self._group)
            self._group.after_content = True

    def events(
        self,
        delta: Any,
        output: ChatStreamOutput,
        *,
        native_reasoning: str | None,
        output_reasoning: bool = True,
    ) -> Iterator[str]:
        details = _reasoning_details(delta)
        if self._text_source is None:
            if native_reasoning:
                self._text_source = "native"
            elif any(_reasoning_detail_text(detail) for detail in details):
                self._text_source = "details"
        visible = (
            [(native_reasoning, False)]
            if self._text_source == "native" and native_reasoning
            else [
                (text, _field(detail, "type") == "reasoning.summary")
                for detail in details
                if (text := _reasoning_detail_text(detail))
            ]
            if self._text_source == "details"
            else []
        )
        if not output_reasoning:
            visible = []
        if visible:
            yield from self.before_reasoning(output)
            if self._text_source is None:
                self._text_source = "native" if native_reasoning else "details"
        if self._group is None and (
            (output_reasoning and native_reasoning)
            or any(
                (output_reasoning and _reasoning_detail_text(detail))
                or _reasoning_detail_opaque(detail)
                for detail in details
            )
        ):
            self._group = _ReasoningGroup(output_reasoning=output_reasoning)
            yield from output.begin_reasoning_record()
        group = self._group
        if group is None:
            return
        if native_reasoning is not None:
            group.native_parts.append(native_reasoning)
        visible_parts: list[tuple[tuple[str, int], str, bool]] = []
        if self._text_source == "native" and native_reasoning and output_reasoning:
            visible_parts.append((("native", 0), native_reasoning, False))
        for detail in details:
            if isinstance(detail, Mapping):
                value = deepcopy(dict(detail))
                identity = value.get("index", value.get("id"))
                key = (identity, value.get("type"))
                if isinstance(identity, str | int) and key in group.slots:
                    slot = group.slots[key]
                    current = group.details[slot]
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
                    slot = len(group.details)
                    if isinstance(identity, str | int):
                        group.slots[key] = slot
                    group.details.append(value)
                if (
                    self._text_source == "details"
                    and output_reasoning
                    and (text := _reasoning_detail_text(detail))
                ):
                    visible_parts.append(
                        (
                            ("detail", slot),
                            text,
                            _field(detail, "type") == "reasoning.summary",
                        )
                    )
        for key, text, summary in visible_parts:
            yield from output.ensure_reasoning_block()
            yield output.emit_reasoning_delta(text, summary=summary)
            group.visible_parts[key] = group.visible_parts.get(key, "") + text

    def _remaining_readable(
        self, output: ChatStreamOutput, group: _ReasoningGroup
    ) -> Iterator[str]:
        if not group.output_reasoning:
            return
        typed = [
            (("detail", slot), text, detail.get("type") == "reasoning.summary")
            for slot, detail in enumerate(group.details)
            if (text := _reasoning_detail_text(detail))
        ]
        represented = set(group.visible_parts.values())
        for summary in (None, False, True):
            emitted = [
                group.visible_parts.get(key, "")
                for key, _, is_summary in typed
                if summary is None or summary == is_summary
            ]
            represented.update(("".join(emitted), "\n\n".join(emitted)))
        native = "".join(group.native_parts)
        if native:
            if native not in represented:
                yield from self._remaining_part(
                    output, group, ("native", 0), native, summary=False
                )
            group.visible_parts[("native", 0)] = native
            represented.add(native)
        combined = "".join(text for _, text, _ in typed)
        combined_lines = "\n\n".join(text for _, text, _ in typed)
        for summary in (False, True):
            parts = [
                (key, text) for key, text, is_summary in typed if is_summary == summary
            ]
            covered = (
                combined in represented
                or combined_lines in represented
                or "".join(text for _, text in parts) in represented
            )
            for key, text in parts:
                if not covered and text not in represented:
                    yield from self._remaining_part(
                        output, group, key, text, summary=summary
                    )
                group.visible_parts[key] = text
                represented.add(text)

    def _remaining_part(
        self,
        output: ChatStreamOutput,
        group: _ReasoningGroup,
        key: tuple[str, int],
        text: str,
        *,
        summary: bool,
    ) -> Iterator[str]:
        remaining = text.removeprefix(group.visible_parts.get(key, ""))
        if remaining:
            yield output.emit_reasoning_delta(remaining, summary=summary)

    def finish(
        self, output: ChatStreamOutput, *, completed: bool = True
    ) -> Iterator[str]:
        group, self._group = self._group, None
        self._text_source = None
        if group is not None:
            yield from self._remaining_readable(output, group)
            if completed:
                yield from output.complete_reasoning_record(
                    [
                        value
                        for detail in group.details
                        for value in _reasoning_detail_opaque(detail)
                    ]
                )


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


def _reasoning_detail_opaque(detail: Any) -> list[str]:
    kind = str(_field(detail, "type") or "").lower()
    keys = (
        ("data", "signature")
        if "encrypted" in kind or "redacted" in kind
        else ("signature",)
        if kind.startswith("reasoning.")
        else ()
    )
    return [
        value for key in keys if isinstance(value := _field(detail, key), str) and value
    ]


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
