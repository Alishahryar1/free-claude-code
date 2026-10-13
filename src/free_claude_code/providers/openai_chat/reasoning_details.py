"""OpenRouter-format structured reasoning replay and stream conversion."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from free_claude_code.core.history_replay import readable_reasoning

from .stream_output import ChatStreamOutput


@dataclass(slots=True)
class _ReasoningGroup:
    details: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[tuple[object, object], int] = field(default_factory=dict)
    native_parts: list[str] = field(default_factory=list)
    visible_parts: dict[tuple[str, int], str] = field(default_factory=dict)
    output_sources: dict[tuple[str, int], int] = field(default_factory=dict)
    output_reasoning: bool = True
    after_content: bool = False
    # Retain native phase transitions without using that field for publication.
    native_primary: bool | None = None


class StructuredReasoningStream:
    """Collect replay groups independently of their visible wire blocks."""

    def __init__(self) -> None:
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
        return iter(())

    def events(
        self,
        delta: Any,
        output: ChatStreamOutput,
        *,
        native_reasoning: str | None,
        output_reasoning: bool = True,
    ) -> Iterator[str]:
        details = _reasoning_details(delta)
        if (
            native_reasoning
            and self._group is not None
            and self._group.native_primary
            and self._group.after_content
        ):
            yield from self.before_reasoning(output)
            yield from output.close_content_blocks()
        if self._group is None and (
            (output_reasoning and native_reasoning)
            or any(
                (output_reasoning and _reasoning_detail_text(detail))
                or _reasoning_detail_opaque(detail)
                for detail in details
            )
        ):
            self._group = _ReasoningGroup(output_reasoning=output_reasoning)
        group = self._group
        if group is None:
            return
        if group.native_primary is None:
            if any(
                _field(detail, "type") == "reasoning.text"
                and _reasoning_detail_text(detail)
                for detail in details
            ):
                group.native_primary = False
            elif native_reasoning:
                group.native_primary = True
            elif any(_reasoning_detail_text(detail) for detail in details):
                group.native_primary = False
        if native_reasoning is not None:
            # This may be an aggregate alias of signed details arriving later.
            group.native_parts.append(native_reasoning)
        visible_parts: list[tuple[tuple[str, int], str, bool]] = []
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
                    output_reasoning
                    and not group.after_content
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
            source = _output_source(group, key)
            group.output_sources[key] = source
            yield from output.ensure_reasoning_block(source=source)
            yield output.emit_reasoning_delta(text, summary=summary, source=source)
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
        for key, text, summary in typed:
            yield from self._remaining_part(output, group, key, text, summary=summary)
            group.visible_parts[key] = text
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
            source = _output_source(group, key)
            group.output_sources[key] = source
            yield from output.begin_reasoning_record(source=source)
            yield output.emit_reasoning_delta(remaining, summary=summary, source=source)

    def finish(
        self, output: ChatStreamOutput, *, completed: bool = True
    ) -> Iterator[str]:
        group, self._group = self._group, None
        if group is not None:
            yield from self._remaining_readable(output, group)
            for slot, detail in enumerate(group.details):
                if detail.get("type") != "reasoning.text":
                    continue
                key = ("detail", slot)
                source = group.output_sources.get(key, slot + 1)
                opaque = _reasoning_detail_opaque(detail) if completed else []
                if (
                    group.output_reasoning
                    and detail.get("text") == ""
                    and opaque
                    and key not in group.output_sources
                ):
                    yield from output.begin_reasoning_record(source=source)
                    yield output.emit_reasoning_delta("", source=source)
                yield from output.complete_reasoning_record(opaque, source=source)
            yield from output.complete_reasoning_record(
                [
                    value
                    for detail in group.details
                    if detail.get("type") != "reasoning.text"
                    for value in _reasoning_detail_opaque(detail)
                ]
                if completed
                else []
            )


def _output_source(group: _ReasoningGroup, key: tuple[str, int]) -> int:
    if key in group.output_sources:
        return group.output_sources[key]
    if key[0] == "detail" and group.details[key[1]].get("type") == "reasoning.text":
        return key[1] + 1
    return 0


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
