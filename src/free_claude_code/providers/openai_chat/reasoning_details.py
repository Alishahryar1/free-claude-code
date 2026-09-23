"""OpenRouter-format structured reasoning replay and stream conversion."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal, cast

from free_claude_code.core.history_replay import readable_reasoning
from free_claude_code.core.json_types import JsonObject, JsonValue

from .stream_output import ChatStreamOutput


class StructuredReasoningStream:
    """Reconcile alternate plaintext reasoning representations for one stream."""

    def __init__(self) -> None:
        self._text_source: Literal["native", "details"] | None = None
        self._details: list[dict[str, Any]] = []
        self._slots: dict[tuple[object, object], int] = {}
        self._native_parts: list[str] = []

    def events(
        self,
        delta: Any,
        output: ChatStreamOutput,
        *,
        native_reasoning: str | None,
    ) -> Iterator[str]:
        """Emit plaintext once while preserving every opaque reasoning detail."""
        details = _reasoning_details(delta)
        if native_reasoning is not None:
            self._native_parts.append(native_reasoning)
        if self._text_source is None:
            if native_reasoning:
                self._text_source = "native"
            elif any(_reasoning_detail_text(detail) for detail in details):
                self._text_source = "details"

        if self._text_source == "native" and native_reasoning:
            yield from output.ensure_reasoning_block()
            yield output.emit_reasoning_delta(native_reasoning)

        for detail in details:
            if self._text_source == "details":
                text = _reasoning_detail_text(detail)
                if text:
                    yield from output.ensure_reasoning_block()
                    yield output.emit_reasoning_delta(text)

            if isinstance(detail, Mapping):
                value = deepcopy(dict(detail))
                identity = value.get("index", value.get("id"))
                key = (identity, value.get("type"))
                if isinstance(identity, str | int) and key in self._slots:
                    current = self._details[self._slots[key]]
                    for field, part in value.items():
                        if (
                            field
                            in {
                                "text",
                                "summary",
                                "content",
                                "reasoning",
                                "data",
                                "signature",
                            }
                            and isinstance(part, str)
                            and isinstance(current.get(field), str)
                        ):
                            current[field] += part
                        else:
                            current[field] = part
                else:
                    if isinstance(identity, str | int):
                        self._slots[key] = len(self._details)
                    self._details.append(value)

    def flush(self, output: ChatStreamOutput) -> Iterator[str]:
        if self._details:
            native: JsonObject = {
                "reasoning_details": cast(list[JsonValue], self._details)
            }
            if self._native_parts:
                native["reasoning_content"] = "".join(self._native_parts)
            yield from output.emit_reasoning_replay(native)
            self._details.clear()
            self._slots.clear()
        self._native_parts.clear()


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
