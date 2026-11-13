"""Remove one logical replay prefix while preserving source parts and events."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import simplejson

from .failures import ExecutionFailure, FailureKind
from .gemini.response import gemini_finished, gemini_parts
from .recovery_limits import RECOVERY_RETENTION_BYTES

type PartKey = tuple[int | str, int]


@dataclass(slots=True)
class _Part:
    order: tuple[int, ...]
    text: str = ""
    seen: int = 0
    closed: bool = False
    removed: int = 0
    fixed: bool = False


@dataclass(slots=True)
class _Text:
    key: PartKey
    order: tuple[int, ...]
    value: dict[str, Any]
    field: str
    delta: bool = False
    part: _Part | None = None
    offset: int = 0


@dataclass(slots=True)
class _Frame:
    data: dict[str, Any]
    texts: list[_Text]


class ContinuationOverlap:
    """Own unpublished replay evidence, never the public response transcript."""

    def __init__(
        self, expected: str, *, channel: Literal["text", "thinking"] = "text"
    ) -> None:
        self._channel = channel
        self._expected = expected
        self._pending = bool(expected)
        self._parts: dict[PartKey, _Part] = {}
        self._ids: dict[int, str] = {}
        self._frames: list[_Frame] = []
        self._frame_bytes = 0
        self._frontier: tuple[int, ...] | None = None

    def feed(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = deepcopy(payload)
        block = data.get("content_block", {})
        field = "thinking" if self._channel == "thinking" else "text"
        if (
            data.get("type") == "content_block_start"
            and block.get("type") == field
            and block.get(field)
        ):
            text, block[field] = block[field], ""
            return [
                *self._feed(data),
                *self._feed(
                    {
                        "type": "content_block_delta",
                        "index": data["index"],
                        "delta": {"type": f"{field}_delta", field: text},
                    }
                ),
            ]
        return self._feed(data)

    def _feed(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        kind = data.get("type", "")
        if kind in {"error", "response.error", "response.failed"} or "error" in data:
            self._frames.clear()
            self._frame_bytes = 0
            return [data]
        if not self._expected:
            return self._render(_Frame(data, self._texts(data)))
        texts = self._texts(data)
        frame = _Frame(data, texts)
        terminal = (
            gemini_finished(data)
            or any(
                choice.get("finish_reason") is not None
                for choice in data.get("choices", [])
            )
            or kind
            in {
                "response.completed",
                "response.incomplete",
                "message_stop",
            }
            or (
                kind == "message_delta"
                and data.get("delta", {}).get("stop_reason") is not None
            )
        )
        if not self._pending:
            if kind in {"response.completed", "response.incomplete"}:
                self._check_terminal_order(texts)
            for text in texts:
                self._settled_text(text)
            return self._render(frame)

        if kind in {"response.completed", "response.incomplete"}:
            supplied = {text.key for text in texts}
            if any(
                part.text and key not in supplied for key, part in self._parts.items()
            ):
                self._inconsistent()

        for text in texts:
            part = self._parts.setdefault(text.key, _Part(text.order))
            part.order = text.order
            text.part = part
            value = text.value.get(text.field, "")
            text.offset = part.seen if text.delta else 0
            if text.delta:
                part.text = (part.text + value)[: len(self._expected) + 1]
                part.seen += len(value)
            else:
                if not value.startswith(part.text):
                    self._inconsistent()
                part.text = value[: len(self._expected) + 1]
                part.seen = len(value)
            if kind.endswith(".done"):
                part.closed = True
        self._close_parts(data, terminal=terminal)
        ordered = sorted(self._parts.items(), key=lambda entry: entry[1].order)
        candidate = ""
        candidates: list[tuple[PartKey, _Part]] = []
        for key, part in ordered:
            candidates.append((key, part))
            candidate += part.text[: len(self._expected) + 1 - len(candidate)]
            if not part.closed or len(candidate) > len(self._expected):
                break
        if terminal or candidate == self._expected or candidate not in self._expected:
            self._resolve(candidate, candidates)
            frames, self._frames = self._frames, []
            self._frame_bytes = 0
            self._check_size()
            return [
                event for ready in [*frames, frame] for event in self._render(ready)
            ]

        if self._frames or any(text.value.get(text.field) for text in texts):
            self._frames.append(frame)
            self._frame_bytes += self._size(data)
            self._check_size()
            return []
        self._check_size()
        return self._render(frame)

    def _identity(self, data: dict[str, Any], item: dict[str, Any]) -> int | str:
        index = data.get("output_index", 0)
        identity = item.get("id") or data.get("item_id") or self._ids.get(index, index)
        if self._pending and isinstance(identity, str) and "output_index" in data:
            self._ids[index] = identity
        return identity

    def _texts(self, data: dict[str, Any]) -> list[_Text]:
        if "choices" in data:
            texts = []
            for choice in data["choices"]:
                delta = choice.get("delta") or {}
                fields = (
                    ("content",)
                    if self._channel == "text"
                    else ("reasoning_content", "reasoning")
                )
                texts.extend(
                    _Text(("chat", 0), (0, 0), delta, field, delta=True)
                    for field in fields
                    if isinstance(delta.get(field), str)
                )
                if self._channel == "thinking":
                    for detail in delta.get("reasoning_details", []):
                        field = (
                            "summary"
                            if detail.get("type") == "reasoning.summary"
                            else "text"
                        )
                        if isinstance(detail.get(field), str):
                            texts.append(
                                _Text(("chat", 0), (0, 0), detail, field, delta=True)
                            )
            return texts
        if "candidates" in data:
            return [
                _Text(("gemini", 0), (0, 0), part, "text", delta=True)
                for part in gemini_parts(data)
                if "text" in part
                and bool(part.get("thought")) == (self._channel == "thinking")
            ]
        kind = data.get("type", "")
        index = data.get("index", 0)
        field = "thinking" if self._channel == "thinking" else "text"
        if kind == "content_block_start":
            block = data.get("content_block", {})
            return (
                [_Text((index, 0), (index, 0), block, field)]
                if block.get("type") == field
                else []
            )
        if kind == "content_block_delta":
            delta = data.get("delta", {})
            return (
                [_Text((index, 0), (index, 0), delta, field, delta=True)]
                if delta.get("type") == f"{field}_delta"
                else []
            )
        if kind in {"response.completed", "response.incomplete"}:
            texts = []
            for position, item in enumerate(data.get("response", {}).get("output", [])):
                texts.extend(self._item_texts(item, item.get("id", position), position))
            return texts
        item = data.get("item", {})
        if kind in {"response.output_item.added", "response.output_item.done"}:
            if item.get("type") != (
                "message" if self._channel == "text" else "reasoning"
            ):
                return []
            identity = self._identity(data, item)
            return self._item_texts(item, identity, data.get("output_index", 0))
        summary = kind.startswith("response.reasoning_summary_")
        text_event = (
            "response.output_text"
            if self._channel == "text"
            else "response.reasoning_summary_text"
            if summary
            else "response.reasoning_text"
        )
        part_event = (
            "response.reasoning_summary_part" if summary else "response.content_part"
        )
        if summary and self._channel == "text":
            return []
        if kind not in {
            f"{text_event}.delta",
            f"{text_event}.done",
            f"{part_event}.added",
            f"{part_event}.done",
        }:
            return []
        position = data.get("summary_index" if summary else "content_index", 0)
        key = (self._identity(data, {}), position * 2 + int(summary))
        order = (data.get("output_index", 0), int(summary), position)
        if kind in {f"{text_event}.delta", f"{text_event}.done"}:
            delta = kind.endswith(".delta")
            return [_Text(key, order, data, "delta" if delta else "text", delta=delta)]
        part = data.get("part", {})
        expected = (
            "output_text"
            if self._channel == "text"
            else "summary_text"
            if summary
            else "reasoning_text"
        )
        return [_Text(key, order, part, "text")] if part.get("type") == expected else []

    def _item_texts(
        self, item: dict[str, Any], identity: int | str, index: int
    ) -> list[_Text]:
        if item.get("type") != ("message" if self._channel == "text" else "reasoning"):
            return []
        fields = (
            (("content", "output_text"),)
            if self._channel == "text"
            else (("content", "reasoning_text"), ("summary", "summary_text"))
        )
        return [
            _Text(
                (identity, position * 2 + rank), (index, rank, position), part, "text"
            )
            for rank, (field, kind) in enumerate(fields)
            for position, part in enumerate(item.get(field) or [])
            if part.get("type") == kind
        ]

    def _close_parts(self, data: dict[str, Any], *, terminal: bool) -> None:
        kind = data.get("type")
        identity = (
            self._identity(data, data.get("item", {}))
            if kind == "response.output_item.done"
            and data.get("item", {}).get("type")
            == ("message" if self._channel == "text" else "reasoning")
            else data.get("index")
            if kind == "content_block_stop"
            else None
        )
        for key, part in self._parts.items():
            if terminal or key[0] == identity:
                part.closed = True

    def _resolve(self, candidate: str, candidates: list[tuple[PartKey, _Part]]) -> None:
        overlap = self._overlap_size(candidate)
        remaining = len(candidate)
        self._pending = False
        self._parts = {}
        for key, part in candidates:
            if not part.text or not remaining:
                continue
            part.text = part.text[:remaining]
            remaining -= len(part.text)
            part.removed = min(overlap, len(part.text))
            overlap -= part.removed
            part.fixed = bool(remaining)
            self._parts[key] = part
            self._frontier = part.order
        identities = {key[0] for key in self._parts}
        self._ids = {
            index: value for index, value in self._ids.items() if value in identities
        }

    def _overlap_size(self, candidate: str) -> int:
        if self._channel == "thinking" and self._expected.startswith(candidate):
            return len(candidate)
        for size in range(min(len(self._expected), len(candidate)), 0, -1):
            if self._expected.endswith(candidate[:size]):
                return size
        return 0

    def _settled_text(self, text: _Text) -> None:
        value = text.value.get(text.field, "")
        part = self._parts.get(text.key)
        if part is None:
            if value and self._precedes_prefix(text):
                self._inconsistent()
            return
        text.part = part
        if text.delta:
            if value and part.fixed:
                self._inconsistent()
            text.offset = part.seen
            part.seen += len(value)
        else:
            if not value.startswith(part.text) or (
                part.fixed and len(value) != part.seen
            ):
                self._inconsistent()
            part.seen = max(part.seen, len(value))

    def _precedes_prefix(self, text: _Text) -> bool:
        if len(text.order) == 3:
            # Responses arrays advance independently within the same item.
            keys = [key for key in self._parts if key[0] == text.key[0]]
            if keys:
                if text.key[0] != next(reversed(self._parts))[0]:
                    return True
                return any(
                    key[1] % 2 == text.key[1] % 2 and text.key[1] < key[1]
                    for key in keys
                )
        return self._frontier is not None and text.order < self._frontier

    def _check_terminal_order(self, texts: list[_Text]) -> None:
        keys = list(self._parts)
        observed = [text.key for text in texts if text.value.get(text.field)]
        items = list(dict.fromkeys(key[0] for key in keys))
        observed_items = list(dict.fromkeys(key[0] for key in observed))
        if observed_items[: len(items)] != items:
            self._inconsistent()
        for identity in items[:-1]:
            if [key for key in observed if key[0] == identity] != [
                key for key in keys if key[0] == identity
            ]:
                self._inconsistent()
        # A final snapshot groups arrays rather than preserving event arrival
        # order. Require each array's matched prefix, allowing its later parts.
        for identity, array in {(key[0], key[1] % 2) for key in keys}:
            expected = [
                key for key in keys if key[0] == identity and key[1] % 2 == array
            ]
            actual = [
                key for key in observed if key[0] == identity and key[1] % 2 == array
            ]
            if actual[: len(expected)] != expected:
                self._inconsistent()
        # Terminal positions may have shifted when the tool gate removed an item.
        for text in texts:
            if text.key not in self._parts:
                text.order = (
                    (self._frontier[0] + 1, 0) if self._frontier else text.order
                )

    @staticmethod
    def _render(frame: _Frame) -> list[dict[str, Any]]:
        for text in frame.texts:
            if text.part is not None:
                removed = (
                    max(0, text.part.removed - text.offset)
                    if text.delta
                    else text.part.removed
                )
                text.value[text.field] = text.value.get(text.field, "")[removed:]
        if (
            frame.texts
            and all(text.delta for text in frame.texts)
            and not any(text.value.get(text.field) for text in frame.texts)
        ):
            if "choices" in frame.data:
                for choice in frame.data["choices"]:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason") is not None or any(
                        value
                        for key, value in delta.items()
                        if key
                        not in {
                            "content",
                            "reasoning",
                            "reasoning_content",
                            "reasoning_details",
                            "role",
                        }
                    ):
                        return [frame.data]
                return []
            if "candidates" in frame.data:
                for candidate in frame.data["candidates"]:
                    content = candidate.get("content", {})
                    content["parts"] = [
                        part
                        for part in content.get("parts", [])
                        if part.get("text")
                        or any(key not in {"text", "thought"} for key in part)
                    ]
                if gemini_finished(frame.data) or gemini_parts(frame.data):
                    return [frame.data]
            return []
        return [frame.data]

    @staticmethod
    def _size(value: object) -> int:
        return len(
            simplejson.dumps(value, use_decimal=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )

    def _check_size(self) -> None:
        retained = self._frame_bytes + self._size(self._ids)
        retained += sum(
            self._size((key, part.order, part.text, part.seen))
            for key, part in self._parts.items()
        )
        if retained > RECOVERY_RETENTION_BYTES:
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message="The continuation replay exceeded the retained-data limit.",
                retryable=False,
            )

    @staticmethod
    def _inconsistent() -> None:
        raise ExecutionFailure(
            kind=FailureKind.UPSTREAM,
            status_code=502,
            message="The continuation changed an already observed output prefix.",
            retryable=False,
        )
