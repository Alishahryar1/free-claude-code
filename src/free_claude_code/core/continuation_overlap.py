"""Remove one logical replay prefix while preserving source parts and events."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import simplejson

from .delivered_response import RECOVERY_RETENTION_BYTES
from .failures import ExecutionFailure, FailureKind

type PartKey = tuple[int | str, int]


@dataclass(slots=True)
class _Part:
    order: tuple[int, int]
    text: str = ""
    seen: int = 0
    closed: bool = False
    removed: int = 0
    fixed: bool = False


@dataclass(slots=True)
class _Text:
    key: PartKey
    order: tuple[int, int]
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

    def __init__(self, expected: str) -> None:
        self._expected = expected
        self._pending = bool(expected)
        self._parts: dict[PartKey, _Part] = {}
        self._ids: dict[int, str] = {}
        self._frames: list[_Frame] = []
        self._frame_bytes = 0
        self._frontier: tuple[int, int] | None = None

    def feed(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = deepcopy(payload)
        block = data.get("content_block", {})
        if (
            data.get("type") == "content_block_start"
            and block.get("type") == "text"
            and block.get("text")
        ):
            text, block["text"] = block["text"], ""
            return [
                *self._feed(data),
                *self._feed(
                    {
                        "type": "content_block_delta",
                        "index": data["index"],
                        "delta": {"type": "text_delta", "text": text},
                    }
                ),
            ]
        return self._feed(data)

    def _feed(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        kind = data.get("type", "")
        if kind in {"error", "response.error", "response.failed"}:
            self._frames.clear()
            self._frame_bytes = 0
            return [data]
        if not self._expected:
            return self._render(_Frame(data, self._texts(data)))
        texts = self._texts(data)
        frame = _Frame(data, texts)
        terminal = kind in {
            "response.completed",
            "response.incomplete",
            "message_stop",
        } or (
            kind == "message_delta"
            and data.get("delta", {}).get("stop_reason") is not None
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
        if terminal or candidate not in self._expected:
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
        kind = data.get("type", "")
        index = data.get("index", 0)
        if kind == "content_block_start":
            block = data.get("content_block", {})
            return (
                [_Text((index, 0), (index, 0), block, "text")]
                if block.get("type") == "text"
                else []
            )
        if kind == "content_block_delta":
            delta = data.get("delta", {})
            return (
                [_Text((index, 0), (index, 0), delta, "text", delta=True)]
                if delta.get("type") == "text_delta"
                else []
            )
        if kind in {"response.completed", "response.incomplete"}:
            texts = []
            for position, item in enumerate(data.get("response", {}).get("output", [])):
                texts.extend(self._item_texts(item, item.get("id", position), position))
            return texts
        item = data.get("item", {})
        if kind in {"response.output_item.added", "response.output_item.done"}:
            if item.get("type") != "message":
                return []
            identity = self._identity(data, item)
            return self._item_texts(item, identity, data.get("output_index", 0))
        if kind not in {
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
        }:
            return []
        position = data.get("content_index", 0)
        key = (self._identity(data, {}), position)
        order = (data.get("output_index", 0), position)
        if kind in {"response.output_text.delta", "response.output_text.done"}:
            delta = kind.endswith(".delta")
            return [_Text(key, order, data, "delta" if delta else "text", delta=delta)]
        part = data.get("part", {})
        return (
            [_Text(key, order, part, "text")]
            if part.get("type") == "output_text"
            else []
        )

    @staticmethod
    def _item_texts(
        item: dict[str, Any], identity: int | str, index: int
    ) -> list[_Text]:
        if item.get("type") != "message":
            return []
        return [
            _Text((identity, position), (index, position), part, "text")
            for position, part in enumerate(item.get("content", []))
            if part.get("type") == "output_text"
        ]

    def _close_parts(self, data: dict[str, Any], *, terminal: bool) -> None:
        kind = data.get("type")
        identity = (
            self._identity(data, data.get("item", {}))
            if kind == "response.output_item.done"
            and data.get("item", {}).get("type") == "message"
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
        for size in range(min(len(self._expected), len(candidate)), 0, -1):
            if self._expected.endswith(candidate[:size]):
                return size
        return 0

    def _settled_text(self, text: _Text) -> None:
        value = text.value.get(text.field, "")
        part = self._parts.get(text.key)
        if part is None:
            if value and self._frontier is not None and text.order < self._frontier:
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

    def _check_terminal_order(self, texts: list[_Text]) -> None:
        keys = list(self._parts)
        observed = [text.key for text in texts if text.value.get(text.field)]
        if observed[: len(keys)] != keys:
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
            message="The continuation changed an already observed answer prefix.",
            retryable=False,
        )
