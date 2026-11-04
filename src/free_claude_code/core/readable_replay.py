"""Project editable replay fields from their exact public character spans."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Literal

from .failures import ExecutionFailure, FailureKind
from .history_replay import ReplayRecord, protected_reasoning
from .stream_recovery import RECOVERY_RETENTION_BYTES

type NativePath = tuple[str | int, ...]
type ReasoningKey = int | tuple[int, Literal["thinking", "summary"]]
type RetainedReasoning = Callable[[ReasoningKey], tuple[int, int] | None]


@dataclass(frozen=True, slots=True)
class _Span:
    path: NativePath
    native_start: int
    public_key: ReasoningKey
    public_start: int
    length: int


class ReadableReplay:
    """Keep only coordinates; the native record and public ledger own the text."""

    def __init__(self) -> None:
        self._spans: list[_Span] = []
        self._bytes = 0
        self._aliases: dict[NativePath, tuple[tuple[NativePath, ...], str]] = {}

    def add(
        self,
        path: NativePath,
        native_start: int,
        public_key: ReasoningKey,
        public_start: int,
        length: int,
    ) -> None:
        if not length:
            return
        span = _Span(path, native_start, public_key, public_start, length)
        if self._spans:
            last = self._spans[-1]
            if (
                last.path == path
                and last.public_key == public_key
                and last.native_start + last.length == native_start
                and last.public_start + last.length == public_start
            ):
                self._spans[-1] = replace(last, length=last.length + length)
                return
        self._bytes += len(str(path)) + 64
        if self._bytes > RECOVERY_RETENTION_BYTES:
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message="Reasoning replay mapping exceeded the continuation retention limit.",
                retryable=False,
            )
        self._spans.append(span)

    def alias(
        self,
        native: dict[str, Any],
        targets: Sequence[NativePath],
        sources: Sequence[NativePath],
    ) -> None:
        """Copy known same-record alternate fields, retaining their native types."""
        mapped = {span.path for span in self._spans}
        if not targets or not sources or any(path in mapped for path in targets):
            return
        if not all(path in mapped for path in sources):
            return
        for source_separator in ("", "\n", "\n\n"):
            source_ranges, source_text = _ranges(native, sources, source_separator)
            for target_separator in ("", "\n", "\n\n"):
                target_ranges, target_text = _ranges(native, targets, target_separator)
                if not source_text or source_text != target_text:
                    continue
                spans = list(self._spans)
                for target, target_start, target_end in target_ranges:
                    for source, source_start, _ in source_ranges:
                        for span in spans:
                            if span.path != source:
                                continue
                            start = max(target_start, source_start + span.native_start)
                            end = min(
                                target_end,
                                source_start + span.native_start + span.length,
                            )
                            if start < end:
                                self.add(
                                    target,
                                    start - target_start,
                                    span.public_key,
                                    span.public_start
                                    + start
                                    - source_start
                                    - span.native_start,
                                    end - start,
                                )
                if len(targets) == 1:
                    self._aliases[targets[0]] = (tuple(sources), source_separator)
                return

    def chat_aliases(self, native: dict[str, Any]) -> None:
        scalars = [
            (key,)
            for key in ("reasoning_content", "reasoning")
            if isinstance(native.get(key), str)
        ]
        details = native.get("reasoning_details") or []
        for summary in (None, False, True):
            paths = [
                (
                    "reasoning_details",
                    index,
                    "summary" if "summary" in detail else "text",
                )
                for index, detail in enumerate(details)
                if isinstance(detail, dict)
                and detail.get("type") in {"reasoning.text", "reasoning.summary"}
                and (
                    summary is None
                    or (detail.get("type") == "reasoning.summary") == summary
                )
                and isinstance(detail.get("summary", detail.get("text")), str)
            ]
            for scalar in scalars:
                for path in paths:
                    self.alias(native, [path], [scalar])
                    self.alias(native, [scalar], [path])
                self.alias(native, [scalar], paths)
                self.alias(native, paths, [scalar])

    def project(
        self, record: ReplayRecord, retained: RetainedReasoning
    ) -> ReplayRecord:
        if not self._spans or protected_reasoning(record.native):
            return record
        edits: dict[NativePath, list[tuple[int, int]]] = {}
        changed: set[NativePath] = set()
        for span in self._spans:
            bounds = retained(span.public_key)
            if bounds is None:
                continue
            removed, length = bounds
            ranges = edits.setdefault(span.path, [])
            start = max(span.public_start, removed)
            end = min(span.public_start + span.length, removed + length)
            if start != span.public_start or end != span.public_start + span.length:
                changed.add(span.path)
            if start < end:
                ranges.append(
                    (
                        span.native_start + start - span.public_start,
                        span.native_start + end - span.public_start,
                    )
                )
        if not changed:
            return record
        native = deepcopy(record.native)
        for path, ranges in edits.items():
            if path not in changed:
                continue
            parent: Any = native
            for key in path[:-1]:
                parent = parent[key]
            value = parent[path[-1]]
            parent[path[-1]] = "".join(
                value[start:end] for start, end in sorted(ranges)
            )
        for path, (sources, separator) in self._aliases.items():
            if path not in changed:
                continue
            values = []
            for source in sources:
                value: Any = native
                for key in source:
                    value = value[key]
                if value:
                    values.append(value)
            alias_parent: Any = native
            for key in path[:-1]:
                alias_parent = alias_parent[key]
            alias_parent[path[-1]] = separator.join(values)
        return replace(record, native=native)

    def coverage(self, removed: int, length: int) -> list[tuple[int, int]]:
        return [
            (
                max(span.public_start, removed) - removed,
                min(span.public_start + span.length, removed + length) - removed,
            )
            for span in self._spans
            if span.public_key == -1
            and max(span.public_start, removed)
            < min(span.public_start + span.length, removed + length)
        ]

    def observed_chat(self, native: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(native)
        paths = {span.path for span in self._spans}
        for name in ("reasoning_content", "reasoning"):
            if name in result and (name,) not in paths:
                result[name] = ""
        for index, detail in enumerate(result.get("reasoning_details") or []):
            for name in ("text", "summary", "content", "reasoning"):
                if (
                    isinstance(detail.get(name), str)
                    and ("reasoning_details", index, name) not in paths
                ):
                    detail[name] = ""
        return result


def _ranges(native: dict[str, Any], paths: Sequence[NativePath], separator: str):
    ranges = []
    values = []
    offset = 0
    for path in paths:
        value: Any = native
        for key in path:
            value = value[key]
        ranges.append((path, offset, offset + len(value)))
        values.append(value)
        offset += len(value) + len(separator)
    return ranges, separator.join(values)
