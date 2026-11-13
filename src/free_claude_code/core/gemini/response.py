"""Gemini response serialization and non-streaming aggregation."""

import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

from ..failures import ExecutionFailure, FailureKind
from ..history_replay import (
    ReplayRecord,
    encode_replay,
    protected_reasoning,
    readable_reasoning,
)
from ..json_types import JsonObject
from ..recovery_limits import RECOVERY_RETENTION_BYTES
from .errors import gemini_error_payload
from .replay import CallReplay, ReplayBundle, encode_bundle


def gemini_parts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        part
        for candidate in payload.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
    ]


def gemini_finished(payload: Mapping[str, Any]) -> bool:
    return any(
        candidate.get("finishReason") for candidate in payload.get("candidates", [])
    )


def gemini_frame(payload: Mapping[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"


def conversion_failure(message: str) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM, status_code=502, message=message, retryable=False
    )


def function_arguments(arguments: str) -> JsonObject:
    try:
        value = json.loads(arguments)
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise conversion_failure(
            "Upstream function arguments are not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise conversion_failure("Upstream function arguments must be a JSON object")
    return value


class GeminiOutput:
    """Serialize native candidate deltas and one final opaque replay bundle."""

    def __init__(self, response_id: str, model: str, *, include_thoughts: bool) -> None:
        self.response_id = response_id
        self.model = model
        self.include_thoughts = include_thoughts
        self.records: list[ReplayRecord] = []
        self.calls: list[CallReplay] = []
        self.completed = False
        self._thought_digest = sha256()
        self._thought_length = 0
        self._ranges: dict[int, list[tuple[int, int]]] = {}
        self._record_sources: dict[int, int] = {}
        self._record_sizes: dict[int, int] = {}
        self._retained_bytes = 0

    def _retain(self, size: int) -> None:
        self._retained_bytes += size
        if self._retained_bytes > RECOVERY_RETENTION_BYTES:
            raise conversion_failure(
                "Protected reasoning exceeded the continuation retention limit."
            )

    def capture(self, record: ReplayRecord, *, source: int | None = None) -> None:
        if source is not None and not readable_reasoning(record.native):
            self.discard(source)
        if not protected_reasoning(record.native):
            if source is not None:
                self.discard(source)
            return
        size = len(encode_replay(record).encode("utf-8"))
        index = self._record_sources.get(source) if source is not None else None
        if index is None:
            index = len(self.records)
            if source is not None:
                self._record_sources[source] = index
            self.records.append(record)
        else:
            self.records[index] = record
        self._retain(size - self._record_sizes.get(index, 0))
        self._record_sizes[index] = size

    def discard(self, source: int) -> None:
        """Release provisional echo ownership for an unsigned or abandoned record."""
        self._retain(-24 * len(self._ranges.pop(source, ())))

    def order_records(self, sources: list[int]) -> None:
        """Publish source-ordered records with their echo references kept intact."""
        order = [
            self._record_sources[source]
            for source in sources
            if source in self._record_sources
        ]
        assert len(order) == len(self.records)
        positions = {old: new for new, old in enumerate(order)}
        self.records = [self.records[index] for index in order]
        self._record_sources = {
            source: positions[index] for source, index in self._record_sources.items()
        }
        self._record_sizes = {
            positions[index]: size for index, size in self._record_sizes.items()
        }

    def capture_call(self, record: CallReplay) -> None:
        self._retain(
            len(
                encode_bundle(
                    ReplayBundle(self.response_id, calls=(record,), protected_echo=())
                )
            )
        )
        self.calls.append(record)

    def payload(self, candidate: JsonObject) -> JsonObject:
        return {
            "responseId": self.response_id,
            "modelVersion": self.model,
            "candidates": [{"index": 0, **candidate}],
        }

    def part(self, part: JsonObject, *, source: int | None = None) -> list[str]:
        if self.completed or (
            part.get("thought") is True and not self.include_thoughts
        ):
            return []
        if part.get("thought") is True and isinstance(text := part.get("text"), str):
            self._thought_digest.update(text.encode("utf-8"))
            start = self._thought_length
            self._thought_length += len(text)
            if source is not None and text:
                spans = self._ranges.setdefault(source, [])
                if spans and spans[-1][1] == start:
                    spans[-1] = (spans[-1][0], self._thought_length)
                else:
                    self._retain(24)
                    spans.append((start, self._thought_length))
        return [
            gemini_frame(self.payload({"content": {"role": "model", "parts": [part]}}))
        ]

    def finish(self, reason: str, usage: JsonObject | None = None) -> list[str]:
        if self.completed:
            return []
        candidate: JsonObject = {"finishReason": reason}
        if self.records or self.calls:
            candidate["content"] = {
                "role": "model",
                "parts": [{"thoughtSignature": self._encoded_bundle()}],
            }
        payload = self.payload(candidate)
        if usage:
            payload["usageMetadata"] = usage
        self.completed = True
        return [gemini_frame(payload)]

    def handoff(self) -> list[str]:
        """Release completed source metadata before the public owner's tool handoff."""
        if not (self.records or self.calls) or self.completed:
            return []
        return self.part({"thoughtSignature": self._encoded_bundle()})

    def _encoded_bundle(self) -> str:
        spans = tuple(
            sorted(
                (
                    (index, start, end)
                    for source, index in self._record_sources.items()
                    for start, end in self._ranges.get(source, ())
                ),
                key=lambda span: span[1],
            )
        )
        return encode_bundle(
            ReplayBundle(
                self.response_id,
                tuple(self.records),
                tuple(self.calls),
                self._thought_digest.hexdigest() if spans else None,
                spans,
            )
        )

    def failure(self, status_code: int, message: str) -> list[str]:
        if self.completed:
            return []
        self.completed = True
        return [gemini_frame(gemini_error_payload(status_code, message))]


class GeminiResponseAccumulator:
    """Aggregate the same public deltas served by streamGenerateContent."""

    def __init__(self) -> None:
        self.response: dict[str, Any] = {}
        self.parts: list[dict[str, Any]] = []
        self.finished = False

    def feed(self, payload: dict[str, Any]) -> None:
        error = payload.get("error")
        if isinstance(error, dict):
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=int(error.get("code", 502)),
                message=str(error.get("message", "Generation failed")),
                retryable=False,
            )
        if self.finished and payload.get("candidates"):
            raise conversion_failure("Upstream emitted data after Gemini completion")
        for key, value in payload.items():
            if key != "candidates":
                self.response[key] = deepcopy(value)
        for candidate in payload.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if (
                    self.parts
                    and set(part) <= {"text", "thought"}
                    and "text" in part
                    and set(self.parts[-1]) <= {"text", "thought"}
                    and "text" in self.parts[-1]
                    and part.get("thought", False)
                    == self.parts[-1].get("thought", False)
                ):
                    self.parts[-1]["text"] += part["text"]
                else:
                    self.parts.append(deepcopy(part))
            target = self.response.setdefault("candidates", [{"index": 0}])[0]
            target.update(
                {
                    key: deepcopy(value)
                    for key, value in candidate.items()
                    if key != "content"
                }
            )
            if candidate.get("finishReason"):
                self.finished = True
        if self.parts:
            self.response["candidates"][0]["content"] = {
                "role": "model",
                "parts": self.parts,
            }

    def result(self) -> dict[str, Any]:
        if not self.finished:
            raise conversion_failure("Gemini stream ended without completion")
        return self.response
