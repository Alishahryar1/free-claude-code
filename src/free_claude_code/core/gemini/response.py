"""Gemini response serialization and non-streaming aggregation."""

import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

from ..failures import ExecutionFailure, FailureKind
from ..history_replay import ReplayRecord, readable_reasoning
from ..json_types import JsonObject
from ..stream_delivery import current_stream_delivery
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
        self._has_thoughts = False
        self._delivery = current_stream_delivery()

    def capture(self, record: ReplayRecord) -> None:
        self.records.append(record)
        self._bind_replay()

    def replace_records(self, records: list[ReplayRecord]) -> None:
        self.records = records
        self._bind_replay()

    def _bind_replay(self) -> None:
        if self._delivery is not None:
            self._delivery.bind_replay_source(lambda: tuple(self.records))

    def _records(self) -> tuple[ReplayRecord, ...]:
        if self._delivery is not None:
            if self._delivery.replay_exhausted:
                raise conversion_failure(
                    "Completed reasoning exceeded the continuation retention limit."
                )
            return self._delivery.committed_replay + tuple(self.records)
        return tuple(self.records)

    def payload(self, candidate: JsonObject) -> JsonObject:
        return {
            "responseId": self.response_id,
            "modelVersion": self.model,
            "candidates": [{"index": 0, **candidate}],
        }

    def part(self, part: JsonObject) -> list[str]:
        if self.completed or (
            part.get("thought") is True and not self.include_thoughts
        ):
            return []
        if part.get("thought") is True and isinstance(text := part.get("text"), str):
            self._thought_digest.update(text.encode("utf-8"))
            self._has_thoughts = True
        return [
            gemini_frame(self.payload({"content": {"role": "model", "parts": [part]}}))
        ]

    def finish(self, reason: str, usage: JsonObject | None = None) -> list[str]:
        if self.completed:
            return []
        candidate: JsonObject = {"finishReason": reason}
        if self._records() or self.calls:
            bundle = self._bundle()
            candidate["content"] = {
                "role": "model",
                "parts": [{"thoughtSignature": encode_bundle(bundle)}],
            }
        payload = self.payload(candidate)
        if usage:
            payload["usageMetadata"] = usage
        self.completed = True
        return [gemini_frame(payload)]

    def handoff(self) -> list[str]:
        """Release completed source metadata before the public owner's tool handoff."""
        if not (self._records() or self.calls) or self.completed:
            return []
        bundle = self._bundle()
        return self.part({"thoughtSignature": encode_bundle(bundle)})

    def _bundle(self) -> ReplayBundle:
        records = self._records()
        covered = sha256()
        for record in records:
            for text, _ in readable_reasoning(record.native):
                covered.update(text.encode("utf-8"))
        digest = self._thought_digest.hexdigest()
        return ReplayBundle(
            self.response_id,
            records,
            tuple(self.calls),
            digest if self._has_thoughts and covered.hexdigest() == digest else None,
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
        if self.finished:
            raise conversion_failure("Upstream emitted data after Gemini completion")
        for key in ("responseId", "modelVersion", "usageMetadata", "promptFeedback"):
            if key in payload:
                self.response[key] = deepcopy(payload[key])
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
