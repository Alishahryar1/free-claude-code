"""Chat SSE serialization, native relay and JSON aggregation."""

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

from ..failures import ExecutionFailure, FailureKind


def chat_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"


def chat_failure(message: str) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM, status_code=502, message=message, retryable=False
    )


class ChatOutput:
    def __init__(self, model: str, *, include_usage: bool = False) -> None:
        self.model = model
        self.response_id = f"chatcmpl-{uuid4().hex}"
        self.created = int(time.time())
        self.include_usage = include_usage
        self.completed = False

    def payload(self, choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }

    def delta(self, delta: dict[str, Any]) -> list[str]:
        if self.completed:
            raise chat_failure("Chat content arrived after completion")
        payload = self.payload([{"index": 0, "delta": delta, "finish_reason": None}])
        if self.include_usage:
            payload["usage"] = None
        return [chat_frame(payload)]

    def start(self) -> list[str]:
        return self.delta({"role": "assistant"})

    def text(self, text: str) -> list[str]:
        return self.delta({"content": text})

    def thought(
        self, text: str, *, summary: bool = False, source: str | None = None
    ) -> list[str]:
        if source is not None:
            return self.delta(
                {
                    "reasoning_details": [
                        {
                            "id": source,
                            "type": "reasoning.summary"
                            if summary
                            else "reasoning.text",
                            "summary" if summary else "text": text,
                        }
                    ]
                }
            )
        return self.delta(
            {"reasoning_details": [{"type": "reasoning.summary", "summary": text}]}
            if summary
            else {"reasoning_content": text}
        )

    def replay(self, carrier: str, *, source: str | None = None) -> list[str]:
        return self.delta(
            {
                "reasoning_details": [
                    {
                        "type": "reasoning.encrypted",
                        "data": carrier,
                        **({"id": source} if source is not None else {}),
                    }
                ]
            }
        )

    def tool(self, index: int, call_id: str, name: str, arguments: str) -> list[str]:
        return self.delta(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
            }
        )

    def finish(self, reason: str, usage: dict[str, Any] | None = None) -> list[str]:
        if self.completed:
            return []
        payload = self.payload([{"index": 0, "delta": {}, "finish_reason": reason}])
        if self.include_usage:
            payload["usage"] = None
        events = [chat_frame(payload)]
        if self.include_usage and usage is not None:
            events.append(chat_frame({**self.payload([]), "usage": usage}))
        events.append("data: [DONE]\n\n")
        self.completed = True
        return events


class NativeChatRelay:
    """Keep native chunks intact apart from the public model identity."""

    def __init__(self, model: str, *, choices: int = 1) -> None:
        self.model = model
        self.expected_choices = choices
        self.finished_choices: set[int] = set()
        self.completed = False

    @property
    def normal_stop_seen(self) -> bool:
        return self.finished_choices == set(range(self.expected_choices))

    def feed(self, payload: dict[str, Any]) -> str:
        if self.completed:
            raise chat_failure("Native Chat chunk arrived after completion")
        if payload.get("error"):
            raise chat_failure(
                str(payload["error"].get("message", "Chat generation failed"))
            )
        result = deepcopy(payload)
        result["model"] = self.model
        for choice in result.get("choices", []):
            index = choice.get("index")
            if not isinstance(index, int) or not 0 <= index < self.expected_choices:
                raise chat_failure("Native Chat returned an unexpected choice index")
            if index in self.finished_choices:
                raise chat_failure(
                    "Native Chat emitted a choice after its finish_reason"
                )
            if choice.get("finish_reason") is not None:
                self.finished_choices.add(index)
        return chat_frame(result)

    def finish(self) -> list[str]:
        if not self.normal_stop_seen:
            raise chat_failure("Native Chat stream ended before all choices completed")
        self.completed = True
        return ["data: [DONE]\n\n"]


class ChatResponseAccumulator:
    """Build a JSON completion from the exact stream delivered to the client."""

    def __init__(self) -> None:
        self.response: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.completed = False

    def feed(self, payload: dict[str, Any]) -> None:
        if payload.get("error"):
            raise chat_failure(
                str(payload["error"].get("message", "Chat generation failed"))
            )
        if self.completed:
            raise chat_failure("Chat data arrived after DONE")
        self.response.update(
            {
                key: deepcopy(value)
                for key, value in payload.items()
                if key not in {"choices", "object"}
            }
        )
        for choice in payload.get("choices", []):
            target = self.choices.setdefault(
                choice["index"],
                {
                    "index": choice["index"],
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": None,
                },
            )
            target.update(
                {
                    key: deepcopy(value)
                    for key, value in choice.items()
                    if key not in {"delta", "logprobs"}
                }
            )
            if choice.get("logprobs"):
                logprobs = target.setdefault("logprobs", {})
                for key, value in choice["logprobs"].items():
                    if value is None:
                        logprobs.setdefault(key, None)
                    elif isinstance(value, list):
                        if logprobs.get(key) is None:
                            logprobs.pop(key, None)
                        logprobs.setdefault(key, []).extend(deepcopy(value))
                    else:
                        logprobs[key] = deepcopy(value)
            message = target["message"]
            for key, value in (choice.get("delta") or {}).items():
                if value is None:
                    message.setdefault(key, None)
                    continue
                if message.get(key) is None:
                    message.pop(key, None)
                if key == "tool_calls":
                    calls = message.setdefault("tool_calls", [])
                    for call in value:
                        index = call["index"]
                        while len(calls) <= index:
                            calls.append({})
                        dest = calls[index]
                        for field, item in call.items():
                            if item is None:
                                continue
                            if field in {"function", "custom"}:
                                tool = dest.setdefault(field, {})
                                for name, text in item.items():
                                    if text is not None:
                                        tool[name] = tool.get(name, "") + text
                            elif field != "index":
                                dest[field] = deepcopy(item)
                elif key == "reasoning_details":
                    details = message.setdefault(key, [])
                    for detail in value:
                        field = {
                            "reasoning.text": "text",
                            "reasoning.summary": "summary",
                        }.get(detail.get("type"))
                        previous = details[-1] if details else {}
                        if (
                            field is not None
                            and previous.get("type") == detail.get("type")
                            and isinstance(previous.get(field), str)
                            and isinstance(detail.get(field), str)
                            and not previous.get("signature")
                            and not detail.get("signature")
                            and all(
                                name not in previous
                                or name not in detail
                                or previous[name] == detail[name]
                                for name in ("id", "index", "format")
                            )
                        ):
                            previous[field] += detail[field]
                            previous.update(
                                {
                                    name: deepcopy(part)
                                    for name, part in detail.items()
                                    if name != field
                                }
                            )
                        else:
                            # Encrypted records are complete opaque objects, even
                            # when a provider repeats an index across blocks.
                            details.append(deepcopy(detail))
                elif isinstance(value, str) and key != "role":
                    message[key] = (message.get(key) or "") + value
                elif isinstance(value, list):
                    message.setdefault(key, []).extend(deepcopy(value))
                elif isinstance(value, dict):
                    existing = message.setdefault(key, {})
                    for name, item in value.items():
                        if item is None:
                            existing.setdefault(name, None)
                            continue
                        if isinstance(item, str) and (
                            name in {"data", "transcript"}
                            or (
                                key == "function_call" and name in {"name", "arguments"}
                            )
                        ):
                            existing[name] = (existing.get(name) or "") + item
                        else:
                            existing[name] = deepcopy(item)
                else:
                    message[key] = deepcopy(value)

    def finish(self) -> None:
        self.completed = True

    def result(self) -> dict[str, Any]:
        if (
            not self.completed
            or not self.choices
            or any(
                choice.get("finish_reason") is None for choice in self.choices.values()
            )
        ):
            raise chat_failure("Chat stream ended without completion")
        return {
            **self.response,
            "object": "chat.completion",
            "choices": [self.choices[index] for index in sorted(self.choices)],
        }
