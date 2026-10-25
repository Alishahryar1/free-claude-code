"""Bounded-memory estimate of the Chat input LM Studio will actually receive."""

import json
from collections.abc import Iterator, Mapping
from typing import Any

from free_claude_code.core.token_estimation import (
    estimate_image_tokens,
    estimate_text_tokens,
)

_CHUNK_CHARS = 16 * 1024


def _text_costs(text: str) -> Iterator[int]:
    for start in range(0, len(text), _CHUNK_CHARS):
        yield estimate_text_tokens(text[start : start + _CHUNK_CHARS])


def _json_costs(value: Any) -> Iterator[int]:
    """Visit JSON leaves without serializing a whole schema or a large string."""
    if isinstance(value, str):
        yield 2  # quotes
        for start in range(0, len(value), _CHUNK_CHARS):
            escaped = json.dumps(
                value[start : start + _CHUNK_CHARS], ensure_ascii=False
            )[1:-1]
            yield from _text_costs(escaped)
    elif isinstance(value, Mapping):
        yield 2  # braces
        for key, child in value.items():
            yield from _json_costs(key)
            yield 2  # colon and separator
            yield from _json_costs(child)
    elif isinstance(value, list):
        yield 2  # brackets
        for child in value:
            yield 1
            yield from _json_costs(child)
    else:
        yield from _text_costs(json.dumps(value))


def _content_costs(content: Any) -> Iterator[int]:
    if isinstance(content, str):
        yield from _text_costs(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                size = None
                if isinstance(url, str) and url.startswith("data:"):
                    comma = url.find(",")
                    if comma >= 0 and url[max(0, comma - 7) : comma] == ";base64":
                        size = len(url) - comma - 1
                yield estimate_image_tokens(size)
            elif isinstance(text := part.get("text", part.get("refusal")), str):
                yield from _text_costs(text)


def _input_costs(body: Mapping[str, Any]) -> Iterator[int]:
    yield 3  # assistant response framing
    for message in body.get("messages", []):
        yield 4  # approximate role/message separators
        yield from _content_costs(message.get("content"))
        for field in ("reasoning_content", "reasoning"):
            if isinstance(text := message.get(field), str):
                yield from _text_costs(text)
        for detail in message.get("reasoning_details", []):
            if detail.get("type") in {"reasoning.text", "reasoning.summary"}:
                text = detail.get("summary", detail.get("text"))
                if isinstance(text, str):
                    yield from _text_costs(text)
        for field in ("name", "tool_call_id", "refusal"):
            if isinstance(text := message.get(field), str):
                yield from _text_costs(text)
        for call in message.get("tool_calls", []):
            yield 4
            for text in (call.get("id"), *call.get("function", {}).values()):
                if isinstance(text, str):
                    yield from _text_costs(text)
    for field in ("tools", "response_format"):
        if field in body:
            yield from _json_costs(body[field])


def estimate_prepared_input_tokens(body: Mapping[str, Any], *, budget: int) -> int:
    """Count every input tail, stopping only once rejection is already decided.

    Chunk boundaries and framing make this an approximation, not a model-specific
    tokenizer. Each tokenizer call and serialized string fragment stays bounded.
    Signatures, metadata, transport headers and generation controls are excluded.
    """
    total = 0
    for cost in _input_costs(body):
        total += cost
        if total > budget:
            break
    return total
