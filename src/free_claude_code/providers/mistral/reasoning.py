"""Mistral La Plateforme reasoning compatibility helpers."""

import json
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from typing import TypeGuard

import openai
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)

from free_claude_code.core.json_types import (
    JsonObject,
    JsonValue,
    load_json_value,
    wire_field,
)
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.providers.http import close_provider_stream

MISTRAL_REASONING_EFFORT = "high"

_REASONING_FIELD_NAMES = frozenset(
    {
        "reasoning_content",
        "reasoning_effort",
        "thinkchunk",
    }
)
_REJECTION_WORDS = ("unsupported", "unknown", "invalid", "forbidden", "extra")


def apply_mistral_reasoning_request_shape(
    body: JsonObject, *, reasoning: ReasoningPolicy
) -> None:
    """Apply Mistral's native reasoning request shape in-place."""
    if reasoning.control is ReasoningControl.OFF:
        body["reasoning_effort"] = "none"
    elif reasoning.requests_reasoning:
        body["reasoning_effort"] = MISTRAL_REASONING_EFFORT
    else:
        body.pop("reasoning_effort", None)


def apply_mistral_history_shape(body: JsonObject) -> None:
    """Encode the already projected scalar, including an empty included slot."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        replayed_reasoning = message.pop("reasoning_content", None)
        if isinstance(replayed_reasoning, str):
            message["content"] = _content_with_prepended_thinking(
                message.get("content"), replayed_reasoning
            )


def clone_body_without_mistral_reasoning(
    body: JsonObject,
) -> JsonObject | None:
    """Return a body clone with Mistral reasoning fields removed."""
    cloned = deepcopy(body)
    removed = cloned.pop("reasoning_effort", None) is not None

    if not removed:
        return None
    return cloned


def is_mistral_reasoning_rejection(error: Exception) -> bool:
    """Return whether an upstream error rejects Mistral reasoning request fields."""
    status_code = wire_field(error, "status_code")
    if not isinstance(error, openai.BadRequestError) and status_code not in (400, 422):
        return False

    error_body = wire_field(error, "body")
    if _contains_reasoning_rejection(error_body):
        return True

    error_text_parts = [str(error)]
    response = wire_field(error, "response")
    if response is not None:
        try:
            response_text = wire_field(response, "text")
        except Exception:
            response_text = None
        if isinstance(response_text, str) and response_text:
            if _contains_reasoning_rejection(_json_payload(response_text)):
                return True
            error_text_parts.append(response_text)

    if error_body is not None:
        error_text_parts.append(json.dumps(error_body, default=str))

    return _reasoning_rejection_text(" ".join(error_text_parts))


def normalize_mistral_stream(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield OpenAI-chat chunks with Mistral native thinking chunks normalized."""

    async def _iter() -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for chunk in stream:
                yield normalize_mistral_chunk(chunk)
        finally:
            await close_provider_stream(
                stream,
                active_error=sys.exception(),
                provider_name="MISTRAL",
                request_id=None,
            )

    return _iter()


def normalize_mistral_chunk(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
    """Normalize one Mistral OpenAI-compatible stream chunk."""
    choices = chunk.choices
    if not choices:
        return chunk

    normalized_choices: list[Choice] = []
    changed = False
    for choice in choices:
        delta: ChoiceDelta | None = getattr(choice, "delta", None)
        normalized_delta = _normalize_mistral_delta(delta)
        if normalized_delta is delta:
            normalized_choices.append(choice)
            continue
        changed = True
        normalized_choices.append(choice.model_copy(update={"delta": normalized_delta}))

    if not changed:
        return chunk
    return chunk.model_copy(update={"choices": normalized_choices})


def _normalize_mistral_delta(delta: ChoiceDelta | None) -> ChoiceDelta | None:
    if delta is None:
        return delta

    content = wire_field(delta, "content")
    text, reasoning, changed = _split_mistral_content(content)
    native_reasoning = _extract_native_reasoning(delta)
    if native_reasoning:
        reasoning = "\n".join(part for part in (reasoning, native_reasoning) if part)
        if isinstance(content, str):
            text = content
        changed = True

    existing_reasoning = wire_field(delta, "reasoning_content")
    if isinstance(existing_reasoning, str) and existing_reasoning:
        reasoning = "\n".join(part for part in (existing_reasoning, reasoning) if part)

    if not changed:
        return delta

    return delta.model_copy(
        update={"content": text, "reasoning_content": reasoning or None}
    )


def _content_with_prepended_thinking(
    content: object, reasoning: str
) -> list[JsonObject]:
    chunks = [_thinking_chunk(reasoning)]
    chunks.extend(_content_to_mistral_text_chunks(content))
    return chunks


def _content_to_mistral_text_chunks(content: object) -> list[JsonObject]:
    if isinstance(content, str):
        return [_text_chunk(content)] if content else []
    if not _is_sequence(content):
        text = _clean_text(content)
        return [_text_chunk(text)] if text else []

    chunks: list[JsonObject] = []
    for chunk in content:
        chunk_type = _chunk_type(chunk)
        if _is_thinking_chunk_type(chunk_type):
            continue
        text = _chunk_text(chunk)
        if text:
            chunks.append(_text_chunk(text))
    return chunks


def _split_mistral_content(content: object) -> tuple[str | None, str | None, bool]:
    if not _is_sequence(content):
        return None, None, False

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for chunk in content:
        chunk_type = _chunk_type(chunk)
        if _is_thinking_chunk_type(chunk_type):
            reasoning = _chunk_reasoning(chunk)
            if reasoning:
                reasoning_parts.append(reasoning)
            continue
        text = _chunk_text(chunk)
        if text:
            text_parts.append(text)

    return (
        "".join(text_parts) or None,
        "".join(reasoning_parts) or None,
        True,
    )


def _extract_native_reasoning(delta: object) -> str | None:
    for name in ("thinking", "reasoning"):
        value = wire_field(delta, name)
        text = _chunk_reasoning(value)
        if text:
            return text
    return None


def _contains_reasoning_rejection(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in _REASONING_FIELD_NAMES:
                return True
            if key_text in {"loc", "param"} and _is_reasoning_field_path(item):
                return True
            if _contains_reasoning_rejection(item):
                return True
        return False
    if _is_sequence(value):
        if _is_reasoning_field_path(value):
            return True
        return any(_contains_reasoning_rejection(item) for item in value)
    if isinstance(value, str):
        return _reasoning_rejection_text(value)
    return False


def _is_reasoning_field_path(value: object) -> bool:
    if not _is_sequence(value):
        return False
    parts = [str(part).lower() for part in value]
    if any(part in _REASONING_FIELD_NAMES for part in parts):
        return True
    return "thinking" in parts and bool(
        {"body", "messages", "assistant", "content"} & set(parts)
    )


def _reasoning_rejection_text(value: str) -> bool:
    lowered = value.lower()
    if "reasoning input" in lowered and any(
        phrase in lowered
        for phrase in (
            "not enabled",
            "not supported",
            "unsupported",
            "disabled",
        )
    ):
        return True
    if any(field in lowered for field in _REASONING_FIELD_NAMES):
        return True
    if "thinking" not in lowered or not any(
        word in lowered for word in _REJECTION_WORDS
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "body.messages",
            "assistant.thinking",
            "messages.assistant",
            "content.thinking",
            "field: thinking",
            "field 'thinking'",
            'field "thinking"',
            "property: thinking",
            "property 'thinking'",
            'property "thinking"',
            "parameter: thinking",
            "parameter 'thinking'",
            'parameter "thinking"',
            "param: thinking",
            "param 'thinking'",
            'param "thinking"',
        )
    )


def _json_payload(value: str) -> JsonValue:
    try:
        return load_json_value(value)
    except json.JSONDecodeError:
        return None


def _thinking_chunk(reasoning: str) -> JsonObject:
    return {
        "type": "thinking",
        "thinking": [_text_chunk(reasoning)],
    }


def _text_chunk(text: str) -> JsonObject:
    return {"type": "text", "text": text}


def _chunk_reasoning(chunk: object) -> str | None:
    if isinstance(chunk, str):
        return chunk or None
    thinking = wire_field(chunk, "thinking")
    if thinking is not None:
        text = _chunk_text(thinking)
        if text:
            return text
    text = _chunk_text(chunk)
    return text or None


def _chunk_text(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk
    if _is_sequence(chunk):
        return "".join(_chunk_text(part) for part in chunk)
    text = wire_field(chunk, "text")
    if isinstance(text, str):
        return text
    content = wire_field(chunk, "content")
    if isinstance(content, str):
        return content
    return ""


def _chunk_type(chunk: object) -> str | None:
    value = wire_field(chunk, "type")
    if isinstance(value, str):
        return value
    return None


def _is_thinking_chunk_type(chunk_type: str | None) -> bool:
    return chunk_type in {"thinking", "reasoning"}


def _clean_text(value: object) -> str | None:
    if isinstance(value, str):
        return value if value else None
    return None


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
