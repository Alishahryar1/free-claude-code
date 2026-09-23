"""Normalize native MiniMax-M3 tool markup exposed by NVIDIA NIM."""

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any

from free_claude_code.core.anthropic.tool_schema import (
    arguments_match_schema,
    coerce_text_argument,
    schema_type,
)
from free_claude_code.providers.failure_policy import RetryableToolProtocolError
from free_claude_code.providers.http import maybe_await_aclose

_NAMESPACE = "]<]minimax[>["
_TOOL_BLOCK_START = f"{_NAMESPACE}<tool_call>"
_TOOL_BLOCK_END = f"{_NAMESPACE}</tool_call>"
_INVOKE_START = f"{_NAMESPACE}<invoke"
_INVOKE_END = f"{_NAMESPACE}</invoke>"
_ELEMENT_START = f"{_NAMESPACE}<"
_ELEMENT_END_START = f"{_NAMESPACE}</"
_MIXED_TEXT_FIELD = "$text"
_MAX_NATIVE_TOOL_BLOCK_CHARS = 4 * 1024 * 1024
_MAX_NATIVE_ARGUMENT_DEPTH = 64


class NimNativeToolProtocolError(RetryableToolProtocolError):
    """NIM exposed malformed model-native tool markup instead of tool deltas."""


class _NormalizedNativeToolStream(AsyncIterator[Any]):
    """Expose normalized chunks while retaining the raw stream's close contract."""

    def __init__(self, iterator: AsyncIterator[Any], source: Any) -> None:
        self._iterator = iterator
        self._source = source
        self._closed = False

    def __aiter__(self) -> _NormalizedNativeToolStream:
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        return await anext(self._iterator)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await maybe_await_aclose(self._iterator)
        await maybe_await_aclose(self._source)


@dataclass(frozen=True, slots=True)
class _Element:
    name: str
    text: str
    children: tuple[_Element, ...]


@dataclass(frozen=True, slots=True)
class _NativeToolCall:
    index: int
    name: str
    arguments: dict[str, Any]


class _FramingMode(StrEnum):
    TEXT = "text"
    TOOL_BLOCK = "tool_block"
    AFTER_TOOL_BLOCK = "after_tool_block"
    FINISHED = "finished"


class _MiniMaxM3ToolFramer:
    """Separate visible text from one namespace-delimited native tool block."""

    def __init__(self) -> None:
        self._mode = _FramingMode.TEXT
        self._text_tail = ""
        self._tool_tail = ""
        self._tool_parts: list[str] = []
        self._tool_length = 0
        self._tool_block: str | None = None

    @property
    def tool_block(self) -> str | None:
        return self._tool_block

    def feed(self, text: str) -> str:
        if self._mode is _FramingMode.FINISHED:
            raise RuntimeError("MiniMax-M3 tool framer is already finished")
        if not text:
            return ""
        if self._mode is _FramingMode.TOOL_BLOCK:
            self._feed_tool_block(text)
            return ""
        if self._mode is _FramingMode.AFTER_TOOL_BLOCK:
            self._validate_trailing_text(text)
            return ""

        candidate = "".join((self._text_tail, text))
        marker_index = candidate.find(_TOOL_BLOCK_START)
        held_length = _partial_marker_suffix_length(candidate, _TOOL_BLOCK_START)
        protected_namespace_index = (
            marker_index
            if marker_index >= 0
            else len(candidate) - held_length
            if held_length
            else -1
        )
        namespace_index = candidate.find(_NAMESPACE)
        if namespace_index >= 0 and namespace_index != protected_namespace_index:
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned malformed native MiniMax tool markup."
            )
        if marker_index >= 0:
            visible = candidate[:marker_index]
            self._text_tail = ""
            self._mode = _FramingMode.TOOL_BLOCK
            self._feed_tool_block(candidate[marker_index + len(_TOOL_BLOCK_START) :])
            return visible

        if held_length:
            visible = candidate[:-held_length]
            self._text_tail = candidate[-held_length:]
            return visible

        self._text_tail = ""
        return candidate

    def finish(self, *, require_complete: bool) -> str:
        if self._mode is _FramingMode.FINISHED:
            return ""
        if self._mode is _FramingMode.TEXT:
            tail = self._text_tail
            self._text_tail = ""
            self._mode = _FramingMode.FINISHED
            if _NAMESPACE in tail:
                if require_complete:
                    raise NimNativeToolProtocolError(
                        "NVIDIA NIM returned an incomplete native MiniMax tool marker."
                    )
                return ""
            return tail
        if self._mode is _FramingMode.TOOL_BLOCK and require_complete:
            self._mode = _FramingMode.FINISHED
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned an incomplete native MiniMax tool call."
            )
        self._tool_parts.clear()
        self._tool_tail = ""
        self._mode = _FramingMode.FINISHED
        return ""

    def _feed_tool_block(self, text: str) -> None:
        candidate = "".join((self._tool_tail, text))
        marker_index = candidate.find(_TOOL_BLOCK_END)
        if marker_index < 0:
            held_length = _partial_marker_suffix_length(candidate, _TOOL_BLOCK_END)
            if held_length:
                self._append_tool_fragment(candidate[:-held_length])
                self._tool_tail = candidate[-held_length:]
            else:
                self._append_tool_fragment(candidate)
                self._tool_tail = ""
            return

        self._append_tool_fragment(candidate[:marker_index])
        self._tool_block = "".join(self._tool_parts)
        self._tool_parts.clear()
        self._tool_tail = ""
        self._mode = _FramingMode.AFTER_TOOL_BLOCK
        self._validate_trailing_text(candidate[marker_index + len(_TOOL_BLOCK_END) :])

    def _append_tool_fragment(self, fragment: str) -> None:
        if not fragment:
            return
        self._tool_length += len(fragment)
        if self._tool_length > _MAX_NATIVE_TOOL_BLOCK_CHARS:
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned an oversized native MiniMax tool call."
            )
        self._tool_parts.append(fragment)

    @staticmethod
    def _validate_trailing_text(text: str) -> None:
        if text.strip():
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned content after a native MiniMax tool call."
            )


def normalize_nim_native_tool_stream(
    stream: Any,
    body: Mapping[str, Any],
) -> AsyncIterator[Any]:
    """Convert leaked MiniMax-M3 markup into ordinary OpenAI tool-call chunks."""
    schemas = _openai_tool_schemas(body)

    async def _iter() -> AsyncIterator[Any]:
        content_framer = _MiniMaxM3ToolFramer()
        reasoning_framer = _MiniMaxM3ToolFramer()
        structured_tool_calls_seen = False
        finalized = False

        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                yield chunk
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                yield chunk
                continue

            native_tool_calls = getattr(delta, "tool_calls", None)
            if _has_structured_tool_calls(native_tool_calls):
                structured_tool_calls_seen = True

            content = getattr(delta, "content", None)
            safe_content = (
                content_framer.feed(content) if isinstance(content, str) else content
            )
            reasoning = getattr(delta, "reasoning_content", None)
            safe_reasoning = (
                reasoning_framer.feed(reasoning)
                if isinstance(reasoning, str)
                else reasoning
            )
            finish_reason = getattr(choice, "finish_reason", None)
            generated_calls: tuple[_NativeToolCall, ...] = ()
            if finish_reason is not None:
                safe_content, safe_reasoning, generated_calls = (
                    _finish_native_tool_framers(
                        content_framer,
                        reasoning_framer,
                        content=safe_content,
                        reasoning_content=safe_reasoning,
                        schemas=schemas,
                        structured_tool_calls_seen=structured_tool_calls_seen,
                    )
                )
                finalized = True

            normalized = _normalized_chunks(
                chunk,
                choice,
                delta,
                content=safe_content,
                reasoning_content=safe_reasoning,
                generated_calls=generated_calls,
                terminal=finish_reason is not None,
            )
            for normalized_chunk in normalized:
                yield normalized_chunk

        if not finalized:
            content, reasoning, generated_calls = _finish_native_tool_framers(
                content_framer,
                reasoning_framer,
                content=None,
                reasoning_content=None,
                schemas=schemas,
                structured_tool_calls_seen=structured_tool_calls_seen,
            )
            if content or reasoning or generated_calls:
                for normalized_chunk in _synthetic_chunks(
                    content=content,
                    reasoning_content=reasoning,
                    generated_calls=generated_calls,
                ):
                    yield normalized_chunk

    return _NormalizedNativeToolStream(_iter(), stream)


def _normalized_chunks(
    source_chunk: Any,
    source_choice: Any,
    source_delta: Any,
    *,
    content: Any,
    reasoning_content: Any,
    generated_calls: tuple[_NativeToolCall, ...],
    terminal: bool,
) -> list[Any]:
    source_content = getattr(source_delta, "content", None)
    source_reasoning = getattr(source_delta, "reasoning_content", None)
    structured_calls = getattr(source_delta, "tool_calls", None)
    source_finish_reason = getattr(source_choice, "finish_reason", None)
    source_usage = getattr(source_chunk, "usage", None)

    if (
        not generated_calls
        and content == source_content
        and reasoning_content == source_reasoning
    ):
        return [source_chunk]

    chunks: list[Any] = []
    has_source_payload = (
        bool(content)
        or bool(reasoning_content)
        or _has_structured_tool_calls(structured_calls)
    )
    if has_source_payload or not generated_calls:
        chunks.append(
            _chunk(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=structured_calls,
            )
        )

    chunks.extend(_tool_call_chunk(call) for call in generated_calls)

    if not chunks:
        chunks.append(_chunk())

    if terminal:
        chunks[-1].choices[0].finish_reason = source_finish_reason
        chunks[-1].usage = source_usage
    return chunks


def _synthetic_chunks(
    *,
    content: str | None,
    reasoning_content: str | None,
    generated_calls: tuple[_NativeToolCall, ...],
) -> list[Any]:
    chunks = (
        [_chunk(content=content, reasoning_content=reasoning_content)]
        if content or reasoning_content
        else []
    )
    chunks.extend(_tool_call_chunk(call) for call in generated_calls)
    return chunks


def _finish_native_tool_framers(
    content_framer: _MiniMaxM3ToolFramer,
    reasoning_framer: _MiniMaxM3ToolFramer,
    *,
    content: Any,
    reasoning_content: Any,
    schemas: Mapping[str, dict[str, Any]],
    structured_tool_calls_seen: bool,
) -> tuple[Any, Any, tuple[_NativeToolCall, ...]]:
    require_complete = not structured_tool_calls_seen
    content = _join_optional_text(
        content,
        content_framer.finish(require_complete=require_complete),
    )
    reasoning_content = _join_optional_text(
        reasoning_content,
        reasoning_framer.finish(require_complete=require_complete),
    )
    if structured_tool_calls_seen:
        return content, reasoning_content, ()

    blocks = tuple(
        block
        for block in (content_framer.tool_block, reasoning_framer.tool_block)
        if block is not None
    )
    if len(blocks) > 1:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned native MiniMax tool calls in multiple output fields."
        )
    generated_calls = _parse_tool_block(blocks[0], schemas) if blocks else ()
    return content, reasoning_content, generated_calls


def _chunk(
    *,
    content: Any = None,
    reasoning_content: Any = None,
    tool_calls: Any = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _tool_call_chunk(call: _NativeToolCall) -> Any:
    tool_call = SimpleNamespace(
        index=call.index,
        id=f"call_nim_native_{uuid.uuid4().hex}",
        function=SimpleNamespace(
            name=call.name,
            arguments=json.dumps(
                call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )
    return _chunk(tool_calls=[tool_call])


def _parse_tool_block(
    block: str,
    schemas: Mapping[str, dict[str, Any]],
) -> tuple[_NativeToolCall, ...]:
    cursor = 0
    calls: list[_NativeToolCall] = []
    while True:
        cursor = _skip_whitespace(block, cursor)
        if cursor == len(block):
            break
        name, body, cursor = _parse_invoke(block, cursor)
        schema = schemas.get(name)
        if schema is None:
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned a native call for an undeclared tool."
            )
        arguments = _parse_arguments(body, schema)
        _validate_arguments(name, arguments, schema)
        calls.append(
            _NativeToolCall(
                index=len(calls),
                name=name,
                arguments=arguments,
            )
        )

    if not calls:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an empty native MiniMax tool-call block."
        )
    return tuple(calls)


def _parse_invoke(text: str, cursor: int) -> tuple[str, str, int]:
    if not text.startswith(_INVOKE_START, cursor):
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned malformed native MiniMax tool markup."
        )
    tag_end = text.find(">", cursor + len(_INVOKE_START))
    if tag_end < 0:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an incomplete native MiniMax invoke tag."
        )
    attributes = text[cursor + len(_INVOKE_START) : tag_end]
    name = _parse_invoke_name(attributes)
    body_start = tag_end + 1
    body_end = text.find(_INVOKE_END, body_start)
    if body_end < 0:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an incomplete native MiniMax invoke block."
        )
    return name, text[body_start:body_end], body_end + len(_INVOKE_END)


def _parse_invoke_name(attributes: str) -> str:
    value = attributes.strip()
    if not value.startswith("name"):
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned a native MiniMax invoke without a tool name."
        )
    remainder = value[len("name") :].lstrip()
    if not remainder.startswith("="):
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned a malformed native MiniMax tool name."
        )
    remainder = remainder[1:].strip()
    if not remainder:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an empty native MiniMax tool name."
        )

    quote = remainder[0]
    if quote in {'"', "'"}:
        if len(remainder) < 2 or not remainder.endswith(quote):
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned an unterminated native MiniMax tool name."
            )
        name = remainder[1:-1]
    else:
        if any(character.isspace() for character in remainder):
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned malformed native MiniMax invoke attributes."
            )
        name = remainder

    name = name.strip()
    if not name:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an empty native MiniMax tool name."
        )
    return name


def _parse_arguments(body: str, schema: dict[str, Any]) -> dict[str, Any]:
    cursor = 0
    elements: list[_Element] = []
    while True:
        cursor = _skip_whitespace(body, cursor)
        if cursor == len(body):
            break
        element, cursor = _parse_element(body, cursor, depth=1)
        elements.append(element)
    return _elements_to_object(elements, schema)


def _parse_element(text: str, cursor: int, *, depth: int) -> tuple[_Element, int]:
    if depth > _MAX_NATIVE_ARGUMENT_DEPTH:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned excessively nested native MiniMax tool arguments."
        )
    if not text.startswith(_ELEMENT_START, cursor) or text.startswith(
        _ELEMENT_END_START, cursor
    ):
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned malformed native MiniMax tool arguments."
        )
    name_start = cursor + len(_ELEMENT_START)
    tag_end = text.find(">", name_start)
    if tag_end < 0:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an incomplete native MiniMax argument tag."
        )
    name = text[name_start:tag_end]
    if (
        not name
        or name.strip() != name
        or any(character.isspace() for character in name)
    ):
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned an invalid native MiniMax argument name."
        )

    close_tag = f"{_ELEMENT_END_START}{name}>"
    cursor = tag_end + 1
    text_parts: list[str] = []
    children: list[_Element] = []
    while True:
        marker = text.find(_NAMESPACE, cursor)
        if marker < 0:
            raise NimNativeToolProtocolError(
                "NVIDIA NIM returned an incomplete native MiniMax argument."
            )
        text_parts.append(text[cursor:marker])
        if text.startswith(close_tag, marker):
            cursor = marker + len(close_tag)
            break
        if text.startswith(_ELEMENT_START, marker) and not text.startswith(
            _ELEMENT_END_START, marker
        ):
            child, cursor = _parse_element(text, marker, depth=depth + 1)
            children.append(child)
            continue
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned mismatched native MiniMax argument tags."
        )

    return (
        _Element(
            name=name,
            text="".join(text_parts),
            children=tuple(children),
        ),
        cursor,
    )


def _elements_to_object(
    elements: Sequence[_Element],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    properties = schema.get("properties")
    property_schemas = properties if isinstance(properties, Mapping) else {}
    additional = schema.get("additionalProperties")
    output: dict[str, Any] = {}
    for element in elements:
        property_schema = property_schemas.get(element.name)
        if not isinstance(property_schema, Mapping):
            property_schema = additional if isinstance(additional, Mapping) else {}
        value = _element_value(element, property_schema)
        _append_object_value(output, element.name, value)
    return output


def _element_value(element: _Element, schema: Mapping[str, Any]) -> Any:
    declared_type = schema_type(schema)
    if element.children:
        if declared_type == "array":
            item_schema = schema.get("items")
            if not isinstance(item_schema, Mapping):
                item_schema = {}
            return [_element_value(child, item_schema) for child in element.children]

        result = _elements_to_object(element.children, schema)
        if element.text.strip():
            mixed_name = _MIXED_TEXT_FIELD
            while mixed_name in result:
                mixed_name = f"${mixed_name}"
            result[mixed_name] = element.text
        return result

    try:
        return coerce_text_argument(element.text, schema)
    except ValueError:
        raise NimNativeToolProtocolError(
            "NVIDIA NIM returned a native MiniMax argument with the wrong type."
        ) from None


def _validate_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    if not arguments_match_schema(arguments, schema):
        raise NimNativeToolProtocolError(
            f"NVIDIA NIM returned schema-invalid arguments for tool {tool_name!r}."
        )


def _openai_tool_schemas(body: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return {}

    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        schemas[name] = dict(parameters) if isinstance(parameters, Mapping) else {}
    return schemas


def _append_object_value(output: dict[str, Any], name: str, value: Any) -> None:
    if name not in output:
        output[name] = value
        return
    existing = output[name]
    if isinstance(existing, list):
        existing.append(value)
    else:
        output[name] = [existing, value]


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _partial_marker_suffix_length(text: str, marker: str) -> int:
    max_length = min(len(text), len(marker) - 1)
    for length in range(max_length, 0, -1):
        if marker.startswith(text[-length:]):
            return length
    return 0


def _join_optional_text(first: Any, second: str) -> Any:
    if not second:
        return first
    if isinstance(first, str):
        return "".join((first, second))
    return second


def _has_structured_tool_calls(value: Any) -> bool:
    return isinstance(value, list | tuple) and bool(value)
