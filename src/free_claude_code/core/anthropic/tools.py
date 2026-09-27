"""Heuristic parser for text-emitted tool calls."""

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..openai_tool_names import OpenAIToolNameCodec
from .models import MessagesRequest
from .tool_schema import prepare_text_tool_input

_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{1,80}\|>")
_CONTROL_TOKEN_START = "<|"
_CONTROL_TOKEN_END = "|>"
_FUNCTION_TAG_BLOCK_START = "<tool_call>"
_FUNCTION_TAG_BLOCK_END = "</tool_call>"
_FUNCTION_TAG_START = "<function="
_FUNCTION_TAG_END = "</function>"
_PARAMETER_TAG_START = "<parameter="
_PARAMETER_TAG_END = "</parameter>"
_MAX_FUNCTION_TAG_CANDIDATE_CHARS = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _RawFunctionTagCall:
    name: str
    arguments: dict[str, str]


class _FunctionTagState(Enum):
    SEARCHING = 1
    CANDIDATE = 2
    DISABLED = 3
    FINISHED = 4


class FunctionTagToolParser:
    """Parse an exact terminal function-tag envelope into tool use."""

    def __init__(self, request: MessagesRequest):
        schemas: dict[str, dict[str, Any]] = {}
        for tool in request.tools or ():
            if tool.name:
                schemas[tool.name] = (
                    tool.input_schema
                    if tool.input_schema is not None
                    else {"type": "object"}
                )
        tool_choice = request.tool_choice
        tool_choice_type = (
            tool_choice.get("type") if isinstance(tool_choice, dict) else None
        )
        self._initialize(
            tool_names=OpenAIToolNameCodec.from_request(request),
            schemas=schemas,
            enabled=tool_choice_type != "none",
        )

    @classmethod
    def from_schemas(
        cls,
        *,
        tool_names: OpenAIToolNameCodec,
        schemas: Mapping[str, Mapping[str, Any]],
        enabled: bool,
    ) -> FunctionTagToolParser:
        """Build the parser from a protocol-neutral Chat tool contract."""
        parser = cls.__new__(cls)
        parser._initialize(
            tool_names=tool_names,
            schemas={name: dict(schema) for name, schema in schemas.items()},
            enabled=enabled,
        )
        return parser

    def _initialize(
        self,
        *,
        tool_names: OpenAIToolNameCodec,
        schemas: dict[str, dict[str, Any]],
        enabled: bool,
    ) -> None:
        self._tool_names = tool_names
        self._schemas = schemas
        self._state = (
            _FunctionTagState.SEARCHING
            if self._schemas and enabled
            else _FunctionTagState.DISABLED
        )
        self._parts: list[str] = []
        self._length = 0
        self._marker_tail = ""

    def feed(self, text: str) -> str:
        """Hold a possible reserved response and return text safe to expose."""
        if not text:
            return ""
        if self._state in {_FunctionTagState.DISABLED, _FunctionTagState.FINISHED}:
            return text

        if self._state is _FunctionTagState.CANDIDATE:
            self._parts.append(text)
            self._length += len(text)
            if self._length > _MAX_FUNCTION_TAG_CANDIDATE_CHARS:
                return self.disable()
            return ""

        candidate = "".join((self._marker_tail, text))
        marker_index = candidate.find(_FUNCTION_TAG_BLOCK_START)
        if marker_index >= 0:
            visible = candidate[:marker_index]
            control = candidate[marker_index:]
            self._state = _FunctionTagState.CANDIDATE
            self._marker_tail = ""
            self._parts.append(control)
            self._length = len(control)
            if self._length > _MAX_FUNCTION_TAG_CANDIDATE_CHARS:
                return "".join((visible, self.disable()))
            return visible

        held_length = _partial_function_tag_marker_suffix_length(candidate)
        if held_length:
            self._marker_tail = candidate[-held_length:]
            return candidate[:-held_length]
        self._marker_tail = ""
        return candidate

    def disable(self) -> str:
        """Disable textual recovery and release any held candidate unchanged."""
        if self._state in {_FunctionTagState.DISABLED, _FunctionTagState.FINISHED}:
            return ""
        self._state = _FunctionTagState.DISABLED
        text = "".join((self._marker_tail, *self._parts))
        self._marker_tail = ""
        self._parts.clear()
        self._length = 0
        return text

    def finish(self) -> tuple[str, tuple[dict[str, Any], ...]]:
        """Finalize one response atomically as visible text or validated tools."""
        if self._state in {_FunctionTagState.DISABLED, _FunctionTagState.FINISHED}:
            return "", ()
        if self._state is _FunctionTagState.SEARCHING:
            self._state = _FunctionTagState.FINISHED
            text = self._marker_tail
            self._marker_tail = ""
            return text, ()
        self._state = _FunctionTagState.FINISHED
        raw = "".join(self._parts)
        self._parts.clear()
        self._length = 0
        try:
            calls = _parse_function_tag_calls(raw)
            tool_uses = self._validated_tool_uses(calls)
        except ValueError:
            return raw, ()
        return "", tool_uses

    def _validated_tool_uses(
        self, calls: tuple[_RawFunctionTagCall, ...]
    ) -> tuple[dict[str, Any], ...]:
        tool_uses: list[dict[str, Any]] = []
        for call in calls:
            name = self._tool_names.decode(call.name)
            schema = self._schemas.get(name)
            if schema is None:
                raise ValueError
            arguments = prepare_text_tool_input(
                call.arguments, schema, text_parameters=True
            )
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_function_tag_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "input": arguments,
                }
            )
        return tuple(tool_uses)


def _parse_function_tag_calls(text: str) -> tuple[_RawFunctionTagCall, ...]:
    cursor = 0
    calls: list[_RawFunctionTagCall] = []
    while True:
        cursor = _skip_function_tag_whitespace(text, cursor)
        if cursor == len(text):
            break
        if not text.startswith(_FUNCTION_TAG_BLOCK_START, cursor):
            raise ValueError

        block_start = cursor + len(_FUNCTION_TAG_BLOCK_START)
        block_end = text.find(_FUNCTION_TAG_BLOCK_END, block_start)
        if block_end < 0:
            raise ValueError
        name, arguments = _parse_function_tag_block(text[block_start:block_end])
        calls.append(_RawFunctionTagCall(name=name, arguments=arguments))
        cursor = block_end + len(_FUNCTION_TAG_BLOCK_END)

    if not calls:
        raise ValueError
    return tuple(calls)


def _parse_function_tag_block(block: str) -> tuple[str, dict[str, str]]:
    cursor = _skip_function_tag_whitespace(block, 0)
    if not block.startswith(_FUNCTION_TAG_START, cursor):
        raise ValueError
    name_end = block.find(">", cursor + len(_FUNCTION_TAG_START))
    if name_end < 0:
        raise ValueError
    name = block[cursor + len(_FUNCTION_TAG_START) : name_end]
    if not _valid_function_tag_name(name):
        raise ValueError

    arguments: dict[str, str] = {}
    cursor = name_end + 1
    while True:
        cursor = _skip_function_tag_whitespace(block, cursor)
        if block.startswith(_FUNCTION_TAG_END, cursor):
            cursor += len(_FUNCTION_TAG_END)
            break
        if cursor == len(block) or not block.startswith(_PARAMETER_TAG_START, cursor):
            raise ValueError

        parameter_name_end = block.find(">", cursor + len(_PARAMETER_TAG_START))
        if parameter_name_end < 0:
            raise ValueError
        parameter_name = block[cursor + len(_PARAMETER_TAG_START) : parameter_name_end]
        if not _valid_function_tag_name(parameter_name) or parameter_name in arguments:
            raise ValueError

        value_start = parameter_name_end + 1
        value_end = block.find(_PARAMETER_TAG_END, value_start)
        if value_end < 0:
            raise ValueError
        arguments[parameter_name] = _unwrap_function_tag_newlines(
            block[value_start:value_end]
        )
        cursor = value_end + len(_PARAMETER_TAG_END)

    if block[cursor:].strip():
        raise ValueError
    return name, arguments


def _skip_function_tag_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _valid_function_tag_name(value: str) -> bool:
    return bool(value) and not any(
        character.isspace() or character in "<>" for character in value
    )


def _unwrap_function_tag_newlines(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def _partial_function_tag_marker_suffix_length(text: str) -> int:
    max_length = min(len(text), len(_FUNCTION_TAG_BLOCK_START) - 1)
    for length in range(max_length, 0, -1):
        if _FUNCTION_TAG_BLOCK_START.startswith(text[-length:]):
            return length
    return 0


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


class HeuristicToolParser:
    """Interpret declared legacy text tools, retaining rejected source as text."""

    _FUNC_START_PATTERN = re.compile(r"●\s*<function=([^>]+)>")
    _PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL
    )
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )

    def __init__(
        self,
        *,
        tool_names: OpenAIToolNameCodec,
        schemas: Mapping[str, Mapping[str, Any]],
        enabled: bool = True,
    ):
        self._tool_names = tool_names
        self._schemas = schemas
        self._enabled = enabled and bool(schemas)
        self._state = ParserState.TEXT
        self._buffer = ""
        self._candidate_parts: list[str] = []
        self._candidate_length = 0
        self._interstitial: list[str] = []
        self._current_function_name = ""
        self._current_parameters: dict[str, str] = {}

    def _tool_use(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        text_parameters: bool,
    ) -> dict[str, Any] | None:
        name = self._tool_names.decode(name)
        schema = self._schemas.get(name)
        if schema is None or not self._enabled:
            return None
        try:
            prepared = prepare_text_tool_input(
                arguments, schema, text_parameters=text_parameters
            )
        except ValueError:
            return None
        return {
            "type": "tool_use",
            "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
            "name": name,
            "input": prepared,
        }

    def _text_parts(self, text: str) -> list[str | dict[str, Any]]:
        parts: list[str | dict[str, Any]] = []
        cursor = 0
        if self._enabled:
            for match in self._WEB_TOOL_JSON_PATTERN.finditer(text):
                try:
                    arguments = json.loads(match.group("json"))
                except ValueError:
                    continue
                name = match.group("tool")
                required = "url" if name == "WebFetch" else "query"
                if not isinstance(arguments, dict) or required not in arguments:
                    continue
                tool = self._tool_use(name, arguments, text_parameters=False)
                if tool is None:
                    continue
                if match.start() > cursor:
                    parts.append(text[cursor : match.start()])
                parts.append(tool)
                cursor = match.end()
        if cursor < len(text):
            parts.append(text[cursor:])
        return parts

    def _take_candidate(self, length: int) -> str:
        consumed, self._buffer = self._buffer[:length], self._buffer[length:]
        self._candidate_parts.append(consumed)
        self._candidate_length += len(consumed)
        return consumed

    def _finish_candidate(self) -> list[str | dict[str, Any]]:
        tool = self._tool_use(
            self._current_function_name, self._current_parameters, text_parameters=True
        )
        parts: list[str | dict[str, Any]] = (
            [*self._interstitial, tool]
            if tool is not None
            else ["".join(self._candidate_parts)]
        )
        self._candidate_parts.clear()
        self._candidate_length = 0
        self._interstitial.clear()
        self._current_parameters = {}
        self._state = ParserState.TEXT
        return parts

    def feed(self, text: str) -> list[str | dict[str, Any]]:
        """Return text and validated calls in their original stream order."""
        self._buffer = _CONTROL_TOKEN_RE.sub("", self._buffer + text)
        parts: list[str | dict[str, Any]] = []
        while self._buffer:
            if self._state is ParserState.TEXT:
                marker = self._buffer.find("●") if self._enabled else -1
                if marker >= 0:
                    parts.extend(self._text_parts(self._buffer[:marker]))
                    self._buffer = self._buffer[marker:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    # Retain a split control token until its closing marker arrives.
                    start = self._buffer.rfind(_CONTROL_TOKEN_START)
                    if start >= 0 and _CONTROL_TOKEN_END not in self._buffer[start:]:
                        parts.extend(self._text_parts(self._buffer[:start]))
                        self._buffer = self._buffer[start:]
                    else:
                        parts.extend(self._text_parts(self._buffer))
                        self._buffer = ""
                    break

            if (
                self._candidate_length + len(self._buffer)
                > _MAX_FUNCTION_TAG_CANDIDATE_CHARS
            ):
                parts.append("".join((*self._candidate_parts, self._buffer)))
                self._buffer = ""
                self._candidate_parts.clear()
                self._candidate_length = 0
                self._interstitial.clear()
                self._current_parameters.clear()
                self._enabled = False
                self._state = ParserState.TEXT
                break

            if self._state is ParserState.MATCHING_FUNCTION:
                match = self._FUNC_START_PATTERN.match(self._buffer)
                if match:
                    self._current_function_name = match.group(1).strip()
                    self._take_candidate(match.end())
                    self._state = ParserState.PARSING_PARAMETERS
                elif len(self._buffer) > 100:
                    parts.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                    self._state = ParserState.TEXT
                    continue
                else:
                    break

            if self._state is ParserState.PARSING_PARAMETERS:
                while match := self._PARAM_PATTERN.search(self._buffer):
                    next_candidate = self._buffer.find("●")
                    if (
                        not match.group(0).endswith("</parameter>")
                        or 0 <= next_candidate < match.start()
                    ):
                        break
                    if match.start():
                        self._interstitial.append(self._buffer[: match.start()])
                    self._current_parameters[match.group(1).strip()] = match.group(
                        2
                    ).strip()
                    self._take_candidate(match.end())
                if "●" in self._buffer:
                    # Do not consume the following candidate as part of this one.
                    parts.extend(self._finish_candidate())
                elif (
                    self._buffer.strip()
                    and not self._buffer.strip().startswith("<")
                    and "<parameter=" not in self._buffer
                ):
                    parts.extend(self._finish_candidate())
                else:
                    break
        return parts

    def flush(self) -> list[str | dict[str, Any]]:
        """Finish a legacy EOF parameter, or release an unrecognized candidate."""
        parts: list[str | dict[str, Any]] = []
        if self._state is ParserState.PARSING_PARAMETERS:
            for match in re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            ):
                self._current_parameters[match.group(1).strip()] = match.group(
                    2
                ).strip()
            self._take_candidate(len(self._buffer))
            parts.extend(self._finish_candidate())
        elif self._buffer:
            parts.extend(self._text_parts(self._buffer))
        self._buffer = ""
        self._state = ParserState.TEXT
        return parts
