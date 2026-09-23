"""Heuristic parser for text-emitted tool calls."""

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from ..openai_tool_names import OpenAIToolNameCodec
from .models import MessagesRequest
from .tool_schema import arguments_match_schema, coerce_text_argument

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
            properties = schema.get("properties")
            property_schemas = properties if isinstance(properties, Mapping) else {}
            additional = schema.get("additionalProperties")
            arguments: dict[str, Any] = {}
            for parameter_name, value in call.arguments.items():
                parameter_schema = property_schemas.get(parameter_name)
                if not isinstance(parameter_schema, Mapping):
                    parameter_schema = (
                        additional if isinstance(additional, Mapping) else {}
                    )
                arguments[parameter_name] = coerce_text_argument(
                    value,
                    parameter_schema,
                )
            if not arguments_match_schema(arguments, schema):
                raise ValueError
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
    """
    Stateful parser for raw text tool calls.

    Some OpenAI-compatible models emit tool calls as text rather than structured
    chunks. This parser converts the common ``● <function=...>`` form into
    Anthropic-style ``tool_use`` blocks.
    """

    _FUNC_START_PATTERN = re.compile(r"●\s*<function=([^>]+)>")
    _PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL
    )
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )

    def __init__(self):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id = None
        self._current_function_name = None
        self._current_parameters = {}

    def _extract_web_tool_json_calls(self) -> tuple[str, list[dict[str, Any]]]:
        detected_tools: list[dict[str, Any]] = []

        for match in self._WEB_TOOL_JSON_PATTERN.finditer(self._buffer):
            try:
                tool_input = json.loads(match.group("json"))
            except json.JSONDecodeError:
                continue
            if not isinstance(tool_input, dict):
                continue

            tool_name = match.group("tool")
            if tool_name == "WebFetch" and "url" not in tool_input:
                continue
            if tool_name == "WebSearch" and "query" not in tool_input:
                continue

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            logger.debug(
                "Heuristic bypass: Detected JSON-style tool call '{}'",
                tool_name,
            )

        if not detected_tools:
            return self._buffer, []

        return "", detected_tools

    def _strip_control_tokens(self, text: str) -> str:
        return _CONTROL_TOKEN_RE.sub("", text)

    def _split_incomplete_control_token_tail(self) -> str:
        start = self._buffer.rfind(_CONTROL_TOKEN_START)
        if start == -1:
            return ""
        end = self._buffer.find(_CONTROL_TOKEN_END, start)
        if end != -1:
            return ""

        prefix = self._buffer[:start]
        self._buffer = self._buffer[start:]
        return prefix

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed text and return safe text plus detected tool calls."""
        self._buffer += text
        self._buffer = self._strip_control_tokens(self._buffer)
        self._buffer, detected_tools = self._extract_web_tool_json_calls()
        filtered_output_parts: list[str] = []

        while True:
            if self._state == ParserState.TEXT:
                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    filtered_output_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    safe_prefix = self._split_incomplete_control_token_tail()
                    if safe_prefix:
                        filtered_output_parts.append(safe_prefix)
                        break

                    filtered_output_parts.append(self._buffer)
                    self._buffer = ""
                    break

            if self._state == ParserState.MATCHING_FUNCTION:
                match = self._FUNC_START_PATTERN.search(self._buffer)
                if match:
                    self._current_function_name = match.group(1).strip()
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}
                    self._buffer = self._buffer[match.end() :]
                    self._state = ParserState.PARSING_PARAMETERS
                    logger.debug(
                        "Heuristic bypass: Detected start of tool call '{}'",
                        self._current_function_name,
                    )
                elif len(self._buffer) > 100:
                    filtered_output_parts.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                    self._state = ParserState.TEXT
                else:
                    break

            if self._state == ParserState.PARSING_PARAMETERS:
                finished_tool_call = False

                while True:
                    param_match = self._PARAM_PATTERN.search(self._buffer)
                    if param_match and "</parameter>" in param_match.group(0):
                        pre_match_text = self._buffer[: param_match.start()]
                        if pre_match_text:
                            filtered_output_parts.append(pre_match_text)

                        key = param_match.group(1).strip()
                        val = param_match.group(2).strip()
                        self._current_parameters[key] = val
                        self._buffer = self._buffer[param_match.end() :]
                    else:
                        break

                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    if idx > 0:
                        filtered_output_parts.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    finished_tool_call = True
                elif len(self._buffer) > 0 and not self._buffer.strip().startswith("<"):
                    if "<parameter=" not in self._buffer:
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True

                if finished_tool_call:
                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": self._current_function_name,
                            "input": self._current_parameters,
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        self._current_function_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                else:
                    break

        return "".join(filtered_output_parts), detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """Flush any remaining tool call in the buffer."""
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        if self._state == ParserState.PARSING_PARAMETERS:
            partial_matches = re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            )
            for match in partial_matches:
                key = match.group(1).strip()
                val = match.group(2).strip()
                self._current_parameters[key] = val

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": self._current_tool_id,
                    "name": self._current_function_name,
                    "input": self._current_parameters,
                }
            )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
