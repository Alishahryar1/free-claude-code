"""Reversible tool names for OpenAI-compatible protocol boundaries."""

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .anthropic.content import get_block_attr, get_block_type
from .anthropic.models import MessagesRequest

OPENAI_TOOL_NAME_MAX_LENGTH = 64
_ALIAS_DIGEST_LENGTH = 16
_PORTABLE_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_INVALID_TOOL_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class OpenAIToolNameCodec:
    """Map non-portable client tool names to deterministic OpenAI aliases."""

    _original_to_alias: dict[str, str]
    _alias_to_original: dict[str, str]
    _unchanged_names: frozenset[str]

    @classmethod
    def from_request(cls, request: MessagesRequest) -> OpenAIToolNameCodec:
        """Build the codec from every tool identity carried by one request."""
        return cls.from_names(_request_tool_names(request))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> OpenAIToolNameCodec:
        """Build order-independent aliases for the supplied non-empty names."""
        unique_names = {name for name in names if name}
        portable_names = {
            name for name in unique_names if _PORTABLE_TOOL_NAME.fullmatch(name)
        }
        reserved = set(portable_names)
        original_to_alias: dict[str, str] = {}
        alias_to_original: dict[str, str] = {}

        for name in sorted(unique_names - portable_names):
            alias = _unique_alias(name, reserved)
            reserved.add(alias)
            original_to_alias[name] = alias
            alias_to_original[alias] = name

        return cls(
            original_to_alias,
            alias_to_original,
            frozenset(portable_names),
        )

    @property
    def has_aliases(self) -> bool:
        """Return whether this request needs any tool-name translation."""
        return bool(self._original_to_alias)

    def encode(self, name: str) -> str:
        """Return the OpenAI wire name for one client name."""
        return self._original_to_alias.get(name, name)

    def decode(self, name: str) -> str:
        """Return the original client name for one known wire alias."""
        return self._alias_to_original.get(name, name)

    def is_alias(self, value: str) -> bool:
        """Return whether value is one complete alias generated for this request."""
        return value in self._alias_to_original

    def is_unchanged_name(self, value: str) -> bool:
        """Return whether value is one complete name sent upstream unchanged."""
        return value in self._unchanged_names

    def is_alias_prefix(self, value: str) -> bool:
        """Return whether value is an incomplete prefix of a generated alias."""
        return bool(value) and any(
            alias != value and alias.startswith(value)
            for alias in self._alias_to_original
        )


def encode_openai_chat_tool_names(
    body: dict[str, object], codec: OpenAIToolNameCodec
) -> None:
    """Encode canonical Chat tool-name locations at the provider boundary."""
    if not codec.has_aliases:
        return

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                function["name"] = codec.encode(name)

    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if isinstance(name, str):
                    function["name"] = codec.encode(name)

    tool_choice = body.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return
    name = function.get("name")
    if not isinstance(name, str):
        return
    encoded = codec.encode(name)
    if encoded != name:
        body["tool_choice"] = {
            **tool_choice,
            "function": {**function, "name": encoded},
        }


def _request_tool_names(request: MessagesRequest) -> Iterable[str]:
    for tool in request.tools or ():
        yield tool.name

    tool_choice = request.tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            name = tool_choice.get("name")
            if isinstance(name, str):
                yield name
        elif choice_type == "function":
            function = tool_choice.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    yield name

    for message in request.messages:
        if message.role != "assistant" or not isinstance(message.content, list):
            continue
        for block in message.content:
            if get_block_type(block) != "tool_use":
                continue
            name = get_block_attr(block, "name")
            if isinstance(name, str):
                yield name


def _unique_alias(name: str, reserved: set[str]) -> str:
    readable = _INVALID_TOOL_NAME_CHARACTERS.sub("_", name).strip("_-") or "tool"
    max_readable_length = OPENAI_TOOL_NAME_MAX_LENGTH - _ALIAS_DIGEST_LENGTH - 1
    readable = readable[:max_readable_length].rstrip("_-") or "tool"
    attempt = 0
    while True:
        digest_input = name if attempt == 0 else f"{name}\0{attempt}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[
            :_ALIAS_DIGEST_LENGTH
        ]
        alias = f"{readable}_{digest}"
        if alias not in reserved:
            return alias
        attempt += 1
