"""Source-preserving edits for the small set of supported client files."""

import json
import tomllib
from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from datetime import datetime, time
from decimal import Decimal

import tomlkit
import tree_sitter_json
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import Table
from tree_sitter import Language, Node, Parser

from free_claude_code.application.integrations import IntegrationError

MISSING = object()
type PathKey = str | int
type SettingPath = tuple[PathKey, ...]


def _invalid() -> IntegrationError:
    return IntegrationError(
        "The configuration file is malformed or has unsupported syntax. Fix it before continuing."
    )


def _constant(_value: str) -> None:
    raise _invalid()


def _same_data(first: object, second: object) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _same_data(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _same_data(left, right) for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, Decimal) and isinstance(second, Decimal):
        if first.is_nan() or second.is_nan():
            return (
                first.is_nan()
                and second.is_nan()
                and first.is_signed() == second.is_signed()
            )
        return first == second and first.is_signed() == second.is_signed()
    if isinstance(first, datetime | time) and isinstance(second, datetime | time):
        return first.isoformat() == second.isoformat()
    return first == second


def _validation_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _validation_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_validation_value(child) for child in value]
    return Decimal(repr(value)) if isinstance(value, float) else value


def _expected_data(data: dict, path: SettingPath, value: object) -> dict:
    expected = deepcopy(data)
    current: object = expected
    for key in path[:-1]:
        if isinstance(current, dict) and isinstance(key, str):
            if key not in current:
                current[key] = {}
            current = current[key]
        elif (
            isinstance(current, list)
            and isinstance(key, int)
            and 0 <= key < len(current)
        ):
            current = current[key]
        else:
            raise IntegrationError(
                "A setting being changed has an incompatible parent."
            )
    key = path[-1]
    if isinstance(current, dict) and isinstance(key, str):
        if value is MISSING:
            current.pop(key, None)
        else:
            current[key] = _validation_value(value)
    elif (
        isinstance(current, list) and isinstance(key, int) and 0 <= key <= len(current)
    ):
        if value is not MISSING:
            if key == len(current):
                current.append(_validation_value(value))
            else:
                current[key] = _validation_value(value)
    else:
        raise IntegrationError("A setting being changed has an incompatible parent.")
    return expected


def _require_expected(actual: dict, expected: dict) -> None:
    if not _same_data(actual, expected):
        raise IntegrationError(
            "This configuration layout cannot be edited without changing other settings. Use manual setup."
        )


def _tokens(node: Node) -> Iterator[Node]:
    if node.is_missing:
        return
    if node.type in {"comment", "string"} or not node.children:
        yield node
    else:
        for child in node.children:
            yield from _tokens(child)


class JsonDocument:
    def __init__(self, source: bytes, *, jsonc: bool) -> None:
        self._bom = b"\xef\xbb\xbf" if source.startswith(b"\xef\xbb\xbf") else b""
        self._body = source[len(self._bom) :]
        self._jsonc = jsonc
        self._parser = Parser(Language(tree_sitter_json.language()))
        self._parse()

    def _parse(self) -> None:
        try:
            self._body.decode("utf-8")
            original = self._parser.parse(self._body)
            tokens = list(_tokens(original.root_node))
            significant = [node for node in tokens if node.type != "comment"]
            self._commas = [node.start_byte for node in significant if node.type == ","]
            projection = bytearray(self._body)
            if self._jsonc:
                for index, node in enumerate(significant):
                    if (
                        node.type == ","
                        and 0 < index < len(significant) - 1
                        and significant[index + 1].type in {"}", "]"}
                        and significant[index - 1].type not in {"{", "[", ":", ","}
                    ):
                        projection[node.start_byte : node.end_byte] = b" " * (
                            node.end_byte - node.start_byte
                        )
                for node in tokens:
                    if node.type == "comment":
                        for index in range(node.start_byte, node.end_byte):
                            if projection[index] not in (10, 13):
                                projection[index] = 32
            self._projection = bytes(projection)
            self.data = json.loads(self._projection, parse_constant=_constant)
            self._semantic = json.loads(
                self._projection, parse_float=Decimal, parse_constant=_constant
            )
            tree = self._parser.parse(self._projection)
            if not isinstance(self.data, dict) or tree.root_node.has_error:
                raise _invalid()
            self._root = tree.root_node.named_children[0]
        except UnicodeError, ValueError, IndexError:
            raise _invalid() from None

    def _child(self, parent: Node, key: PathKey) -> Node | None:
        if isinstance(key, str) and parent.type == "object":
            matches = []
            for pair in parent.named_children:
                key_node = pair.child_by_field_name("key")
                if (
                    key_node is not None
                    and json.loads(
                        self._projection[key_node.start_byte : key_node.end_byte]
                    )
                    == key
                ):
                    matches.append(pair.child_by_field_name("value"))
            if len(matches) > 1:
                raise IntegrationError(
                    "A setting being changed has duplicate keys. Remove the duplicate before continuing."
                )
            return matches[0] if matches else None
        if isinstance(key, int) and parent.type == "array" and key >= 0:
            children = parent.named_children
            return children[key] if key < len(children) else None
        raise IntegrationError(
            "A setting being changed has an incompatible object or array type."
        )

    def _find(self, path: SettingPath) -> Node | None:
        node = self._root
        for key in path:
            child = self._child(node, key)
            if child is None:
                return None
            node = child
        return node

    def get(self, path: SettingPath) -> object:
        node = self._find(path)
        if node is None:
            return MISSING
        return json.loads(self._projection[node.start_byte : node.end_byte])

    def render(self) -> bytes:
        return self._bom + self._body

    def _edit(self, edits: list[tuple[int, int, bytes]], expected: dict) -> None:
        body = self._body
        grouped: dict[tuple[int, int], bytes] = {}
        for start, end, replacement in edits:
            grouped[start, end] = grouped.get((start, end), b"") + replacement
        for (start, end), replacement in sorted(grouped.items(), reverse=True):
            body = body[:start] + replacement + body[end:]
        # Validate before replacing this document, including all untouched bytes.
        candidate = JsonDocument(self._bom + body, jsonc=self._jsonc)
        _require_expected(candidate._semantic, expected)
        self.__dict__.update(candidate.__dict__)

    def _insert(self, parent: Node, value: bytes, expected: dict) -> None:
        children = parent.named_children
        close = parent.end_byte - 1
        edits = []
        if children:
            last = children[-1]
            if not any(last.end_byte <= comma < close for comma in self._commas):
                edits.append((last.end_byte, last.end_byte, b","))
        newline = b"\r\n" if b"\r\n" in self._body else b"\n"
        line_start = self._body.rfind(b"\n", 0, close) + 1
        closing_indent = self._body[line_start:close]
        if line_start > parent.start_byte and not closing_indent.strip():
            indent = closing_indent + b"  "
            if children:
                first = children[0].start_byte
                first_line = self._body.rfind(b"\n", 0, first) + 1
                existing = self._body[first_line:first]
                if not existing.strip():
                    indent = existing
            edits.append((line_start, line_start, indent + value + newline))
        else:
            edits.append((close, close, (b" " if children else b"") + value))
        self._edit(edits, expected)

    def set(self, path: SettingPath, value: object) -> None:
        expected = _expected_data(self._semantic, path, value)
        node = self._root
        for index, key in enumerate(path):
            child = self._child(node, key)
            if child is None:
                nested = value
                for tail in reversed(path[index + 1 :]):
                    if not isinstance(tail, str):
                        raise IntegrationError(
                            "A missing array must be created before adding entries."
                        )
                    nested = {tail: nested}
                encoded = json.dumps(
                    nested, ensure_ascii=False, allow_nan=False
                ).encode()
                if isinstance(key, str):
                    encoded = (
                        json.dumps(key, ensure_ascii=False).encode() + b": " + encoded
                    )
                elif key != len(node.named_children):
                    raise IntegrationError(
                        "An array entry no longer exists.", status_code=409
                    )
                self._insert(node, encoded, expected)
                return
            node = child
        if _same_data(self._semantic, expected):
            return
        self._edit(
            [
                (
                    node.start_byte,
                    node.end_byte,
                    json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                )
            ],
            expected,
        )


class TomlDocument:
    def __init__(self, source: bytes) -> None:
        try:
            self._text = source.decode("utf-8")
            self.data = tomllib.loads(self._text)
            self._semantic = tomllib.loads(self._text, parse_float=Decimal)
            self._document = tomlkit.parse(self._text)
            if tomlkit.dumps(self._document) != self._text:
                raise IntegrationError(
                    "This TOML layout cannot be edited without reformatting other settings. Use manual setup."
                )
        except UnicodeError, ValueError, TOMLKitError:
            raise _invalid() from None

    def render(self) -> bytes:
        return self._text.encode()

    def get(self, path: SettingPath) -> object:
        current: object = self.data
        for key in path:
            if not isinstance(key, str) or not isinstance(current, dict):
                raise IntegrationError(
                    "A setting being changed has an incompatible table type."
                )
            if key not in current:
                return MISSING
            current = current[key]
        return current

    def _change(self, path: SettingPath, value: object) -> None:
        expected = _expected_data(self._semantic, path, value)
        if _same_data(self._semantic, expected):
            return
        try:
            candidate = tomlkit.parse(self._text)
            current: MutableMapping = candidate
            ancestors: list[tuple[MutableMapping, str]] = []
            for key in path[:-1]:
                if not isinstance(key, str):
                    raise IntegrationError("Unsupported TOML setting path.")
                if key not in current:
                    current[key] = tomlkit.inline_table()
                child = current[key]
                if not isinstance(child, MutableMapping):
                    raise IntegrationError(
                        "A setting being changed has an incompatible table type."
                    )
                ancestors.append((current, key))
                current = child
            key = path[-1]
            if not isinstance(key, str):
                raise IntegrationError("Unsupported TOML setting path.")
            if value is MISSING:
                current.pop(key, None)
                for parent, name in reversed(ancestors):
                    child = parent[name]
                    if (
                        isinstance(child, Table)
                        and child.is_super_table()
                        and not child
                    ):
                        parent[name] = tomlkit.inline_table()
            else:
                current[key] = value
            text = tomlkit.dumps(candidate)
            data = tomllib.loads(text)
            semantic = tomllib.loads(text, parse_float=Decimal)
            _require_expected(semantic, expected)
        except ValueError, TOMLKitError:
            raise _invalid() from None
        self._text, self._document, self.data, self._semantic = (
            text,
            candidate,
            data,
            semantic,
        )

    def set(self, path: SettingPath, value: object) -> None:
        self._change(path, value)

    def delete(self, path: SettingPath) -> None:
        if self.get(path) is not MISSING:
            self._change(path, MISSING)
