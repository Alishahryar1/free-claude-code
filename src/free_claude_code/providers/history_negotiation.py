"""Request-local readable history negotiation and server-lifetime suggestions."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, cast

from free_claude_code.core.history_replay import (
    HistoryMode,
    HistoryPath,
    ReplayOrigin,
    decode_replay,
    is_replay,
    normalize_history_source,
    prepare_history,
    readable_reasoning,
    resolve_messages_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

type HistoryKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class HistorySuggestion:
    revision: int = 0
    mode: HistoryMode | None = None


class HistoryModeCache:
    """Loop-owned CAS entries. Keys contain no account, credential or transcript."""

    def __init__(self) -> None:
        self._entries: dict[HistoryKey, HistorySuggestion] = {}

    def read(self, key: HistoryKey) -> HistorySuggestion:
        return self._entries.get(key, HistorySuggestion())

    def update(
        self, key: HistoryKey, revision: int, mode: HistoryMode | None
    ) -> int | None:
        if self.read(key).revision != revision:
            return None
        self._entries[key] = HistorySuggestion(revision + 1, mode)
        return revision + 1

    def clear(self) -> None:
        self._entries.clear()


def resolve_history_overrides(body: Mapping[str, Any]) -> JsonObject:
    """Apply SDK overrides before identifying the model or projecting history."""
    result = deepcopy(dict(body))
    extra = result.get("extra_body")
    if isinstance(extra, dict):
        messages = extra.get("messages")
        if isinstance(messages, list):
            _normalize_chat_override(messages)
        for key in ("model", "messages", "input"):
            if key in extra:
                result[key] = extra.pop(key)
        if not extra:
            result.pop("extra_body", None)
    return result


def _normalize_chat_override(messages: list[JsonValue]) -> None:
    """Resolve raw scalar copies before conversion provenance is lost."""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        details = message.get("reasoning_details")
        if not isinstance(details, list):
            continue
        full: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            native: JsonObject = {"reasoning_details": [detail]}
            for key in ("data", "signature"):
                carrier = detail.get(key)
                if isinstance(carrier, str) and is_replay(carrier):
                    native = decode_replay(carrier).native
                    break
            full.extend(
                text for text, summary in readable_reasoning(native) if not summary
            )
        represented = set(full)
        if full:
            represented.update(separator.join(full) for separator in ("", "\n", "\n\n"))
        for key in ("reasoning_content", "reasoning"):
            if isinstance(text := message.get(key), str) and text in represented:
                message.pop(key)


def effective_body(body: Mapping[str, Any]) -> JsonObject:
    result = deepcopy(dict(body))
    extra = result.pop("extra_body", None)
    if isinstance(extra, dict):

        def merge(target: dict, overlay: dict) -> None:
            for key, value in overlay.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value

        merge(result, extra)
    return result


class HistoryNegotiation:
    """Visit at most three modes, preserving the source across corrections."""

    def __init__(self, cache: HistoryModeCache, origin: ReplayOrigin) -> None:
        self._cache = cache
        self.paths: dict[HistoryPath, HistoryPath] = {}
        self._source: JsonObject = {}
        self._render: (
            Callable[[HistoryMode, dict[HistoryPath, HistoryPath]], JsonObject] | None
        ) = None
        self._start((origin.provider, origin.endpoint, origin.protocol, origin.model))

    def _start(self, key: HistoryKey) -> None:
        self._key = key
        suggestion = self._cache.read(key)
        self._revision: int | None = suggestion.revision
        self._cached_mode = suggestion.mode
        self._order = list(
            dict.fromkeys(
                [
                    *([suggestion.mode] if suggestion.mode is not None else []),
                    *HistoryMode,
                ]
            )
        )
        self.mode = self._order.pop(0)
        self._seen: list[JsonObject] = []
        self._learnable = False
        self._accepted = False

    def prepare(
        self,
        body: Mapping[str, Any],
        origin: ReplayOrigin,
        *,
        reasoning_field: str = "reasoning_content",
        structured_details: bool = True,
        finalize: Callable[[JsonObject], JsonObject] | None = None,
        continuation: bool = False,
        native_chat: bool = False,
    ) -> JsonObject:
        self._source = normalize_history_source(
            resolve_history_overrides(body), origin.protocol
        )
        origin = replace(origin, model=str(self._source["model"]))
        if origin.protocol == "messages":
            messages = self._source.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "assistant"
                        and isinstance(message.get("content"), list)
                    ):
                        message["content"] = resolve_messages_replay(
                            cast(list[JsonValue], message["content"])
                        )
        key = (origin.provider, origin.endpoint, origin.protocol, origin.model)
        if key != self._key:
            self._start(key)

        def render(
            mode: HistoryMode, paths: dict[HistoryPath, HistoryPath]
        ) -> JsonObject:
            projected = prepare_history(
                self._source,
                origin,
                scope=mode,
                reasoning_field=reasoning_field,
                structured_details=structured_details,
                source_paths=paths,
                continuation=continuation,
                native_chat=native_chat,
            )
            return finalize(projected) if finalize is not None else projected

        self._render = render
        wire = render(self.mode, self.paths)
        effective = effective_body(wire)
        self._seen.append(effective)
        self._learnable = any(
            effective_body(render(mode, {})) != effective for mode in HistoryMode
        )
        return wire

    @property
    def source(self) -> JsonObject:
        return self._source

    def rejected(self) -> JsonObject | None:
        if (
            self._learnable
            and self.mode == self._cached_mode
            and self._revision is not None
        ):
            self._revision = self._cache.update(self._key, self._revision, None)
            self._cached_mode = None
        assert self._render is not None
        while self._order:
            mode = self._order.pop(0)
            effective = effective_body(self._render(mode, {}))
            if effective not in self._seen:
                self.mode = mode
                self._accepted = False
                return deepcopy(self._source)
        return None

    def accepted(self) -> None:
        if not self._accepted and self._learnable and self._revision is not None:
            self._revision = self._cache.update(self._key, self._revision, self.mode)
        self._accepted = True
