"""OpenAI-SDK-shaped adapter backed by Antigravity Cloud Code."""

import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any

from .client import AntigravityClient
from .conversion import (
    StreamState,
    final_chunk,
    gemini_event_chunks,
    openai_chat_to_cloudcode,
)

PROJECT_ID_ENV = "CLOUDCODE_GCP_PROJECT_ID"
DIAGNOSTIC_DISABLE_TOOLS_ENV = "ANTIGRAVITY_DIAGNOSTIC_DISABLE_TOOLS"
DIAGNOSTIC_DISABLE_SYSTEM_ENV = "ANTIGRAVITY_DIAGNOSTIC_DISABLE_SYSTEM"
DIAGNOSTIC_REPLACE_CONTENTS_ENV = "ANTIGRAVITY_DIAGNOSTIC_REPLACE_CONTENTS"
DIAGNOSTIC_REDACT_CONTENT_TEXT_ENV = "ANTIGRAVITY_DIAGNOSTIC_REDACT_CONTENT_TEXT"
DIAGNOSTIC_REDACT_SYSTEM_TEXT_ENV = "ANTIGRAVITY_DIAGNOSTIC_REDACT_SYSTEM_TEXT"
DIAGNOSTIC_SYSTEM_SLICE_ENV = "ANTIGRAVITY_DIAGNOSTIC_SYSTEM_SLICE"
_LOGGER = logging.getLogger(__name__)


class AntigravityChatAdapter:
    """Expose the tiny AsyncOpenAI surface consumed by FCC's chat transport."""

    def __init__(self, client: AntigravityClient, *, base_url: str) -> None:
        self._client = client
        self.base_url = base_url
        # FCC fingerprints replay origins from the OpenAI client metadata.
        # Keep this credential-free: Antigravity authorization is injected by
        # AntigravityClient directly on the Cloud Code request.
        self.default_headers: Mapping[str, str] = {}
        self.api_key: str | None = None
        self.chat = _ChatResource(self)
        self._project_id: str | None = None

    async def project_id(self) -> str:
        if self._project_id is None:
            override = os.getenv(PROJECT_ID_ENV)
            if override and override.strip():
                self._project_id = override.strip()
                return self._project_id

            payload = await self._client.load_code_assist()
            project = payload.get("cloudaicompanionProject")
            if not isinstance(project, str) or not project:
                managed = payload.get("gcpManaged") is True
                hint = (
                    f" Set {PROJECT_ID_ENV} to the Cloud Code project ID."
                    if managed
                    else ""
                )
                raise RuntimeError(
                    "Antigravity loadCodeAssist did not return "
                    f"cloudaicompanionProject.{hint}"
                )
            self._project_id = project
        return self._project_id

    async def create_stream(self, body: dict[str, Any]) -> _AntigravitySDKStream:
        project_id = await self.project_id()
        create_body = body
        if _env_truthy(DIAGNOSTIC_DISABLE_TOOLS_ENV) and body.get("tools"):
            create_body = body.copy()
            create_body.pop("tools", None)
            create_body.pop("tool_choice", None)
            _LOGGER.warning(
                "Antigravity diagnostic mode active: tool declarations disabled"
            )
        envelope = openai_chat_to_cloudcode(create_body, project_id=project_id)
        envelope = _apply_diagnostic_overrides(envelope)
        _log_request_metrics(envelope)
        source = self._client.stream_generate_content(envelope)
        model = envelope["model"]
        assert isinstance(model, str)
        return _AntigravitySDKStream(source, model=model)


class _ChatResource:
    def __init__(self, adapter: AntigravityChatAdapter) -> None:
        self.completions = _CompletionsResource(adapter)


class _CompletionsResource:
    def __init__(self, adapter: AntigravityChatAdapter) -> None:
        self._adapter = adapter

    async def create(self, **body: Any) -> _AntigravitySDKStream:
        body.pop("stream", None)
        body.pop("stream_options", None)
        return await self._adapter.create_stream(body)


class _AntigravitySDKStream(AsyncIterator[Any]):
    """Look like an OpenAI AsyncStream to FCC without owning the HTTP client."""

    def __init__(
        self,
        source: AsyncIterator[Mapping[str, Any]],
        *,
        model: str,
    ) -> None:
        self._source = source
        self._state = StreamState(model=model)
        self._pending: list[Any] = []
        self._finished = False
        self._closed = False

    def __aiter__(self) -> _AntigravitySDKStream:
        return self

    async def __anext__(self) -> Any:
        while not self._pending:
            if self._finished or self._closed:
                raise StopAsyncIteration
            try:
                event = await anext(self._source)
            except StopAsyncIteration:
                self._finished = True
                return final_chunk(self._state)
            self._pending.extend(gemini_event_chunks(dict(event), self._state))
        return self._pending.pop(0)

    async def close(self) -> None:
        self._closed = True
        close = getattr(self._source, "aclose", None)
        if close is not None:
            await close()


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _diagnostic_system_slice() -> tuple[int, int] | None:
    raw = os.getenv(DIAGNOSTIC_SYSTEM_SLICE_ENV)
    if raw is None or not raw.strip():
        return None
    start_text, separator, end_text = raw.strip().partition(":")
    if not separator:
        raise ValueError(
            f"{DIAGNOSTIC_SYSTEM_SLICE_ENV} must use START:END character offsets"
        )
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as error:
        raise ValueError(
            f"{DIAGNOSTIC_SYSTEM_SLICE_ENV} must use integer START:END offsets"
        ) from error
    if start < 0 or end <= start:
        raise ValueError(
            f"{DIAGNOSTIC_SYSTEM_SLICE_ENV} requires 0 <= START < END"
        )
    return start, end


def _apply_diagnostic_overrides(envelope: dict[str, Any]) -> dict[str, Any]:
    """Apply opt-in local diagnostics without logging or persisting prompt content."""

    request_value = envelope.get("request")
    if not isinstance(request_value, Mapping):
        return envelope

    disable_system = _env_truthy(DIAGNOSTIC_DISABLE_SYSTEM_ENV)
    replace_contents = _env_truthy(DIAGNOSTIC_REPLACE_CONTENTS_ENV)
    redact_content_text = _env_truthy(DIAGNOSTIC_REDACT_CONTENT_TEXT_ENV)
    redact_system_text = _env_truthy(DIAGNOSTIC_REDACT_SYSTEM_TEXT_ENV)
    system_slice = _diagnostic_system_slice()
    if (
        not disable_system
        and not replace_contents
        and not redact_content_text
        and not redact_system_text
        and system_slice is None
    ):
        return envelope

    updated = dict(envelope)
    request = dict(request_value)
    if disable_system:
        request.pop("systemInstruction", None)
        _LOGGER.warning(
            "Antigravity diagnostic mode active: system instruction disabled"
        )
    elif system_slice is not None:
        system_instruction = request.get("systemInstruction")
        if isinstance(system_instruction, Mapping):
            start, end = system_slice
            request["systemInstruction"] = _slice_text_fields(
                system_instruction, start, end
            )
        _LOGGER.warning(
            "Antigravity diagnostic mode active: system instruction slice=%s:%s",
            system_slice[0],
            system_slice[1],
        )
    elif redact_system_text:
        system_instruction = request.get("systemInstruction")
        if isinstance(system_instruction, Mapping):
            request["systemInstruction"] = _redact_text_fields(system_instruction)
        _LOGGER.warning(
            "Antigravity diagnostic mode active: system instruction text redacted"
        )
    if replace_contents:
        request["contents"] = [
            {"role": "user", "parts": [{"text": "Sadece TEST_OK yaz."}]}
        ]
        _LOGGER.warning(
            "Antigravity diagnostic mode active: conversation contents replaced"
        )
    elif redact_content_text:
        contents = request.get("contents")
        if isinstance(contents, list):
            request["contents"] = [_redact_text_fields(item) for item in contents]
        _LOGGER.warning(
            "Antigravity diagnostic mode active: conversation text redacted"
        )
    updated["request"] = request
    return updated


def _redact_text_fields(value: Any) -> Any:
    """Replace text fields with same-length filler while preserving request shape."""

    if isinstance(value, list):
        return [_redact_text_fields(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    redacted = dict(value)
    text = redacted.get("text")
    if isinstance(text, str):
        redacted["text"] = "A" * len(text)
    parts = redacted.get("parts")
    if isinstance(parts, list):
        redacted["parts"] = [_redact_text_fields(part) for part in parts]
    return redacted


def _slice_text_fields(value: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    """Keep only an overall character range across ordered text parts."""

    sliced = dict(value)
    parts = value.get("parts")
    if not isinstance(parts, list):
        return sliced

    offset = 0
    selected: list[Any] = []
    for raw in parts:
        if not isinstance(raw, Mapping):
            continue
        text = raw.get("text")
        if not isinstance(text, str):
            selected.append(dict(raw))
            continue
        part_start = offset
        part_end = offset + len(text)
        overlap_start = max(start, part_start)
        overlap_end = min(end, part_end)
        if overlap_start < overlap_end:
            item = dict(raw)
            item["text"] = text[
                overlap_start - part_start : overlap_end - part_start
            ]
            selected.append(item)
        offset = part_end
    sliced["parts"] = selected
    return sliced


def _log_request_metrics(envelope: Mapping[str, Any]) -> None:
    """Log request shape without exposing prompts, tool names, or credentials."""

    request = envelope.get("request")
    if not isinstance(request, Mapping):
        return

    contents = request.get("contents")
    content_items = contents if isinstance(contents, list) else []
    content_text_chars = sum(_text_chars(item) for item in content_items)

    system_instruction = request.get("systemInstruction")
    system_chars = _text_chars(system_instruction)
    system_part_chars: list[int] = []
    if isinstance(system_instruction, Mapping):
        system_parts = system_instruction.get("parts")
        if isinstance(system_parts, list):
            system_part_chars = [
                len(part["text"])
                for part in system_parts
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            ]

    tools = request.get("tools")
    tool_items = tools if isinstance(tools, list) else []
    tool_count = 0
    for item in tool_items:
        if not isinstance(item, Mapping):
            continue
        declarations = item.get("functionDeclarations")
        if isinstance(declarations, list):
            tool_count += len(declarations)

    generation = request.get("generationConfig")
    max_output_tokens: Any = None
    thinking_level: Any = None
    thinking_budget: Any = None
    if isinstance(generation, Mapping):
        max_output_tokens = generation.get("maxOutputTokens")
        thinking = generation.get("thinkingConfig")
        if isinstance(thinking, Mapping):
            thinking_level = thinking.get("thinkingLevel")
            thinking_budget = thinking.get("thinkingBudget")

    envelope_bytes = len(
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    tools_bytes = len(
        json.dumps(tool_items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    _LOGGER.info(
        "Antigravity request metrics: model=%s project=%s json_bytes=%s "
        "contents=%s content_text_chars=%s system_chars=%s system_parts=%s "
        "system_part_chars=%s tools=%s tool_json_bytes=%s max_output_tokens=%s "
        "thinking_level=%s thinking_budget=%s",
        envelope.get("model"),
        envelope.get("project"),
        envelope_bytes,
        len(content_items),
        content_text_chars,
        system_chars,
        len(system_part_chars),
        system_part_chars,
        tool_count,
        tools_bytes,
        max_output_tokens,
        thinking_level,
        thinking_budget,
    )


def _text_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        total = 0
        text = value.get("text")
        if isinstance(text, str):
            total += len(text)
        parts = value.get("parts")
        if isinstance(parts, list):
            total += sum(_text_chars(part) for part in parts)
        return total
    if isinstance(value, list):
        return sum(_text_chars(item) for item in value)
    return 0
