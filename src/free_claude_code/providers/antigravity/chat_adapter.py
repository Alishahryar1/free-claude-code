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
ANTHROPIC_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
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
        envelope = openai_chat_to_cloudcode(body, project_id=project_id)
        envelope = _strip_anthropic_billing_header(envelope)
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


def _strip_anthropic_billing_header(envelope: dict[str, Any]) -> dict[str, Any]:
    """Remove Claude Code billing metadata before sending a Cloud Code request."""

    request_value = envelope.get("request")
    if not isinstance(request_value, Mapping):
        return envelope

    system_value = request_value.get("systemInstruction")
    if not isinstance(system_value, Mapping):
        return envelope

    raw_parts = system_value.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        return envelope

    first_part = raw_parts[0]
    if not isinstance(first_part, Mapping):
        return envelope
    text = first_part.get("text")
    if not isinstance(text, str) or not text.startswith(ANTHROPIC_BILLING_HEADER_PREFIX):
        return envelope

    separators = [
        (position, delimiter)
        for delimiter in ("\r\n\r\n", "\n\n")
        if (position := text.find(delimiter)) >= 0
    ]
    if not separators:
        return envelope

    position, delimiter = min(separators, key=lambda item: item[0])
    remaining = text[position + len(delimiter) :]

    updated = dict(envelope)
    request = dict(request_value)
    system = dict(system_value)
    parts = list(raw_parts)
    if remaining:
        first = dict(first_part)
        first["text"] = remaining
        parts[0] = first
    else:
        parts.pop(0)

    if parts:
        system["parts"] = parts
        request["systemInstruction"] = system
    else:
        request.pop("systemInstruction", None)
    updated["request"] = request
    return updated


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
