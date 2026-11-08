"""Convert the delivered native suffix without rewriting the client conversation."""

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, cast

from .anthropic.conversion import AnthropicToOpenAIConverter, ReasoningReplayMode
from .anthropic.models import Message, MessagesRequest
from .anthropic.native import NativeMessagesOptions
from .chat_completions.request import (
    ChatRequest,
    chat_messages_body,
    chat_responses_body,
)
from .gemini.models import Content, GeminiRequest
from .gemini.request import (
    build_gemini_chat_request,
    build_gemini_messages_request,
    build_gemini_responses_request,
)
from .history_replay import HistoryPath
from .json_types import JsonObject, JsonValue
from .openai_responses import (
    OpenAIResponsesRequest,
    build_responses_chat_request,
    build_responses_messages_suffix,
    build_responses_provider_request,
)
from .reasoning import ReasoningPolicy
from .stream_recovery import ContinuationSeed

type UpstreamProtocol = Literal["chat", "messages", "responses", "gemini"]


class ContinuationRequest:
    """Own the replaceable readable suffix of one corrected provider source."""

    def __init__(self, protocol: UpstreamProtocol) -> None:
        self.protocol = protocol
        self._injected_rows = 0

    def build(self, body: dict[str, Any], seed: ContinuationSeed) -> dict[str, Any]:
        result = deepcopy(body)
        field = (
            "input"
            if self.protocol == "responses"
            else "contents"
            if self.protocol == "gemini"
            else "messages"
        )
        history = result.get(field, [])
        if isinstance(history, str):
            history = [{"role": "user", "content": history}]
        elif isinstance(history, dict):
            history = [history]
        if self._injected_rows:
            # Targeted opaque-state corrections cannot delete this readable suffix.
            history = history[: -self._injected_rows]
        tail = self.history(seed, result)
        instruction = (
            "The previous provider stream was interrupted. Continue the assistant "
            "response exactly where it stopped. Do not repeat thinking or text "
            "already written."
        )
        if result.get("tools"):
            instruction += (
                " No tool calls from this interrupted assistant turn were delivered "
                "or executed. Any unfinished calls were discarded. Emit the required "
                "tool calls using the provided tools."
            )
        if self.protocol == "gemini":
            tail.append({"role": "user", "parts": [{"text": instruction}]})
        else:
            tail.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": instruction}]
                    if self.protocol == "responses"
                    else instruction,
                }
            )
        result[field] = [*history, *tail]
        self._injected_rows = len(tail)
        return result

    def history(
        self,
        seed: ContinuationSeed,
        body: dict[str, Any],
        *,
        allow_pending_calls: bool = False,
    ) -> list[Any]:
        """Project the delivered prefix into this upstream's history format."""
        if not seed.items:
            return []
        items = deepcopy(list(seed.items))
        model = str(body.get("model", "continuation"))
        options = NativeMessagesOptions(model, int(body.get("max_tokens") or 1))
        if seed.wire_api == "chat":
            if self.protocol == "chat":
                return items
            request = ChatRequest(model=model, messages=items)
            if self.protocol == "messages":
                return cast(list[Any], chat_messages_body(request)["messages"])
            if self.protocol == "responses":
                return cast(list[Any], chat_responses_body(request)["input"])
            from .gemini.native_request import chat_to_gemini

            return cast(
                list[Any],
                chat_to_gemini(request, allow_pending_calls=allow_pending_calls)[
                    "contents"
                ],
            )
        if self.protocol == "gemini":
            from .gemini.native_request import messages_to_gemini, responses_to_gemini

            if seed.wire_api == "gemini":
                # Public frame parsing retains Decimal; use the native JSON
                # model's numeric representation before storing the echo.
                content = Content.model_validate({"role": "model", "parts": items})
                return [
                    content.model_dump(mode="json", by_alias=True, exclude_none=True)
                ]
            if seed.wire_api == "messages":
                # Completed public tool blocks retain their streamed JSON deltas.
                for item in items:
                    if item.get("type") == "tool_use" and "partial_json" in item:
                        item["input"] = json.loads(cast(str, item.pop("partial_json")))
                return cast(
                    list[Any],
                    messages_to_gemini(
                        MessagesRequest(
                            model=model,
                            messages=[Message(role="assistant", content=items)],
                        ),
                        allow_pending_calls=allow_pending_calls,
                    )["contents"],
                )
            return cast(
                list[Any],
                responses_to_gemini(
                    OpenAIResponsesRequest(model=model, input=items),
                    allow_pending_calls=allow_pending_calls,
                )["contents"],
            )
        if seed.wire_api == "messages":
            if self.protocol == "messages":
                return [{"role": "assistant", "content": items}]
            if self.protocol == "chat":
                # Chat has one answer string. Its ordinary conversation converter
                # inserts paragraph separators between blocks; an interrupted
                # response must instead retain the exact characters delivered.
                text = "".join(
                    cast(str, item["text"])
                    for item in items
                    if item.get("type") == "text"
                )
                items = [item for item in items if item.get("type") != "text"]
                if text:
                    items.append({"type": "text", "text": text})
            messages = [Message(role="assistant", content=items)]
            if self.protocol == "chat":
                return AnthropicToOpenAIConverter.convert_messages(
                    messages, structured_reasoning_details=True
                )
            request = MessagesRequest(model=model, messages=messages)
            return build_responses_provider_request(
                request, reasoning=ReasoningPolicy()
            )["input"]
        if seed.wire_api == "responses":
            # A partially published item is local context, not a stored item lookup.
            for item in items:
                item.pop("status", None)
                if self.protocol == "chat" and item.get("type") == "message":
                    parts = cast(list[dict[str, Any]], item.get("content") or [])
                    text = "".join(
                        str(part.get("text") or "")
                        for part in parts
                        if part.get("type") == "output_text"
                    )
                    content = [
                        part for part in parts if part.get("type") != "output_text"
                    ]
                    if text:
                        content.append({"type": "output_text", "text": text})
                    item["content"] = content
            if self.protocol == "responses":
                return items
            request = OpenAIResponsesRequest(model=model, input=items)
            if self.protocol == "chat":
                return cast(
                    list[Any],
                    build_responses_chat_request(
                        request,
                        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
                        structured_reasoning_details=True,
                    ).body["messages"],
                )
            return build_responses_messages_suffix(cast(list[JsonValue], items))
        request = GeminiRequest.model_validate(
            {"model": model, "contents": [{"role": "model", "parts": items}]}
        )
        if self.protocol == "chat":
            prepared = build_gemini_chat_request(request)
        elif self.protocol == "messages":
            prepared = build_gemini_messages_request(request, options)
        else:
            prepared = build_gemini_responses_request(request, ReasoningPolicy())
        return cast(
            list[Any],
            prepared.body["input" if self.protocol == "responses" else "messages"],
        )

    def sent_context(
        self,
        body: JsonObject,
        *,
        source: JsonObject,
        source_paths: Mapping[HistoryPath, HistoryPath],
    ) -> tuple[JsonObject, ...] | None:
        """Capture the injected Gemini suffix after accepted request corrections."""
        if self.protocol != "gemini" or not self._injected_rows:
            return None
        start = len(cast(list[JsonObject], source["contents"])) - self._injected_rows
        return tuple(
            deepcopy(content)
            for index, content in enumerate(cast(list[JsonObject], body["contents"]))
            if cast(int, source_paths[("contents", index)][1]) >= start
        )


def make_text_recovery_body(
    body: dict[str, Any],
    partial_text: str,
    partial_thinking: str = "",
) -> dict[str, Any]:
    """Build a text-only continuation request for an OpenAI-chat upstream."""
    recovery = make_response_recovery_body(body, partial_text, partial_thinking)
    recovery.pop("tools", None)
    recovery.pop("tool_choice", None)
    return recovery


def make_response_recovery_body(
    body: dict[str, Any],
    partial_text: str,
    partial_thinking: str = "",
) -> dict[str, Any]:
    """Build a continuation request that retains the original tool contract."""
    blocks: list[JsonObject] = []
    if partial_thinking:
        blocks.append({"type": "thinking", "thinking": partial_thinking})
    if partial_text:
        blocks.append({"type": "text", "text": partial_text})
    return ContinuationRequest("chat").build(
        body, ContinuationSeed("messages", tuple(blocks))
    )
