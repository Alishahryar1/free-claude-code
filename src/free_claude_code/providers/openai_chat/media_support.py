"""Replace media for models whose upstream rejects it, and remember those models.

Some OpenAI-compatible upstreams serve text-only models that reject a whole
request when it carries any image, audio or video part. Clients resend the
conversation every turn, so one attached image would fail every later request
too. A recognized rejection is corrected by replacing each media part with a
note that tells the model what was removed and asks it to tell the user. The
model is remembered so later requests replace media before they are sent.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import openai

from free_claude_code.providers.request_correction import Proposal, Rejection

_MEDIA_KINDS = {
    "image_url": "image",
    "input_audio": "audio",
    "audio_url": "audio",
    "video_url": "video",
}

_REJECTION_PHRASES = (
    # vLLM, including hosted NVIDIA NIM, serving a model without multimodal
    # support. Hosted NIM reports it inside an HTTP 200 stream.
    "multimodal processing is not enabled",
    "received multimodal data",
    # DeepSeek and other OpenAI-compatible APIs serving a text-only model.
    "does not support image input",
)


def omitted_media_note(kind: str) -> str:
    """The text that stands in for one removed media part."""
    return (
        f"[{kind} omitted: this model cannot accept {kind} input. "
        f"Tell the user the {kind} was not seen.]"
    )


def is_media_rejection(error: Exception) -> bool:
    """Whether an upstream rejected the request because it carried media."""
    if not isinstance(error, openai.APIError):
        return False
    if getattr(error, "status_code", None) not in (None, 400):
        return False
    return any(
        phrase in text for text in _error_texts(error) for phrase in _REJECTION_PHRASES
    )


def without_media(body: Mapping[str, Any]) -> tuple[dict[str, Any], int] | None:
    """Return a copy of ``body`` with each media part replaced by a note.

    Returns ``None`` when the body carries no media. Messages keep their own
    class, which history replay uses as a marker, and ``body`` is not changed.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    replaced = 0
    result: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list) or not any(map(_media_kind, content)):
            result.append(message)
            continue
        parts: list[Any] = []
        for part in content:
            kind = _media_kind(part)
            if kind is None:
                parts.append(part)
                continue
            parts.append({"type": "text", "text": omitted_media_note(kind)})
            replaced += 1
        corrected = copy.copy(message)
        corrected["content"] = parts
        result.append(corrected)
    if not replaced:
        return None
    return {**body, "messages": result}, replaced


class TextOnlyModels:
    """Remember the models whose upstream rejected media, and strip it for them."""

    def __init__(self) -> None:
        self._models: set[str] = set()

    def strip(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Replace media for a remembered model; return the body and the count."""
        model = body.get("model")
        if not isinstance(model, str) or model not in self._models:
            return body, 0
        stripped = without_media(body)
        return stripped if stripped is not None else (body, 0)

    def correct(self, rejection: Rejection) -> Proposal | None:
        """Replace the rejected request's media, and learn its model."""
        stripped = without_media(rejection.body)
        if stripped is None:
            return None
        body, replaced = stripped
        model = body.get("model")
        lesson = TextOnlyModelLearned(self, model) if isinstance(model, str) else None
        noun = "part" if replaced == 1 else "parts"
        return Proposal(body, f"replaced {replaced} media {noun} with a note", lesson)

    def learn(self, model: str) -> None:
        self._models.add(model)


@dataclass(frozen=True, slots=True)
class TextOnlyModelLearned:
    """A model whose upstream rejected media input."""

    memory: TextOnlyModels
    model: str

    def apply(self) -> None:
        self.memory.learn(self.model)


def _media_kind(part: object) -> str | None:
    if not isinstance(part, Mapping):
        return None
    kind = part.get("type")
    return _MEDIA_KINDS.get(kind) if isinstance(kind, str) else None


def _error_texts(error: Exception) -> tuple[str, ...]:
    texts = [str(error)]
    body = getattr(error, "body", None)
    pending = [body]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, Mapping):
            pending.extend(value.get(field) for field in ("message", "detail", "error"))
    return tuple(text.lower() for text in texts)
