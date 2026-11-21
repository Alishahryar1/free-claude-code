"""Recognizing media rejections and replacing media in Chat requests."""

from typing import Any

import httpx2
import openai
import pytest

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_chat import ChatToolResultImages
from free_claude_code.providers.openai_chat.media_support import (
    TextOnlyModels,
    is_media_rejection,
    omitted_media_note,
    without_media,
)
from free_claude_code.providers.request_correction import Rejection
from tests.wire_payload import array_payload, object_array

VLLM = (
    "ValueError: Received multimodal data but multimodal processing is not "
    "enabled. Use --enable-multimodal flag to enable multimodal processing."
)
_REQUEST = httpx2.Request("POST", "https://upstream.invalid/v1/chat/completions")
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}


def _status_error(message: str, *, status: int = 400, body: object = None):
    response = httpx2.Response(status, request=_REQUEST)
    error_type = {400: openai.BadRequestError, 500: openai.InternalServerError}.get(
        status, openai.APIStatusError
    )
    return error_type(message, response=response, body=body)


def _in_stream(body: object) -> openai.APIError:
    message = body.get("message") if isinstance(body, dict) else None
    return openai.APIError(str(message or "error"), _REQUEST, body=body)


@pytest.mark.parametrize(
    "error",
    [
        _in_stream({"message": VLLM, "type": "invalid_request_error", "code": 400}),
        _status_error(VLLM, body={"message": VLLM, "type": "invalid_request_error"}),
        _status_error("Error code: 400", body={"error": {"message": VLLM}}),
        _status_error("Error code: 400", body={"detail": VLLM}),
        _status_error(f"Error code: 400 - {VLLM}"),
        _status_error(
            "This model does not support image inputs",
            body={"message": "This model does not support image inputs"},
        ),
    ],
)
def test_media_rejections_are_recognized(error: Exception) -> None:
    assert is_media_rejection(error)


@pytest.mark.parametrize(
    "error",
    [
        _status_error("Image is too large", body={"message": "Image is too large"}),
        _status_error(
            "context length exceeded",
            body={"message": "maximum context length is 8192 tokens"},
        ),
        _status_error(VLLM, status=500, body={"message": VLLM}),
        _status_error(VLLM, status=404, body={"message": VLLM}),
        _in_stream({"message": "upstream overloaded"}),
        ValueError(VLLM),
    ],
)
def test_other_failures_are_not_media_rejections(error: Exception) -> None:
    assert not is_media_rejection(error)


def test_each_media_part_becomes_a_note_in_place() -> None:
    audio = {"type": "input_audio", "input_audio": {"data": "YQ==", "format": "wav"}}
    video = {"type": "video_url", "video_url": {"url": "https://v.invalid/a.mp4"}}
    text = {"type": "text", "text": "Read these"}
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [text, IMAGE, audio, IMAGE, video]},
            {"role": "assistant", "content": "Done."},
        ],
    }

    stripped = without_media(body)

    assert stripped is not None
    corrected, replaced = stripped
    assert replaced == 4
    assert corrected["messages"][1]["content"] == [
        text,
        {"type": "text", "text": omitted_media_note("image")},
        {"type": "text", "text": omitted_media_note("audio")},
        {"type": "text", "text": omitted_media_note("image")},
        {"type": "text", "text": omitted_media_note("video")},
    ]
    assert corrected["messages"][0] is body["messages"][0]
    assert corrected["messages"][2] is body["messages"][2]
    assert body["messages"][1]["content"][1] is IMAGE


def test_note_asks_the_model_to_tell_the_user() -> None:
    assert omitted_media_note("image") == (
        "[image omitted: this model cannot accept image input. "
        "Tell the user the image was not seen.]"
    )


def test_relocated_tool_images_keep_their_history_marker() -> None:
    parts: list[Any] = [{"type": "text", "text": "Image-bearing output:"}, IMAGE]
    relocated = ChatToolResultImages(role="user", content=parts)
    body = {"model": "m", "messages": [relocated]}

    stripped = without_media(body)

    assert stripped is not None
    corrected: JsonObject = stripped[0]
    message = object_array(corrected["messages"])[0]
    assert type(message) is ChatToolResultImages
    assert array_payload(message["content"])[1] == {
        "type": "text",
        "text": omitted_media_note("image"),
    }
    assert relocated["content"] is parts
    assert parts[1] is IMAGE


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
        {"model": "m"},
    ],
)
def test_a_body_without_media_is_left_alone(body: JsonObject) -> None:
    assert without_media(body) is None


def test_correction_teaches_only_the_rejected_model() -> None:
    memory = TextOnlyModels()
    with_image: dict[str, Any] = {"messages": [{"role": "user", "content": [IMAGE]}]}
    rejected: dict[str, Any] = {"model": "text-only", **with_image}
    other: dict[str, Any] = {"model": "vision", **with_image}

    proposal = memory.correct(
        Rejection(_in_stream({"message": VLLM}), rejected, rejected)
    )

    assert proposal is not None
    assert proposal.note == "replaced 1 media part with a note"
    corrected: dict[str, Any] = dict(proposal.body)
    assert corrected["messages"][0]["content"] == [
        {"type": "text", "text": omitted_media_note("image")}
    ]
    assert memory.strip(rejected) == (rejected, 0)
    assert proposal.lesson is not None
    proposal.lesson.apply()
    assert memory.strip(rejected) == (proposal.body, 1)
    assert memory.strip(other) == (other, 0)


def test_correction_without_media_proposes_nothing() -> None:
    memory = TextOnlyModels()
    body: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
    }

    assert memory.correct(Rejection(_in_stream({"message": VLLM}), body, body)) is None
