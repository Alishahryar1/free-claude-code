"""Gemini tool batches retain their result boundary and signed call ordering."""

from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.gemini.errors import GeminiConversionError
from free_claude_code.core.gemini.native_request import messages_to_gemini
from free_claude_code.core.history_replay import (
    HistoryMode,
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    prepare_history,
)

ORIGIN = ReplayOrigin(
    "gemini", "gemini", "https://google.invalid/v1beta", "test-key", "models/test"
)


def parallel_turn(shared_name=False):
    native: dict[str, Any] = {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "id": call_id,
                    "name": "echo" if shared_name else name,
                    "args": {"value": value},
                },
                "thoughtSignature": f"signature-{call_id}",
            }
            for call_id, name, value in [("a", "first", "A"), ("b", "second", "B")]
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": part["functionCall"]["id"],
                "name": part["functionCall"]["name"],
                "input": part["functionCall"]["args"],
            }
            for part in cast(list[dict[str, Any]], native["parts"])
        ]
        + [
            {
                "type": "redacted_thinking",
                "data": encode_replay(ReplayRecord(ORIGIN, native)),
            }
        ],
    }
    results = [
        {
            "type": "tool_result",
            "tool_use_id": "a",
            "content": {"count": 0, "nested": {"ok": True}},
        },
        {
            "type": "tool_result",
            "tool_use_id": "b",
            "content": "unavailable",
            "is_error": True,
        },
    ]
    return native, assistant, results


@pytest.mark.parametrize("separate_messages", [False, True])
@pytest.mark.parametrize("reverse_results", [False, True])
@pytest.mark.parametrize("shared_name", [False, True])
def test_parallel_results_share_one_turn_before_followup(
    separate_messages, reverse_results, shared_name
):
    native, assistant, results = parallel_turn(shared_name)
    ordered = list(reversed(results)) if reverse_results else results
    result_messages = (
        [{"role": "user", "content": [result]} for result in ordered]
        if separate_messages
        else [{"role": "user", "content": ordered}]
    )
    request = MessagesRequest(
        model="models/test",
        messages=[
            assistant,
            *result_messages,
            {"role": "user", "content": "Summarize"},
        ],
    )
    before = request.model_dump()
    body = messages_to_gemini(request)
    prepared = cast(
        dict[str, Any], prepare_history(body, ORIGIN, scope=HistoryMode.ALL)
    )
    assert prepared["contents"] == [
        native,
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "a",
                        "name": "echo" if shared_name else "first",
                        "response": {"output": {"count": 0, "nested": {"ok": True}}},
                    }
                },
                {
                    "functionResponse": {
                        "id": "b",
                        "name": "echo" if shared_name else "second",
                        "response": {"error": "unavailable"},
                    }
                },
            ],
        },
        {"role": "user", "parts": [{"text": "Summarize"}]},
    ]
    assert request.model_dump() == before


def test_trailing_prompt_and_image_result_keep_their_own_positions():
    native, assistant, results = parallel_turn()
    results[1] = {
        "type": "tool_result",
        "tool_use_id": "b",
        "content": [
            {"type": "text", "text": "Image result"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8=",
                },
            },
        ],
    }
    body = cast(
        dict[str, Any],
        messages_to_gemini(
            MessagesRequest(
                model="models/test",
                messages=[
                    assistant,
                    {"role": "user", "content": [results[1]]},
                    {
                        "role": "user",
                        "content": [results[0], {"type": "text", "text": "Summarize"}],
                    },
                ],
            )
        ),
    )
    response_turn = body["contents"][1]
    assert [part["functionResponse"]["id"] for part in response_turn["parts"]] == [
        "a",
        "b",
    ]
    assert response_turn["parts"][1]["functionResponse"]["parts"] == [
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}
    ]
    assert body["contents"][2] == {"role": "user", "parts": [{"text": "Summarize"}]}
    prepared = cast(
        dict[str, Any], prepare_history(body, ORIGIN, scope=HistoryMode.ALL)
    )
    assert prepared["contents"][0] == native


def test_result_groups_do_not_cross_assistant_turns():
    _, assistant, results = parallel_turn()
    turn = [assistant, *[{"role": "user", "content": [r]} for r in results]]
    body = cast(
        dict[str, Any],
        messages_to_gemini(
            MessagesRequest(model="models/test", messages=[*turn, *deepcopy(turn)])
        ),
    )
    assert [content["role"] for content in body["contents"]] == [
        "model",
        "user",
        "model",
        "user",
    ]
    for index in (1, 3):
        assert [
            part["functionResponse"]["id"] for part in body["contents"][index]["parts"]
        ] == ["a", "b"]


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown", "interleaved"])
def test_invalid_result_batches_are_rejected(invalid):
    _, assistant, results = parallel_turn()
    messages = [assistant, {"role": "user", "content": [results[0]]}]
    if invalid == "duplicate":
        messages.append({"role": "user", "content": results})
    elif invalid == "unknown":
        messages.append(
            {"role": "user", "content": [{**results[1], "tool_use_id": "unknown"}]}
        )
    elif invalid == "interleaved":
        messages.extend(
            [
                {"role": "user", "content": "Summarize"},
                {"role": "user", "content": [results[1]]},
            ]
        )
    with pytest.raises(GeminiConversionError):
        messages_to_gemini(MessagesRequest(model="models/test", messages=messages))


def test_pending_call_projection_retains_partial_results():
    _, assistant, results = parallel_turn()
    body = cast(
        dict[str, Any],
        messages_to_gemini(
            MessagesRequest(
                model="models/test",
                messages=[assistant, {"role": "user", "content": [results[1]]}],
            ),
            allow_pending_calls=True,
        ),
    )
    assert body["contents"][1]["parts"][0]["functionResponse"]["id"] == "b"
