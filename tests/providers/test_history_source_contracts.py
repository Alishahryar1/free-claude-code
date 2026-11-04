"""Source ownership survives protocol conversion and targeted corrections."""

from copy import deepcopy
from typing import Any, cast

import httpx
import pytest

from free_claude_code.core.anthropic.conversion import AnthropicToOpenAIConverter
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    prepare_history,
)
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    HistoryNegotiation,
)
from free_claude_code.providers.history_replay import (
    history_retry_body,
    normalize_messages_history,
)


def origin(protocol="chat"):
    return ReplayOrigin("provider", protocol, "https://test.invalid", "key", "model")


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_messages_normalization_keeps_distinct_thoughts_but_not_aliases(
    wrapped, separate
):
    thought = "Stored thought"
    block = {"type": "thinking", "thinking": thought}
    if wrapped:
        block = {
            "type": "redacted_thinking",
            "data": encode_replay(
                ReplayRecord(origin(), {"reasoning_content": thought})
            ),
        }
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {
                    "role": "assistant",
                    "reasoning_content": "Independent thought" if separate else thought,
                    "content": [block, {"type": "text", "text": "Answer"}],
                }
            ],
        }
    )
    before = request.model_dump()
    normalized = normalize_messages_history(request)
    converted = AnthropicToOpenAIConverter.convert_messages(
        normalized.messages, structured_reasoning_details=True
    )
    wire = cast(dict[str, Any], prepare_history({"messages": converted}, origin()))
    assert wire["messages"][0]["reasoning_content"].count(thought) == 1
    assert (
        "Independent thought" in wire["messages"][0]["reasoning_content"]
    ) is separate
    assert request.model_dump() == before


@pytest.mark.parametrize("negotiate", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_singleton_responses_id_correction_uses_owned_array_source(negotiate, override):
    item = {
        "type": "reasoning",
        "id": "rs_bad:id",
        "summary": [],
        "encrypted_content": "cipher",
    }
    body = {"model": "model", "input": item}
    if override:
        body = {"model": "model", "input": "Ignored", "extra_body": {"input": item}}
    before = deepcopy(body)
    state = HistoryNegotiation(HistoryModeCache(), origin("responses"))
    wire = state.prepare(body, origin("responses"))
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://test.invalid"),
        json={
            "error": {
                "message": "Invalid input[0].id: only letters and characters allowed",
                "param": "input[0].id",
            }
        },
    )
    error = httpx.HTTPStatusError(
        "rejected", request=response.request, response=response
    )
    result = history_retry_body(
        error,
        wire if negotiate else {"input": item},
        "responses",
        source=state.source if negotiate else None,
        source_paths=state.paths if negotiate else None,
    )
    assert result is not None
    assert result["input"][0]["id"].startswith("rs_")
    assert ":" not in result["input"][0]["id"]
    assert body == before


def test_unsigned_rejection_never_converts_thinking_into_prose():
    body = {
        "input": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Private thought"}],
                "summary": [],
            }
        ]
    }
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://test.invalid"),
        json={
            "error": {
                "code": "invalid_encrypted_content",
                "message": "Invalid reasoning",
                "param": "input[0]",
            }
        },
    )
    error = httpx.HTTPStatusError(
        "rejected", request=response.request, response=response
    )
    assert history_retry_body(error, body, "responses") is None


def test_equal_full_and_summary_text_remain_distinct_representations():
    from free_claude_code.core.history_replay import readable_reasoning

    assert readable_reasoning(
        {
            "content": [{"type": "reasoning_text", "text": "Same words"}],
            "summary": [{"type": "summary_text", "text": "Same words"}],
        }
    ) == [("Same words", False), ("Same words", True)]
