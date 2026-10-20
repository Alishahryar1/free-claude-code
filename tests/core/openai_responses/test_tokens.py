import json
from collections.abc import Mapping, Sequence

import free_claude_code.core.openai_responses.tokens as responses_tokens
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)


def test_responses_token_estimate_counts_only_input_bearing_fields() -> None:
    base = OpenAIResponsesRequest(
        model="provider/model",
        input="hello",
        instructions="be concise",
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "look up one value",
                "parameters": {"type": "object"},
            }
        ],
        metadata={"large-unrelated-value": "x" * 20_000},
    )
    without_metadata = base.model_copy(update={"metadata": None})

    assert estimate_responses_input_tokens(base) == (
        estimate_responses_input_tokens(without_metadata)
    )


def test_responses_token_estimate_increases_with_input_content() -> None:
    short = OpenAIResponsesRequest(model="provider/model", input="hello")
    long = OpenAIResponsesRequest(
        model="provider/model",
        input="hello " * 1_000,
    )

    assert estimate_responses_input_tokens(short) > 0
    assert estimate_responses_input_tokens(long) > estimate_responses_input_tokens(
        short
    )


def test_structured_estimate_never_serializes_an_unbounded_value(monkeypatch) -> None:
    original_dumps = json.dumps

    def guarded_dumps(value: object, *, ensure_ascii: bool = True) -> str:
        assert not isinstance(value, Mapping)
        assert not (isinstance(value, Sequence) and not isinstance(value, str))
        if isinstance(value, str):
            assert len(value) <= 1_000_000
        return original_dumps(value, ensure_ascii=ensure_ascii)

    monkeypatch.setattr(responses_tokens.json, "dumps", guarded_dumps)
    request = OpenAIResponsesRequest(
        model="provider/model",
        input=[
            {"role": "user", "content": "x" * 1_100_000},
            {"role": "user", "content": "must not be traversed"},
        ],
        tools=[{"type": "function", "name": "lookup"}],
    )

    assert estimate_responses_input_tokens(request) > 0
