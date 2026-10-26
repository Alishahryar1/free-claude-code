"""Estimate final model input without omitting large tails or counting carriers."""

import pytest

from free_claude_code.providers.lmstudio import tokens


def estimate(body):
    return tokens.estimate_prepared_input_tokens(body, budget=10**9)


def test_text_parts_and_string_content_have_the_same_cost():
    assert estimate({"messages": [{"content": "hello world"}]}) == estimate(
        {"messages": [{"content": [{"type": "text", "text": "hello world"}]}]}
    )


@pytest.mark.parametrize(
    "field",
    ["text", "reasoning", "arguments", "result", "tool_schema", "response_schema"],
)
def test_large_tails_are_counted_with_bounded_tokenizer_and_json_work(
    monkeypatch, field
):
    visited = []

    def count(text):
        assert len(text) <= 16384
        visited.append(text)
        return len(text)

    monkeypatch.setattr(tokens, "estimate_text_tokens", count)
    dumps = tokens.json.dumps

    def bounded_dumps(value, **kwargs):
        assert not isinstance(value, (dict, list))
        if isinstance(value, str):
            assert len(value) <= 16384
        return dumps(value, **kwargs)

    monkeypatch.setattr(tokens.json, "dumps", bounded_dumps)
    text = 'a\\"\n' * 400000 + "TAIL-MUST-SURVIVE"
    message = {"role": "assistant", "content": "hello"}
    body = {"messages": [message]}
    if field == "text":
        message["content"] = text
    elif field == "reasoning":
        message["reasoning_content"] = text
    elif field == "arguments":
        message["tool_calls"] = [
            {"id": "call", "function": {"name": "read", "arguments": text}}
        ]
    elif field == "result":
        message.update(role="tool", tool_call_id="call", content=text)
    elif field == "tool_schema":
        body["tools"] = [
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"description": text}},
            }
        ]
    else:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": {"description": text}},
        }
    assert estimate(body) >= len(text)
    assert any("TAIL-MUST-SURVIVE" in value for value in visited)


def test_counting_stops_once_budget_is_exceeded(monkeypatch):
    monkeypatch.setattr(tokens, "estimate_text_tokens", len)
    body = {"messages": [{"content": "x" * 40000}], "tools": object()}
    assert tokens.estimate_prepared_input_tokens(body, budget=10) > 10


def test_carriers_metadata_and_generation_controls_are_not_model_text():
    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "readable",
    }
    baseline = estimate({"messages": [message]})
    message.update(
        extra_content={"google": {"thought_signature": "opaque" * 10000}},
        reasoning_details=[{"type": "reasoning.encrypted", "data": "hidden" * 10000}],
    )
    assert (
        estimate(
            {
                "messages": [message],
                "extra_headers": {"Authorization": "secret"},
                "metadata": {"user": "meta" * 1000},
                "max_tokens": 1000000,
            }
        )
        == baseline
    )


@pytest.mark.parametrize(
    "url,cost",
    [
        ("https://image.invalid/a.png", 765),
        ("data:image/png;base64,AAAA", 85),
        ("data:image/png;base64," + "A" * 300000, 100),
    ],
    ids=["url", "small_inline", "large_inline"],
)
def test_image_cost_uses_size_without_tokenizing_pixels(monkeypatch, url, cost):
    def no_pixels(text):
        assert "AAAA" not in text
        return 0

    monkeypatch.setattr(tokens, "estimate_text_tokens", no_pixels)
    assert (
        estimate(
            {
                "messages": [
                    {"content": [{"type": "image_url", "image_url": {"url": url}}]}
                ]
            }
        )
        == 7 + cost
    )
