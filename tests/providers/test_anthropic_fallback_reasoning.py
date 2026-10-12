"""Explicit native fallback controls retain the raw Anthropic request contract."""

import json

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from tests.providers.test_anthropic_native_transport import collect
from tests.providers.test_anthropic_provider import native_body, provider


def model_metadata(*, adaptive=True, supported_efforts=None):
    return {
        "id": "claude-test",
        "max_tokens": 10000,
        "capabilities": {
            "thinking": {
                "supported": True,
                "types": {
                    "adaptive": {"supported": adaptive},
                    "enabled": {"supported": not adaptive},
                },
            },
            "effort": {
                "supported": adaptive,
                **{
                    level: {"supported": True}
                    for level in supported_efforts
                    or ["low", "medium", "high", "xhigh", "max"]
                },
            },
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("effort", ["off", "low", "medium", "high", "xhigh", "max"])
async def test_native_override_replaces_controls_and_preserves_extensions(
    adaptive, effort
):
    received = []
    lookups = []

    def handle(request):
        if request.method == "GET":
            lookups.append(request.url.path)
            return httpx.Response(200, json=model_metadata(adaptive=adaptive))
        received.append(request)
        if len(received) == 1:
            return httpx.Response(529, json={"error": {"type": "overloaded_error"}})
        return httpx.Response(200, json={"type": "message", "content": []})

    p = provider(handle)
    body = native_body()
    body["thinking"] = {
        "type": "enabled",
        "budget_tokens": 6000,
        "display": "omitted",
        "future": "keep",
    }
    body["output_config"] = {
        "effort": "high",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
        "future": True,
    }
    body["extra_body"] = {"new_option": {"keep": True}}
    body["betas"] = ["request-beta"]
    original = json.loads(json.dumps(body))
    reasoning = (
        ReasoningPolicy.off()
        if effort == "off"
        else ReasoningPolicy.on(effort=ReasoningEffort(effort))
    )
    try:
        await collect(
            p,
            body,
            reasoning_override=reasoning,
            request_headers={"anthropic-beta": "header-beta"},
        )
        assert lookups == ["/v1/models/claude-test"]
        sent, retried = [json.loads(request.content) for request in received]
        assert sent == retried and body == original
        assert sent["max_tokens"] == 10000
        assert (
            sent["messages"] == original["messages"]
            and sent["tools"] == original["tools"]
        )
        assert sent["future_option"] == original["future_option"]
        assert sent["new_option"] == {"keep": True} and "extra_body" not in sent
        assert sent["output_config"]["format"] == original["output_config"]["format"]
        assert sent["output_config"]["future"] is True
        assert all(
            request.headers["anthropic-beta"] == "header-beta,request-beta"
            for request in received
        )
        thinking = sent["thinking"]
        assert thinking["future"] == "keep"
        if effort == "off":
            assert thinking == {"type": "disabled", "future": "keep"}
            assert "effort" not in sent["output_config"]
        elif adaptive:
            assert thinking == {
                "type": "adaptive",
                "display": "omitted",
                "future": "keep",
            }
            assert sent["output_config"]["effort"] == effort
        else:
            assert thinking["type"] == "enabled" and thinking["display"] == "omitted"
            budget = reasoning.numeric_budget_tokens
            assert budget is not None
            assert thinking["budget_tokens"] == min(max(1024, budget), 9999)
            assert "effort" not in sent["output_config"]
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unknown", "unsupported_effort", "small_output"])
async def test_native_override_uses_existing_capability_validation(case):
    sent = []

    def handle(request):
        if request.method == "GET":
            if case == "unknown":
                return httpx.Response(404, json={"error": "unknown"})
            return httpx.Response(
                200,
                json=model_metadata(
                    adaptive=case != "small_output", supported_efforts=["low"]
                ),
            )
        sent.append(request)
        raise AssertionError("Invalid controls reached generation")

    p = provider(handle)
    body = native_body()
    if case == "small_output":
        body["max_tokens"] = 1024
    try:
        with pytest.raises(InvalidRequestError):
            await collect(
                p,
                body,
                reasoning_override=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            )
        assert sent == []
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_native_inherit_preserves_unknown_controls_without_metadata():
    sent = []

    def handle(request):
        assert request.method == "POST"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"type": "message", "content": []})

    p = provider(handle)
    body = native_body()
    body["thinking"] = {"type": "future_mode", "future_budget": 7}
    body["output_config"] = {"effort": "future_effort", "future": True}
    try:
        await collect(p, body, reasoning_override=None)
        assert sent == [body]
    finally:
        await p.cleanup()
