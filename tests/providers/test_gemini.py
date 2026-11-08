"""All client protocols exercise native Gemini over HTTP, including replay."""

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.gemini import GeminiProvider
from tests.providers.support import immediate_admission, make_provider_config


def _sse(parts):
    events: list[dict[str, Any]] = [
        {"candidates": [{"index": 0, "content": {"role": "model", "parts": [part]}}]}
        for part in parts
    ]
    events.append(
        {
            "modelVersion": "actual",
            "candidates": [{"index": 0, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 4,
                "totalTokenCount": 19,
            },
        }
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


def _request(wire, history=None):
    model = "models/test-model"
    if wire == "messages":
        return MessagesRequest(
            model=model,
            messages=history or [{"role": "user", "content": "hello"}],
            max_tokens=100,
        )
    if wire == "responses":
        return OpenAIResponsesRequest(
            model=model, input=history or [{"role": "user", "content": "hello"}]
        )
    if wire == "chat":
        return ChatRequest(
            model=model,
            messages=history or [{"role": "user", "content": "hello"}],
            stream_options={"include_usage": True},
        )
    return GeminiRequest.model_validate(
        {
            "model": model,
            "contents": history or [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {"thinkingConfig": {"includeThoughts": True}},
        }
    )


async def _collect(wire, stream):
    if wire == "messages":
        message, error, stopped = await aggregate_anthropic_sse_to_message(stream)
        assert error is None and stopped
        return message
    chunks = [chunk async for chunk in stream]
    events = parse_sse_text("".join(chunks))
    if wire == "responses":
        return events[-1].data["response"]
    if wire == "gemini":
        accumulator = GeminiResponseAccumulator()
        for event in events:
            accumulator.feed(event.data)
        return accumulator.result()
    accumulator = ChatResponseAccumulator()
    for event in events:
        accumulator.feed(event.data)
    assert chunks[-1] == "data: [DONE]\n\n"
    accumulator.finish()
    return accumulator.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "chat", "gemini"])
@pytest.mark.parametrize("interleaved", [False, True])
async def test_all_ingresses_use_native_gemini_and_replay_signed_parts(
    wire, interleaved
):
    bodies = []
    parts = [
        {"text": "typed thought", "thought": True},
        {
            "text": "<thought>literal answer</thought>",
            "thoughtSignature": "c2lnbmF0dXJl",
        },
    ]

    if interleaved:
        parts.extend(
            [{"text": "a later thought", "thought": True}, {"text": "a later answer"}]
        )

    def reply(request):
        assert request.url.path == "/v1beta/models/test-model:streamGenerateContent"
        assert request.url.params["alt"] == "sse"
        assert request.headers["x-goog-api-key"] == "test-key"
        bodies.append(json.loads(request.content))
        return _sse(parts)

    provider = GeminiProvider(
        make_provider_config(api_key="test-key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        first = await _collect(
            wire,
            getattr(provider, f"stream_{wire}")(
                _request(wire), response_model="public"
            ),
        )
        if wire == "messages":
            history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": first["content"]},
                {"role": "user", "content": "next"},
            ]
        elif wire == "responses":
            history = [
                {"role": "user", "content": "hello"},
                *first["output"],
                {"role": "user", "content": "next"},
            ]
        elif wire == "chat":
            history = [
                {"role": "user", "content": "hello"},
                first["choices"][0]["message"],
                {"role": "user", "content": "next"},
            ]
        else:
            history = [
                {"role": "user", "parts": [{"text": "hello"}]},
                first["candidates"][0]["content"],
                {"role": "user", "parts": [{"text": "next"}]},
            ]
        original = deepcopy(history)
        await _collect(
            wire, getattr(provider, f"stream_{wire}")(_request(wire, history))
        )
        assert history == original
        assert bodies[1]["contents"][1] == {"role": "model", "parts": parts}
        assert "model" not in bodies[0]
        assert "literal answer" in str(first)
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "chat"])
@pytest.mark.parametrize("native_id", [None, "google-call"])
async def test_native_signed_tool_round_trip_keeps_original_parts_and_ids(
    wire, native_id
):
    parts = [
        {"text": "inspect the tool", "thought": True},
        {
            "functionCall": {
                "name": "lookup",
                "args": {"x": 7},
                **({"id": native_id} if native_id else {}),
            },
            "thoughtSignature": "signed-call",
        },
        {"text": "", "thoughtSignature": "late-signature"},
    ]
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return _sse(parts if len(bodies) == 1 else [{"text": "complete"}])

    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    function = {"name": "lookup", "parameters": schema}
    tools = (
        [{"name": "lookup", "input_schema": schema}]
        if wire == "messages"
        else [
            {"type": "function", "function": function}
            if wire == "chat"
            else {"type": "function", **function}
        ]
    )
    initial = _request(wire)
    request = type(initial).model_validate(
        {**initial.model_dump(mode="json", exclude_unset=True), "tools": tools}
    )
    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        first = await _collect(wire, getattr(provider, f"stream_{wire}")(request))
        if wire == "messages":
            call = next(
                block for block in first["content"] if block["type"] == "tool_use"
            )
            history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": "result",
                        }
                    ],
                },
            ]
        elif wire == "responses":
            call = next(
                item for item in first["output"] if item["type"] == "function_call"
            )
            history = [
                {"role": "user", "content": "hello"},
                *first["output"],
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "result",
                },
            ]
        else:
            message = first["choices"][0]["message"]
            call = message["tool_calls"][0]
            history = [
                {"role": "user", "content": "hello"},
                message,
                {"role": "tool", "tool_call_id": call["id"], "content": "result"},
            ]
        next_request = type(request).model_validate(
            {
                **request.model_dump(mode="json", exclude_unset=True),
                "input" if wire == "responses" else "messages": history,
            }
        )
        await _collect(wire, getattr(provider, f"stream_{wire}")(next_request))
        assert bodies[-1]["contents"][1] == {"role": "model", "parts": parts}
        result = bodies[-1]["contents"][2]["parts"][0]["functionResponse"]
        assert result.get("id") == native_id
        assert result["name"] == "lookup"
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["models/gemini-2.5-pro", "models/gemini-3.5-flash-lite", "unknown-model"]
)
async def test_prefer_off_uses_normal_allowance_without_unavailable_disable_control(
    model,
):
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return _sse([{"text": "verdict"}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        await _collect(
            "messages",
            provider.stream_messages(
                MessagesRequest(
                    model=model,
                    messages=[{"role": "user", "content": "classify"}],
                    max_tokens=64,
                ),
                reasoning=ReasoningPolicy.prefer_off(),
            ),
        )
        assert len(bodies) == 1
        config = bodies[0]["generationConfig"]
        assert "maxOutputTokens" not in config
        assert "thinkingBudget" not in config["thinkingConfig"]
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "chat"])
async def test_explicit_off_hides_native_thoughts_but_keeps_signed_replay(wire):
    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(
            lambda request: _sse(
                [
                    {"text": "hidden thought", "thought": True},
                    {"text": "answer", "thoughtSignature": "sig"},
                ]
            )
        ),
    )
    try:
        result = await _collect(
            wire,
            getattr(provider, f"stream_{wire}")(
                _request(wire), reasoning=ReasoningPolicy.off()
            ),
        )
        assert "hidden thought" not in json.dumps(result)
        assert "fcc:history:" in json.dumps(result)
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_explicit_off_rejection_does_not_relax_caller_control():
    from free_claude_code.core.failures import ExecutionFailure

    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400, json={"error": {"message": "Thinking cannot be disabled"}}
        )

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        with pytest.raises(ExecutionFailure):
            await _collect(
                "messages",
                provider.stream_messages(
                    _request("messages"), reasoning=ReasoningPolicy.off()
                ),
            )
        assert len(bodies) == 1
        assert bodies[0]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_native_model_discovery_follows_pages_and_keeps_native_names():
    urls = []

    def reply(request):
        urls.append(str(request.url))
        assert request.headers["x-goog-api-key"] == "key"
        if request.url.params.get("pageToken") == "second":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/second",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/first",
                        "supportedGenerationMethods": ["generateContent"],
                        "thinking": True,
                        "inputTokenLimit": 1000,
                        "outputTokenLimit": 100,
                    },
                    {
                        "name": "models/embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ],
                "nextPageToken": "second",
            },
        )

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        models = {model.model_id: model for model in await provider.list_model_infos()}
        assert set(models) == {"models/first", "models/second"}
        assert models["models/first"].context_window_tokens == 1000
        assert models["models/first"].max_output_tokens == 100
        assert len(urls) == 2
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (
            ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            {"includeThoughts": True, "thinkingBudget": 2048},
        ),
        (
            ReasoningPolicy.on(budget_tokens=777),
            {"includeThoughts": True, "thinkingBudget": 777},
        ),
        (ReasoningPolicy.off(), {"includeThoughts": False, "thinkingBudget": 0}),
    ],
)
async def test_native_reasoning_uses_one_computation_control(reasoning, expected):
    bodies: list[dict[str, Any]] = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return _sse([{"text": "answer"}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        await _collect(
            "messages",
            provider.stream_messages(_request("messages"), reasoning=reasoning),
        )
        assert bodies[0]["generationConfig"]["thinkingConfig"] == expected
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_native_empty_signed_part_does_not_emit_empty_text_block():
    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(
            lambda _: _sse(
                [{"text": "answer"}, {"text": "", "thoughtSignature": "sig"}]
            )
        ),
    )
    try:
        events = parse_sse_text(
            "".join(
                [
                    chunk
                    async for chunk in provider.stream_messages(_request("messages"))
                ]
            )
        )
    finally:
        await provider.cleanup()
    text_starts = [
        event.data
        for event in events
        if event.data.get("type") == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "text"
    ]
    assert len(text_starts) == 1
    stops = [
        event.data["index"]
        for event in events
        if event.data.get("type") == "content_block_stop"
    ]
    assert stops[-1] == text_starts[0]["index"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [400, 429, 503])
async def test_native_inband_error_preserves_retry_semantics_without_history_probes(
    code,
):
    from free_claude_code.core.failures import ExecutionFailure

    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps({'error': {'code': code, 'message': 'provider failure'}})}\n\n",
        )

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=2),
        transport=httpx.MockTransport(reply),
    )
    try:
        with pytest.raises(ExecutionFailure) as caught:
            await _collect(
                "chat",
                provider.stream_chat(
                    _request(
                        "chat",
                        [
                            {
                                "role": "assistant",
                                "content": "old",
                                "reasoning_content": "thinking",
                            },
                            {"role": "user", "content": "next"},
                        ],
                    )
                ),
            )
        assert caught.value.status_code == code
        assert caught.value.retryable == (code in {429, 503})
        assert len(bodies) == (2 if code in {429, 503} else 1)
        assert all(body == bodies[0] for body in bodies)
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_native_history_modes_share_attempt_budget_and_restore_source():
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        if len(bodies) < 3:
            return httpx.Response(
                400, json={"error": {"code": 400, "message": "Unsupported history"}}
            )
        return _sse([{"text": "answer"}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(max_attempts=3),
        transport=httpx.MockTransport(reply),
    )
    history = [
        {
            "role": "model",
            "parts": [{"text": "old thought", "thought": True}, {"text": "old answer"}],
        },
        {"role": "user", "parts": [{"text": "next"}]},
        {
            "role": "model",
            "parts": [
                {"text": "current thought", "thought": True},
                {"text": "current answer"},
            ],
        },
    ]
    original = deepcopy(history)
    try:
        await _collect("gemini", provider.stream_gemini(_request("gemini", history)))
    finally:
        await provider.cleanup()
    assert history == original
    thoughts = [
        [
            part["text"]
            for content in body["contents"]
            for part in content["parts"]
            if part.get("thought")
        ]
        for body in bodies
    ]
    assert thoughts == [["old thought", "current thought"], ["current thought"], []]
