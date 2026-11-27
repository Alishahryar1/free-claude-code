"""The local context guard observes each physical call after history projection."""

import json
import time
from collections.abc import Mapping
from copy import deepcopy
from unittest.mock import patch

import httpx2
import pytest

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    gemini_sse_streaming_response,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.gemini.replay import ReplayBundle, encode_bundle
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.history_replay import replay_origin
from free_claude_code.providers.recovery_policy import RecoveryTiming
from tests.api.test_response_streams import _json_error
from tests.providers.test_gemini_transport import chat_response, gemini_request
from tests.providers.test_google_cache_provenance import history_request
from tests.providers.test_lmstudio import (
    completion_response,
    metadata_response,
    provider_with_http,
    stream_request,
)
from tests.wire_payload import decode_object, object_array, object_payload


def replay_request(wire: str, record: ReplayRecord):
    carrier = encode_replay(record)
    if wire == "gemini":
        return gemini_request(
            contents=[
                {
                    "role": "model",
                    "parts": [
                        {
                            "thoughtSignature": encode_bundle(
                                ReplayBundle("source", (record,))
                            )
                        }
                    ],
                },
                {"role": "user", "parts": [{"text": "continue"}]},
            ]
        )
    if wire == "responses":
        return OpenAIResponsesRequest.model_validate(
            {
                "model": "model",
                "input": [
                    {"type": "reasoning", "encrypted_content": carrier},
                    {"role": "user", "content": "continue"},
                ],
            }
        )
    return MessagesRequest.model_validate(
        {
            "model": "model",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": carrier}],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
@pytest.mark.parametrize("source", ["same", "foreign", "opaque"])
async def test_budget_counts_readable_projection_but_not_opaque_carriers(
    wire: str, source: str
) -> None:
    sent = []

    def reply(request: httpx2.Request):
        if request.method == "GET":
            return metadata_response(1024)
        sent.append(decode_object(request.content))
        return completion_response()

    async with provider_with_http(reply) as provider:
        origin = replay_origin("LMSTUDIO", "chat", "model", client=provider._client)
        native: JsonObject = {"reasoning_content": "replay reasoning " * 6000}
        if source != "same":
            origin = ReplayOrigin("foreign", "messages", "", "", "model")
            native = (
                {
                    "type": "thinking",
                    "thinking": "replay reasoning " * 6000,
                    "signature": "opaque",
                }
                if source == "foreign"
                else {"type": "redacted_thinking", "data": "opaque" * 100000}
            )
        request = replay_request(wire, ReplayRecord(origin, native))
        original = request.model_dump(mode="json")
        if source == "opaque":
            assert [
                event async for event in stream_request(provider, wire, request=request)
            ]
            assert len(sent) == 1
            assert "opaque" not in json.dumps(sent)
        else:
            with pytest.raises(ExecutionFailure) as caught:
                _ = [
                    event
                    async for event in stream_request(provider, wire, request=request)
                ]
            assert caught.value.kind == FailureKind.CONTEXT_WINDOW_EXCEEDED
            assert not sent
        assert request.model_dump(mode="json") == original
        assert provider._admission._capacity._active_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
async def test_correction_revalidates_the_exact_prepared_body(wire: str) -> None:
    sent: list[JsonObject] = []
    inspected: list[JsonObject] = []

    def reply(request: httpx2.Request):
        if request.method == "GET":
            return metadata_response()
        sent.append(decode_object(request.content))
        if len(sent) == 1:
            return httpx2.Response(
                400,
                json={"error": {"message": "Unsupported parameter: stream_options"}},
            )
        return completion_response()

    async with provider_with_http(reply) as provider:
        original_check = provider._chat._validate_prepared_body
        assert original_check is not None

        async def validate(body: Mapping[str, JsonValue]):
            before = deepcopy(dict(body))
            await original_check(body)
            assert body == before
            inspected.append(before)

        provider._chat._validate_prepared_body = validate
        _ = [
            event
            async for event in stream_request(
                provider, wire, request=history_request(wire)
            )
        ]
    assert len(inspected) == len(sent) == 2
    assert "stream_options" in inspected[0] and "stream_options" not in inspected[1]
    for checked, actual in zip(inspected, sent, strict=True):
        assert checked["messages"] == actual["messages"]
        assert checked["tools"] == actual["tools"]
        assert not {"messages", "tools"}.intersection(
            object_payload(checked.get("extra_body", {}))
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
async def test_local_rejection_abandons_probe_without_upstream_rejection(
    wire: str,
) -> None:
    inference = 0
    checks = 0

    def reply(request: httpx2.Request):
        nonlocal inference
        if request.method == "GET":
            return metadata_response()
        inference += 1
        return httpx2.Response(
            503, json={"error": {"message": "temporarily unavailable"}}
        )

    async with provider_with_http(reply) as provider:
        admission = provider._admission
        provider._client.max_retries = 0
        admission._timing = RecoveryTiming(base_delay=0, max_delay=0, jitter=0)
        original_check = provider._chat._validate_prepared_body
        assert original_check is not None

        async def validate(body: Mapping[str, JsonValue]):
            nonlocal checks
            checks += 1
            if checks == 2:
                assert admission._episodes["model"].probe_active
                provider._loaded_context_cache = (time.monotonic(), 1)
            await original_check(body)

        provider._chat._validate_prepared_body = validate
        with (
            patch.object(
                admission, "_attempt_rejected", wraps=admission._attempt_rejected
            ) as rejected,
            patch.object(
                admission, "_attempt_abandoned", wraps=admission._attempt_abandoned
            ) as abandoned,
        ):
            with pytest.raises(ExecutionFailure) as caught:
                _ = [
                    event
                    async for event in stream_request(
                        provider, wire, request=history_request(wire)
                    )
                ]
            assert caught.value.kind == FailureKind.CONTEXT_WINDOW_EXCEEDED
            rejected.assert_not_awaited()
            abandoned.assert_awaited_once()
        assert checks == 2 and inference == 1
        assert admission._capacity._active_attempts == 0


@pytest.mark.asyncio
async def test_internal_continuation_is_rechecked_before_another_inference() -> None:
    inference = 0
    checked = []

    def reply(request: httpx2.Request):
        nonlocal inference
        if request.method == "GET":
            return metadata_response(1024)
        inference += 1
        return chat_response([{"content": "prefix " * 20000}], finish=None)

    async with provider_with_http(reply) as provider:
        original_check = provider._chat._validate_prepared_body
        assert original_check is not None

        async def validate(body: Mapping[str, JsonValue]):
            checked.append(deepcopy(dict(body)))
            await original_check(body)

        provider._chat._validate_prepared_body = validate
        response = await gemini_sse_streaming_response(
            provider.stream_gemini(gemini_request()),
            pre_start_error_response=_json_error,
            request_id="prepared-guard",
        )
        assert isinstance(response, ManagedStreamingResponse)
        try:
            output = "".join([str(frame) async for frame in response.body_iterator])
        finally:
            await response.aclose()
        assert "error" in output
        assert len(checked) == 2
        assert "prefix " in json.dumps(checked[1])
        assert inference == 1
        assert provider._admission._capacity._active_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
@pytest.mark.parametrize("tool_result", [False, True])
async def test_direct_and_tool_result_images_are_counted_once(
    wire: str, tool_result: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from free_claude_code.providers.lmstudio import tokens

    data = "A" * 900000
    sizes: list[int] = []
    sent: list[httpx2.Request] = []
    original_cost = tokens.estimate_image_tokens

    def image_cost(size: int):
        sizes.append(size)
        return original_cost(size)

    monkeypatch.setattr(tokens, "estimate_image_tokens", image_cost)

    def reply(request: httpx2.Request):
        if request.method == "GET":
            return metadata_response(200)
        sent.append(request)
        return completion_response()

    if wire == "gemini":
        image: JsonObject = {"inlineData": {"mimeType": "image/png", "data": data}}
        payload: JsonObject = history_request(wire, replay=tool_result).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        parts = object_array(object_array(payload["contents"])[-1]["parts"])
        if tool_result:
            object_payload(parts[0]["functionResponse"])["parts"] = [image]
        else:
            parts.append(image)
        request = GeminiRequest.model_validate(payload)
    elif wire == "messages":
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        }
        request = history_request(wire, replay=tool_result).model_dump(mode="json")
        if tool_result:
            request["messages"][-1]["content"][0]["content"] = [image]
        else:
            request["messages"][-1]["content"] = [image]
        request = MessagesRequest.model_validate(request)
    else:
        image = {"type": "input_image", "image_url": "data:image/png;base64," + data}
        request = history_request(wire, replay=tool_result).model_dump(mode="json")
        request["input"][-1]["output" if tool_result else "content"] = [image]
        request = OpenAIResponsesRequest.model_validate(request)
    async with provider_with_http(reply) as provider:
        with pytest.raises(ExecutionFailure) as caught:
            _ = [
                event async for event in stream_request(provider, wire, request=request)
            ]
        assert caught.value.kind == FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert sizes == [len(data)]
    assert not sent
