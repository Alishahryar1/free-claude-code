"""Regression fixture from one authorized Copilot Responses capture.

Opaque identifiers in the checked-in fixture are stable SHA256 labels. Their
changes, output positions, call identities, event order, and synthetic arguments
match the raw HTTP capture; no credentials or model reasoning are retained.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import httpx2
import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.github_copilot.types import CopilotEgress
from tests.providers.test_github_copilot_provider import Harness, collect, responses_sse
from tests.providers.test_openai_responses_transport import (
    _completed_event,
    _text_delta,
)


def _capture() -> list[JsonObject]:
    path = Path(__file__).parent / "fixtures" / "copilot_responses_changing_ids.json"
    return cast(
        list[JsonObject], json.loads(path.read_text(encoding="utf-8"))["events"]
    )


def _item(event: JsonObject) -> JsonObject:
    value = event["item"]
    assert isinstance(value, dict)
    return value


def _response(event: JsonObject) -> JsonObject:
    value = event["response"]
    assert isinstance(value, dict)
    return value


def _output(event: JsonObject) -> list[JsonObject]:
    output = _response(event)["output"]
    assert isinstance(output, list)
    assert all(isinstance(item, dict) for item in output)
    return cast(list[JsonObject], output)


def _input(responses: bool, model: str) -> MessagesRequest | OpenAIResponsesRequest:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    if responses:
        return OpenAIResponsesRequest.model_validate(
            {
                "model": model,
                "input": "Call fcc_echo with value FCC_TOOL_OK.",
                "tools": [
                    {"type": "function", "name": "fcc_echo", "parameters": schema}
                ],
            }
        )
    return MessagesRequest.model_validate(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Call fcc_echo with value FCC_TOOL_OK."}
            ],
            "tools": [{"name": "fcc_echo", "input_schema": schema}],
        }
    )


async def _run(
    harness: Harness, request: MessagesRequest | OpenAIResponsesRequest
) -> list[JsonObject]:
    stream = (
        harness.provider.stream_messages(request)
        if isinstance(request, MessagesRequest)
        else harness.provider.stream_responses(request)
    )
    raw = await collect(stream)
    parsed = parse_sse_text(raw)
    if isinstance(request, MessagesRequest):
        assert_anthropic_stream_contract(parsed)
    return [cast(JsonObject, event.data) for event in parsed]


def _assert_native_identity(events: list[JsonObject]) -> list[JsonObject]:
    responses = [_response(event) for event in events if "response" in event]
    assert len({response["id"] for response in responses}) == 1
    by_position: dict[int, str] = {}
    for event in events:
        position = event.get("output_index")
        if not isinstance(position, int):
            continue
        current = _item(event)["id"] if "item" in event else event.get("item_id")
        assert isinstance(current, str)
        if position not in by_position:
            by_position[position] = current
        assert current == by_position[position]
    output = _output(events[-1])
    for position, item in enumerate(output):
        assert item["id"] == by_position[position]
    return output


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True], ids=["messages", "responses"])
@pytest.mark.parametrize("missing_delta_id", [False, True])
async def test_captured_copilot_tool_identity_survives_stream_and_result_continuation(
    tmp_path: Path, responses: bool, missing_delta_id: bool
) -> None:
    capture = _capture()
    if missing_delta_id:
        capture[5].pop("item_id")
    call_id = _item(capture[2])["call_id"]
    harness = Harness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*capture).encode()
    try:
        events = await _run(harness, _input(responses, harness.runtime.name))
        if responses:
            output = _assert_native_identity(events)
            assert len(output) == 1
            assert output[0] == {
                **_output(capture[-1])[0],
                "id": _item(capture[2])["id"],
            }
            followup = OpenAIResponsesRequest.model_validate(
                {
                    "model": harness.runtime.name,
                    "input": [
                        *output,
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": "FCC_TOOL_OK",
                        },
                    ],
                }
            )
        else:
            starts = [
                event["content_block"]
                for event in events
                if event["type"] == "content_block_start"
            ]
            assert starts == [
                {"type": "tool_use", "id": call_id, "name": "fcc_echo", "input": {}}
            ]
            fragments = [
                delta["partial_json"]
                for event in events
                if event["type"] == "content_block_delta"
                and isinstance(delta := event.get("delta"), dict)
                and delta.get("type") == "input_json_delta"
            ]
            assert all(isinstance(fragment, str) for fragment in fragments)
            arguments = json.loads("".join(cast(list[str], fragments)))
            assert arguments == {"value": "FCC_TOOL_OK"}
            followup = MessagesRequest.model_validate(
                {
                    "model": harness.runtime.name,
                    "messages": [
                        {"role": "user", "content": "Call fcc_echo."},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": call_id,
                                    "name": "fcc_echo",
                                    "input": arguments,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": "FCC_TOOL_OK",
                                }
                            ],
                        },
                    ],
                }
            )
        harness.responses_content = responses_sse(
            _text_delta("FCC_TOOL_OK"), _completed_event()
        ).encode()
        await _run(harness, followup)
        body = json.loads(harness.seen[-1].content)
        calls = [item for item in body["input"] if item.get("type") == "function_call"]
        results = [
            item for item in body["input"] if item.get("type") == "function_call_output"
        ]
        assert len(calls) == len(results) == 1
        assert calls[0]["call_id"] == results[0]["call_id"] == call_id
        assert calls[0]["name"] == "fcc_echo"
        assert json.loads(calls[0]["arguments"]) == {"value": "FCC_TOOL_OK"}
        assert results[0]["output"] == "FCC_TOOL_OK"
        if responses:
            assert calls == output
        assert all(wire.closed for wire in harness.wires)
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True], ids=["messages", "responses"])
async def test_copilot_done_only_tool_binds_identity_without_an_added_event(
    tmp_path: Path, responses: bool
) -> None:
    capture = _capture()
    capture = [
        event
        for event in capture
        if event["type"]
        in {"response.created", "response.output_item.done", "response.completed"}
    ]
    harness = Harness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*capture).encode()
    try:
        events = await _run(harness, _input(responses, harness.runtime.name))
        if responses:
            output = _assert_native_identity(events)
            assert output[0]["id"] == _item(capture[1])["id"]
        else:
            starts = [
                event["content_block"]
                for event in events
                if event["type"] == "content_block_start"
            ]
            assert len(starts) == 1
            assert starts[0] == {
                "type": "tool_use",
                "id": _item(capture[1])["call_id"],
                "name": "fcc_echo",
                "input": {},
            }
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True], ids=["messages", "responses"])
@pytest.mark.parametrize("arguments_done_only", [False, True])
async def test_copilot_argument_delta_without_item_metadata_fails_before_inventing_a_tool(
    tmp_path: Path, responses: bool, arguments_done_only: bool
) -> None:
    capture = [
        event for event in _capture() if event["type"] != "response.output_item.added"
    ]
    if arguments_done_only:
        capture = [
            event
            for event in capture
            if event["type"] != "response.function_call_arguments.delta"
        ]
    harness = Harness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*capture).encode()
    emitted: list[JsonObject] = []
    failure: ExecutionFailure | None = None
    try:
        request = _input(responses, harness.runtime.name)
        stream = (
            harness.provider.stream_messages(request)
            if isinstance(request, MessagesRequest)
            else harness.provider.stream_responses(request)
        )
        try:
            async for chunk in stream:
                emitted.extend(
                    cast(JsonObject, event.data) for event in parse_sse_text(chunk)
                )
        except ExecutionFailure as error:
            failure = error
        assert not any(
            isinstance(block := event.get("content_block"), dict)
            and block.get("type") == "tool_use"
            for event in emitted
        )
        assert not any(
            event["type"] == "response.function_call_arguments.delta"
            for event in emitted
        )
        assert failure is not None or emitted[-1]["type"] == "response.failed"
        assert all(wire.closed for wire in harness.wires)
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_copilot_interleaved_tools_use_output_positions_and_preserve_final_replay_payloads(
    tmp_path: Path,
) -> None:
    capture = _capture()
    first = deepcopy(_item(capture[2]))
    second = {**first, "id": "opaque-second-added", "call_id": "call-second"}
    reasoning = {"type": "reasoning", "id": "opaque-reasoning-added", "summary": []}
    final_reasoning = {
        **reasoning,
        "id": "opaque-reasoning-final",
        "summary": [{"type": "summary_text", "text": "synthetic fixture reasoning"}],
        "encrypted_content": "synthetic-final-replay-content",
    }
    events: list[JsonObject] = [
        capture[0],
        capture[1],
        capture[2],
        {"type": "response.output_item.added", "output_index": 1, "item": second},
        {"type": "response.output_item.added", "output_index": 2, "item": reasoning},
    ]
    for number, fragment in enumerate(('{"value":', '"FCC_TOOL_OK"}')):
        events.extend(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": position,
                "item_id": f"opaque-delta-{position}-{number}",
                "delta": fragment,
            }
            for position in (0, 1)
        )
    for position, item in enumerate((first, second)):
        events.append(
            {
                "type": "response.output_item.done",
                "output_index": position,
                "item": {
                    **item,
                    "id": f"opaque-done-{position}",
                    "status": "completed",
                    "arguments": '{"value":"FCC_TOOL_OK"}',
                },
            }
        )
    events.append(
        {
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {**final_reasoning, "id": "opaque-reasoning-done"},
        }
    )
    final_output: list[JsonObject] = [
        {
            **first,
            "id": "opaque-final-0",
            "status": "completed",
            "arguments": '{"value":"FCC_TOOL_OK"}',
        },
        {
            **second,
            "id": "opaque-final-1",
            "status": "completed",
            "arguments": '{"value":"FCC_TOOL_OK"}',
        },
        final_reasoning,
    ]
    events.append(
        {
            "type": "response.completed",
            "response": {**_response(capture[-1]), "output": final_output},
        }
    )
    harness = Harness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*events).encode()
    try:
        received = await _run(harness, _input(True, harness.runtime.name))
        output = _assert_native_identity(received)
        assert output == [
            {**final_output[0], "id": first["id"]},
            {**final_output[1], "id": second["id"]},
            {**final_reasoning, "id": reasoning["id"]},
        ]
        assert output[0]["call_id"] != output[1]["call_id"]
        followup = OpenAIResponsesRequest.model_validate(
            {
                "model": harness.runtime.name,
                "input": [
                    *output,
                    {
                        "type": "function_call_output",
                        "call_id": first["call_id"],
                        "output": "FCC_TOOL_OK",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": second["call_id"],
                        "output": "FCC_TOOL_OK",
                    },
                ],
            }
        )
        harness.responses_content = responses_sse(
            _text_delta("done"), _completed_event()
        ).encode()
        await _run(harness, followup)
        body = json.loads(harness.seen[-1].content)
        assert body["input"][:3] == output
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("field", ["call_id", "name", "type"])
@pytest.mark.parametrize("missing", [False, True])
async def test_copilot_rejects_incomplete_or_changed_tool_identity(
    tmp_path: Path, responses: bool, field: str, missing: bool
) -> None:
    capture = _capture()
    if missing:
        _item(capture[2]).pop(field)
    else:
        _item(capture[11])[field] = "different"
        _output(capture[-1])[0][field] = "different"
    harness = Harness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*capture).encode()
    failed = False
    try:
        try:
            events = await _run(harness, _input(responses, harness.runtime.name))
            failed = any(
                event["type"] in {"error", "response.failed"} for event in events
            )
        except ExecutionFailure:
            failed = True
        assert failed, "Contradictory or incomplete tool identity must not succeed"
        assert all(wire.closed for wire in harness.wires)
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_copilot_committed_failure_retains_public_response_and_item_identity(
    tmp_path: Path,
) -> None:
    capture = _capture()
    capture[3]["delta"] = " " * 70000 + str(capture[3]["delta"])
    capture[-1]["type"] = "response.failed"
    _response(capture[-1])["status"] = "failed"
    _response(capture[-1])["error"] = {
        "code": "server_error",
        "message": "Synthetic upstream failure",
    }
    harness = Harness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*capture).encode()
    try:
        events = await _run(harness, _input(True, harness.runtime.name))
        assert events[-1]["type"] == "response.failed"
        output = _assert_native_identity(events)
        assert output[0]["id"] == _item(capture[2])["id"]
        assert len(harness.seen) == 1
        assert all(wire.closed for wire in harness.wires)
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
async def test_copilot_partial_failure_output_does_not_mask_authentication_refresh(
    tmp_path: Path, responses: bool
) -> None:
    capture = _capture()
    failed = deepcopy(capture[-1])
    failed["type"] = "response.failed"
    _response(failed).update(
        status="failed",
        error={"code": "invalid_api_key", "message": "expired"},
        output=[{"id": "partial-final-tool", "type": "function_call"}],
    )
    recovered = deepcopy(capture)
    _response(recovered[0])["id"] = "recovered-response"
    _item(recovered[2])["id"] = "recovered-tool"

    class RefreshHarness(Harness):
        def openai(self, request: httpx2.Request) -> httpx2.Response:
            if self.runtime.sessions[0].endpoint_calls > 1:
                self.responses_content = responses_sse(*recovered).encode()
            return super().openai(request)

    harness = RefreshHarness(tmp_path, CopilotEgress.RESPONSES)
    harness.responses_content = responses_sse(*capture[:3], failed).encode()
    try:
        events = await _run(harness, _input(responses, harness.runtime.name))
        assert events[-1]["type"] == (
            "response.completed" if responses else "message_stop"
        )
        assert harness.runtime.sessions[0].endpoint_calls == 2
        assert len(harness.seen) == 2
        if responses:
            output = _assert_native_identity(events)
            assert _response(events[-1])["id"] == "recovered-response"
            assert output[0]["id"] == "recovered-tool"
    finally:
        await harness.close()
