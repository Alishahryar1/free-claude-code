import json
from copy import deepcopy

import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.native import (
    NativeResponsesRelay,
    build_native_responses_request,
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy


def _event_payload(frame: str) -> tuple[str, dict[str, object]]:
    lines = frame.splitlines()
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


def test_native_request_preserves_extensions_and_forces_stateless_streaming() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "gateway-model",
            "input": [{"role": "user", "content": "hello"}],
            "instructions": "be concise",
            "stream": False,
            "store": True,
            "previous_response_id": "resp_previous",
            "metadata": {"source": "codex"},
            "future_option": {"enabled": True},
        }
    )

    body = build_native_responses_request(
        request,
        model="upstream-model",
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body == {
        "model": "upstream-model",
        "input": [{"role": "user", "content": "hello"}],
        "instructions": "be concise",
        "stream": True,
        "store": False,
        "metadata": {"source": "codex"},
        "future_option": {"enabled": True},
    }
    assert request.model == "gateway-model"
    assert request.previous_response_id == "resp_previous"


def test_native_request_preserves_complete_multimodal_input_tree() -> None:
    input_tree = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AA==",
                    "detail": "high",
                    "future_image_option": True,
                },
                {"type": "input_image", "file_id": "file_image"},
                {"type": "input_file", "file_id": "file_document"},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call_image",
            "output": [
                {"type": "input_text", "text": "result"},
                {
                    "type": "input_image",
                    "image_url": "https://images.example.test/result.png",
                },
                {"type": "input_file", "file_id": "file_result"},
                {"type": "future_output", "value": 7},
            ],
        },
        {
            "type": "computer_call_output",
            "call_id": "computer_1",
            "output": {
                "type": "computer_screenshot",
                "file_id": "file_screen",
                "future_screenshot_option": "kept",
            },
            "acknowledged_safety_checks": [{"id": "check_1"}],
        },
    ]
    request = OpenAIResponsesRequest.model_validate(
        {"model": "gateway-model", "input": input_tree}
    )

    body = build_native_responses_request(
        request,
        model="upstream-model",
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body["input"] == input_tree


def test_native_request_recovers_mistyped_tool_ids_without_mutating_history() -> None:
    custom: JsonObject = {
        "type": "custom_tool_call",
        "name": "edit",
        "call_id": "call_edit",
        "input": "patch",
        "provider_extension": {"id": "fc_nested"},
    }
    function: JsonObject = {
        "type": "function_call",
        "name": "lookup",
        "call_id": "call_lookup",
        "arguments": "{}",
    }
    remaining: list[JsonValue] = [
        {**custom, "id": "ctc_shared", "call_id": "call_other"},
        {"type": "custom_tool_call_output", "call_id": "call_edit", "output": "done"},
        {"type": "function_call_output", "call_id": "call_lookup", "output": "found"},
        {"type": "item_reference", "id": "fc_shared"},
    ]
    request = OpenAIResponsesRequest(
        model="example",
        input=[
            {**custom, "id": "fc_shared"},
            {**function, "id": "ctc_lookup"},
            *remaining,
        ],
        metadata={"id": "fc_metadata"},
    )
    original = deepcopy(request.model_dump())
    body = build_native_responses_request(
        request, model="example", reasoning=ReasoningPolicy()
    )
    assert body["input"] == [custom, function, *remaining]
    assert body["metadata"] == {"id": "fc_metadata"}
    assert request.model_dump() == original
    assert (
        build_native_responses_request(
            OpenAIResponsesRequest.model_validate(body),
            model="example",
            reasoning=ReasoningPolicy(),
        )
        == body
    )


@pytest.mark.parametrize(
    "item",
    [
        {"type": "custom_tool_call", "id": "ctc_valid"},
        {"type": "function_call", "id": "fc_valid"},
        {"type": "custom_tool_call"},
        {"type": "function_call"},
        {"type": "custom_tool_call", "id": "opaque"},
        {"type": "function_call", "id": "opaque"},
        {"type": "custom_tool_call", "id": None},
        {"type": "function_call", "id": 42},
        {"type": "custom_tool_call_output", "id": "fc_result"},
        {"type": "function_call_output", "id": "ctc_result"},
        {"type": "message", "id": "ctc_message"},
        {"type": "reasoning", "id": "fc_reasoning"},
        {"type": "item_reference", "id": "fc_reference"},
        {"type": "tool_search_call", "id": "fc_search"},
    ],
)
def test_native_request_preserves_ids_outside_known_call_mismatches(
    item: JsonObject,
) -> None:
    request = OpenAIResponsesRequest(model="example", input=[item])
    body = build_native_responses_request(
        request, model="example", reasoning=ReasoningPolicy()
    )
    assert body["input"] == [item]


def test_native_request_preserves_reasoning_when_fcc_did_not_override_it() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "gateway-model",
            "input": "hello",
            "reasoning": {
                "effort": "high",
                "summary": "detailed",
                "future_hint": True,
            },
        }
    )

    body = build_native_responses_request(
        request,
        model="upstream-model",
        reasoning=ReasoningPolicy(
            effort=ReasoningEffort.HIGH,
        ),
    )

    assert body["reasoning"] == {
        "effort": "high",
        "summary": "detailed",
        "future_hint": True,
    }


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    (
        (ReasoningPolicy.off(), {"effort": "none"}),
        (
            ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
            {"effort": "xhigh", "summary": "auto"},
        ),
    ),
)
def test_native_request_applies_fcc_reasoning_override(
    reasoning: ReasoningPolicy,
    expected: dict[str, str],
) -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "gateway-model",
            "input": "hello",
            "reasoning": {"effort": "low", "summary": "detailed"},
        }
    )

    body = build_native_responses_request(
        request,
        model="upstream-model",
        reasoning=reasoning,
    )

    assert body["reasoning"] == expected


def test_native_relay_preserves_payload_and_rewrites_only_response_model() -> None:
    relay = NativeResponsesRelay(public_model="gateway-model")
    created = {
        "type": "response.created",
        "sequence_number": 7,
        "response": {
            "id": "resp_upstream",
            "model": "upstream-model",
            "status": "in_progress",
            "output": [],
            "future_field": {"kept": True},
        },
    }
    item = {
        "type": "response.output_item.added",
        "sequence_number": 8,
        "output_index": 0,
        "item": {
            "id": "item_upstream",
            "type": "hosted_tool_call",
            "call_id": "call_upstream",
            "status": "in_progress",
        },
    }
    completed = {
        "type": "response.completed",
        "sequence_number": 9,
        "response": {
            "id": "resp_upstream",
            "model": "upstream-model",
            "status": "completed",
            "output": [item["item"]],
            "usage": {
                "input_tokens": 11,
                "input_tokens_details": {
                    "cached_tokens": 6,
                    "cache_write_tokens": 2,
                },
                "output_tokens": 4,
                "total_tokens": 15,
            },
        },
    }

    frames = [
        relay.feed("response.created", created),
        relay.feed("response.output_item.added", item),
        relay.feed("response.completed", completed),
    ]

    event_types_and_payloads = [_event_payload(frame) for frame in frames]
    assert [event_type for event_type, _ in event_types_and_payloads] == [
        "response.created",
        "response.output_item.added",
        "response.completed",
    ]
    assert event_types_and_payloads[0][1]["response"] == {
        **created["response"],
        "model": "gateway-model",
    }
    assert event_types_and_payloads[1][1] == item
    assert event_types_and_payloads[2][1]["response"] == {
        **completed["response"],
        "model": "gateway-model",
    }
    assert created["response"]["model"] == "upstream-model"
    assert relay.response_id == "resp_upstream"
    assert relay.terminal_type == "response.completed"


@pytest.mark.parametrize(
    ("timestamp", "expected_type"),
    [
        (1788587503.0, int),
        (1788587503, int),
        (1788587503.25, float),
        (None, type(None)),
    ],
)
@pytest.mark.parametrize("event_type", ["response.created", "response.failed"])
def test_native_relay_serializes_whole_second_timestamps_as_integers(
    timestamp: float | int | None, expected_type: type, event_type: str
) -> None:
    relay = NativeResponsesRelay(public_model="gateway-model")
    original = {
        "type": event_type,
        "response": {
            "id": "resp_upstream",
            "created_at": timestamp,
            "completed_at": timestamp,
            "temperature": 1.0,
            "metadata": {"created_at": 1788587503.0},
        },
    }

    _, payload = _event_payload(relay.feed(event_type, original))

    response = payload["response"]
    assert isinstance(response, dict)
    for key in ("created_at", "completed_at"):
        assert type(response[key]) is expected_type
        assert response[key] == timestamp
        assert type(original["response"][key]) is type(timestamp)
    assert type(response["temperature"]) is float
    metadata = response["metadata"]
    assert isinstance(metadata, dict)
    assert type(metadata["created_at"]) is float


def test_native_relay_rejects_events_after_one_terminal() -> None:
    relay = NativeResponsesRelay(public_model="gateway-model")
    relay.feed(
        "response.incomplete",
        {
            "type": "response.incomplete",
            "response": {"id": "resp_upstream", "model": "upstream-model"},
        },
    )

    with pytest.raises(ValueError, match="after terminal"):
        relay.feed(
            "response.completed",
            {
                "type": "response.completed",
                "response": {"id": "resp_upstream", "model": "upstream-model"},
            },
        )


def test_native_relay_synthesizes_one_failed_terminal_with_public_identity() -> None:
    relay = NativeResponsesRelay(public_model="gateway-model")
    relay.feed(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_upstream",
                "model": "upstream-model",
                "status": "in_progress",
                "output": [],
            },
        },
    )
    failure = ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=502,
        message="Provider stream ended before a terminal event.",
        retryable=True,
    )

    event_type, payload = _event_payload(relay.synthesize_failure(failure))

    assert event_type == "response.failed"
    response = payload["response"]
    assert isinstance(response, dict)
    assert response["id"] == "resp_upstream"
    assert response["model"] == "gateway-model"
    assert response["status"] == "failed"
    assert response["error"] == {
        "message": "Provider stream ended before a terminal event.",
        "type": "api_error",
        "param": None,
        "code": None,
    }
    assert relay.terminal_type == "response.failed"
