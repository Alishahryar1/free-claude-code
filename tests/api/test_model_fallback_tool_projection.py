"""A failed native tool snapshot must preserve routing and the provider error."""

from collections.abc import AsyncIterator
from copy import deepcopy

import pytest

from free_claude_code.api.response_streams import (
    openai_responses_sse_streaming_response,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import ResponsesToolPolicy
from free_claude_code.core.stream_output import StreamOutput
from tests.api.test_delivered_stream_recovery import public_text, text_events
from tests.api.test_midstream_model_fallback import delivered_candidates
from tests.api.test_response_streams import _json_error
from tests.core.openai_responses.test_client_tool_discovery import SEARCH
from tests.providers.test_history_transports import _harness
from tests.wire_payload import array_payload, object_payload, string_payload


@pytest.fixture(autouse=True)
def public_responses(monkeypatch: pytest.MonkeyPatch):
    async def response(wire: str, body: AsyncIterator[StreamOutput]):
        assert wire == "responses"
        return await openai_responses_sse_streaming_response(
            body,
            headers={},
            pre_start_error_response=_json_error,
            request_id="failed-tool-projection",
        )

    monkeypatch.setattr("tests.api.test_midstream_model_fallback._response", response)


def interrupted_search(
    bodies: list[JsonObject],
    status: str | None,
    *,
    text: str = "",
    terminal: str = "failed",
):
    events = text_events("responses", text, complete=False)
    if not text:
        events = events[:1]
    response = deepcopy(object_payload(text_events("responses", text)[-1]["response"]))
    if not text:
        response.update(output=[])
    call = {
        "type": "function_call",
        "id": "fc_abandoned",
        "call_id": "call_abandoned",
        "name": object_payload(array_payload(bodies[-1]["tools"])[0])["name"],
        "arguments": '{"query":',
    }
    if status is not None:
        call["status"] = status
    events.extend(
        [
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**call, "status": "in_progress", "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": call["id"],
                "delta": call["arguments"],
            },
            {
                "type": f"response.{terminal}",
                "extension": {"keep": "envelope"},
                "response": {
                    **response,
                    "status": terminal,
                    "metadata": {"keep": "metadata"},
                    "output": [*array_payload(response["output"]), call],
                    "error": {
                        "code": "server_error",
                        "message": "Native failure during client search",
                        "param": None,
                        "provider_extension": {"keep": True},
                    }
                    if terminal == "failed"
                    else None,
                },
            },
        ]
    )
    return [{**event, "sequence_number": index} for index, event in enumerate(events)]


@pytest.mark.asyncio
@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("status", [None, "completed", "incomplete"])
async def test_failed_client_search_projection_preserves_model_fallback(
    monkeypatch: pytest.MonkeyPatch, release: bool, status: str | None
) -> None:
    async with (
        _harness(
            "responses",
            lambda bodies: (200, interrupted_search(bodies, status)),
            max_attempts=1,
        ) as (first, first_bodies, native),
        _harness(
            "chat", lambda _: (200, text_events("chat", "Recovered")), max_attempts=1
        ) as (second, second_bodies, _),
    ):
        native._tool_policy = ResponsesToolPolicy(client_tool_search=True)
        raw = await delivered_candidates([first, second], "responses", tools=[SEARCH])
    assert len(first_bodies) == len(second_bodies) == 1
    events = parse_sse_text(raw)
    assert public_text(events, "responses") == "Recovered"
    assert events[-1].event == "response.completed"
    assert "call_abandoned" not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [None, "completed", "incomplete"])
async def test_final_search_failure_keeps_native_error_metadata_and_valid_output(
    monkeypatch: pytest.MonkeyPatch, status: str | None
) -> None:
    source = []

    def failed(bodies: list[JsonObject]):
        source.extend(interrupted_search(bodies, status, text="Saved text"))
        return 200, source

    async with _harness("responses", failed, max_attempts=1) as (send, bodies, native):
        native._tool_policy = ResponsesToolPolicy(client_tool_search=True)
        raw = await delivered_candidates([send], "responses", tools=[SEARCH])
    assert len(bodies) == 1
    events = parse_sse_text(raw)
    assert public_text(events, "responses") == "Saved text"
    final = events[-1]
    assert final.event == "response.failed"
    if status == "completed":
        # Explicit completion reaches representation validation. Invalid search
        # arguments fail atomically before publishing the call.
        assert (
            object_payload(object_payload(final.data["response"])["error"])["type"]
            == "api_error"
        )
        assert "call_abandoned" not in raw
        return
    assert final.data["extension"] == source[-1]["extension"]
    for field in ("error", "metadata", "usage"):
        assert (
            object_payload(final.data["response"])[field]
            == object_payload(source[-1]["response"])[field]
        )
    assert (
        object_payload(final.data["response"])["output"]
        == array_payload(object_payload(source[-1]["response"])["output"])[:1]
    )
    assert "call_abandoned" not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["completed", "incomplete"])
async def test_invalid_successful_search_snapshot_still_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
) -> None:
    async with (
        _harness(
            "responses",
            lambda bodies: (
                200,
                interrupted_search(bodies, "completed", terminal=terminal),
            ),
            max_attempts=1,
        ) as (first, first_bodies, native),
        _harness("chat") as (second, second_bodies, _),
    ):
        native._tool_policy = ResponsesToolPolicy(client_tool_search=True)
        raw = await delivered_candidates([first, second], "responses", tools=[SEARCH])
    assert len(first_bodies) == 1 and not second_bodies
    final = parse_sse_text(raw)[-1]
    assert final.event == "response.failed"
    assert "Invalid client search arguments" in string_payload(
        object_payload(object_payload(final.data["response"])["error"])["message"]
    )
    assert "call_abandoned" not in raw
