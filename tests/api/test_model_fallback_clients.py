"""Native routing and official client consumption across a model transition."""

import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import create_test_app, provider_manager_for_app
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_midstream_model_fallback import delivered_candidates
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_anthropic_messages_transport import Wire, _sse
from tests.providers.test_anthropic_provider import native_body, provider
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def release_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def test_native_fallback_closes_source_preserves_options_and_filters_candidates():
    bodies = []
    wires = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) > 1:
            assert wires[-1].closed
        wire = Wire(
            [
                _sse(
                    *text_events(
                        "messages",
                        "Hello " if len(bodies) == 1 else "world.",
                        complete=len(bodies) > 1,
                    )
                )
            ]
        )
        wires.append(wire)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    app = create_test_app(
        Settings(
            MODEL="anthropic/primary",
            MODEL_FALLBACKS=["nvidia_nim/excluded", "anthropic/backup"],
        ),
        providers={"anthropic": provider(reply, max_attempts=1)},
    )
    request = native_body(True)
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=request)
    assert response.status_code == 200
    assert [body["model"] for body in bodies] == ["primary", "backup"]
    assert bodies[1]["messages"][:-2] == request["messages"]
    assert bodies[1]["messages"][-2] == {"role": "assistant", "content": "Hello "}
    for key in ("tools", "thinking", "max_tokens", "future_option"):
        assert bodies[1][key] == request[key]
    assert all(wire.closed for wire in wires)
    events = parse_sse_text(response.text)
    assert_completed(events, "messages")
    assert public_text(events, "messages") == "Hello world."
    assert events[0].data["message"]["model"] == request["model"]
    assert provider_manager_for_app(app)._current.active_leases == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["chat", "messages", "responses"])
async def test_responses_sdk_consumes_fallback_and_replays_only_real_tool_history(
    target,
):
    async with (
        _harness(
            "chat",
            lambda _: (200, text_events("chat", "Before. ", complete=False)),
            max_attempts=1,
        ) as (first, _, _),
        _harness(
            target,
            lambda bodies: (
                200,
                tool_events(target, '{"path":"winning"}')
                if len(bodies) == 1
                else text_events(target, "Result."),
            ),
        ) as (second, second_bodies, _),
    ):
        raw = await delivered_candidates(
            [first, second], "responses", tools=_tools("responses")
        )
        http = httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=raw.encode(),
                )
            )
        )
        async with (
            AsyncOpenAI(api_key="fixture", max_retries=0, http_client=http) as client,
            client.responses.stream(model="fixture", input="greet") as stream,
        ):
            seen = [event async for event in stream]
            final = await stream.get_final_response()
        assert seen[-1].type == "response.completed"
        items = [
            item.model_dump(mode="json", exclude_none=True) for item in final.output
        ]
        call = next(item for item in items if item["type"] == "function_call")
        assert call["call_id"] == "call_probe"
        assert call["arguments"] == '{"path":"winning"}'
        assert (
            "".join(part["text"] for item in items for part in item.get("content", []))
            == "Before. "
        )
        history = [
            {"role": "user", "content": "greet"},
            *items,
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "file-data",
            },
        ]
        following = await delivered(
            second("responses", history, tools=_tools("responses")), "responses"
        )
        assert_completed(parse_sse_text(following), "responses")
        assert "file-data" in str(second_bodies[-1])
        assert "The previous provider stream was interrupted" not in str(
            second_bodies[-1]
        )
        assert len(second_bodies) == 2
