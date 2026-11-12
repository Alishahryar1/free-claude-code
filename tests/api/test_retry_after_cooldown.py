"""A long upstream cooldown surfaces through the ordinary wire error."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import httpx2
import pytest

from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers import admission, failure_policy
from tests.api.test_delivered_stream_recovery import public_text, text_events
from tests.api.test_four_protocol_matrix import ingress_request, matrix_app


@pytest.fixture
def cooldown_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monotonic_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(admission, "time", monotonic_time)
    monkeypatch.setattr(failure_policy, "time", monotonic_time)
    return clock


@pytest.mark.parametrize(
    ("wire", "protocol"),
    [
        ("messages", "chat"),
        ("messages", "messages"),
        ("chat", "chat"),
        ("responses", "messages"),
        ("responses", "responses"),
        ("gemini", "gemini"),
    ],
)
@pytest.mark.asyncio
async def test_cooldown_reports_remaining_time_for_initial_and_later_requests(
    wire, protocol, cooldown_clock
):
    calls = []

    def rate_limit(request):
        calls.append(request)
        response = (
            httpx.Response if protocol in {"messages", "gemini"} else httpx2.Response
        )
        return response(
            429,
            headers={"Retry-After": "3600"},
            json={
                "error": {"message": "test quota exhausted", "type": "rate_limit_error"}
            },
        )

    app = matrix_app(protocol, rate_limit, max_attempts=10)
    path, body = ingress_request(wire, wire == "responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        for remaining in (3600, 3300):
            response = await asyncio.wait_for(client.post(path, json=body), timeout=2)
            assert response.status_code == 429, response.text
            assert response.headers["x-should-retry"] == "false"
            assert "retry-after" not in response.headers
            error = response.json()["error"]
            assert error.get("status" if wire == "gemini" else "type") == (
                "RESOURCE_EXHAUSTED" if wire == "gemini" else "rate_limit_error"
            )
            assert "test quota exhausted" in error["message"]
            assert f"{remaining} seconds" in error["message"]
            cooldown_clock.now += 300
    assert len(calls) == 1


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.asyncio
async def test_continuation_cooldown_uses_terminal_event(
    protocol, monkeypatch, cooldown_clock
):
    calls = []
    response_type = httpx.Response if protocol == "messages" else httpx2.Response

    def reply(request):
        calls.append(request)
        if len(calls) == 1:
            events = text_events(protocol, "Hello", complete=False)
            return response_type(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
            )
        return response_type(
            429,
            headers={"Retry-After": "3600"},
            json={
                "error": {
                    "message": "continuation quota exhausted",
                    "type": "rate_limit_error",
                }
            },
        )

    wire = "responses" if protocol == "responses" else "messages"
    path, body = ingress_request(wire, True)
    app = matrix_app(protocol, reply, max_attempts=10)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await asyncio.wait_for(client.post(path, json=body), timeout=2)
    assert response.status_code == 200, response.text
    events = parse_sse_text(response.text)
    assert public_text(events, wire) == "Hello"
    assert events[-1].event == ("response.failed" if wire == "responses" else "error")
    assert "continuation quota exhausted" in response.text
    assert "3600 seconds" in response.text
    assert len(calls) == 2


@pytest.mark.parametrize("with_backup", [False, True])
@pytest.mark.asyncio
async def test_automatic_websearch_preserves_continuation_cooldown(
    with_backup, cooldown_clock
):
    partial = "Partial answer" if with_backup else "a" * 66_000
    calls = []

    def reply(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["model"] == "backup":
            events = text_events("chat", "Answer from backup.")
        elif len(calls) == 1:
            events = text_events("chat", partial, complete=False)
        else:
            return httpx2.Response(
                429,
                headers={"Retry-After": "3600"},
                json={
                    "error": {
                        "message": "continuation quota",
                        "type": "rate_limit_error",
                    }
                },
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    app = matrix_app(
        "chat",
        reply,
        max_attempts=2 if with_backup else 10,
        model_fallbacks=[ModelFallback(model="open_router/backup")]
        if with_backup
        else None,
    )
    path, request = ingress_request("messages", False)
    request["tools"] = [{"name": "web_search", "type": "web_search_20250305"}]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        first = await asyncio.wait_for(client.post(path, json=request), 3)
        if not with_backup:
            second = await asyncio.wait_for(client.post(path, json=request), 3)
            assert second.status_code == 429, second.text
            assert "3600 seconds" in second.json()["error"]["message"]

    assert partial not in json.dumps(calls[1]["messages"])
    models = [body["model"] for body in calls]
    if with_backup:
        assert models == ["backend", "backend", "backup"]
        assert first.status_code == 200, first.text
        assert [block["text"] for block in first.json()["content"]] == [
            "Answer from backup."
        ]
    else:
        assert models == ["backend", "backend"]
        assert first.status_code == 429, first.text
        error = first.json()["error"]
        assert error["type"] == "rate_limit_error"
        assert "continuation quota" in error["message"]
        assert "3600 seconds" in error["message"]
