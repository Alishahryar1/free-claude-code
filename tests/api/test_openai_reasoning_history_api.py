"""Public API routes recover multi-turn OpenAI reasoning history the upstream rejects (#2080)."""

import asyncio
import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.openai_api.provider import OpenAIAPIProvider
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_history_transports import _events_for

TURNS = ("one", "two")


def _turn(name):
    """One upstream reply carrying an opaque reasoning item ``rs_<name>``."""
    return json.loads(
        json.dumps(_events_for("responses"))
        .replace("rs_a:rs_b", f"rs_{name}")
        .replace("Find 17.", f"Thought {name}.")
    )


def _rejection(item_id):
    # OpenAI's shape: ``param`` null and the item named unquoted.
    return {
        "message": f"The encrypted content for item {item_id} could not be verified. "
        "Reason: Encrypted content organization_id did not match the target organization.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_encrypted_content",
    }


def _opaque(body):
    return [
        item
        for item in body["input"]
        if item.get("type") == "reasoning" and item.get("encrypted_content")
    ]


def _replies(responder):
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        status, payload = responder(sent)
        if status != 200:
            return httpx2.Response(status, json={"error": payload})
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(f"data: {json.dumps(event)}\n\n" for event in payload),
        )

    provider = OpenAIAPIProvider(
        make_provider_config("platform-key", "https://api.openai.com/v1"),
        admission=immediate_admission(provider_name="openai_api", max_attempts=5),
        transport=httpx2.MockTransport(reply),
    )
    app = create_test_app(
        Settings(
            MODEL="openai_api/test-model",
            REASONING_POLICY="high",
            ENABLE_WEB_SERVER_TOOLS=False,
        ),
        providers={"openai_api": provider},
    )
    return TestClient(app), sent


def _post(client, wire, streaming, history):
    if wire == "messages":
        body = {
            "model": "openai_api/test-model",
            "max_tokens": 100,
            "stream": streaming,
            "messages": history,
        }
    else:
        body = {"model": "openai_api/test-model", "stream": True, "input": history}
    response = client.post(f"/v1/{wire}", json=body)
    assert response.status_code == 200, response.text
    return response


def _saved(response, wire, streaming):
    """What the client stores and sends back on its next turn."""
    if wire == "responses":
        events = parse_sse_text(response.text)
        assert events[-1].event == "response.completed", response.text
        return events[-1].data["response"]["output"]
    if not streaming:
        return [{"role": "assistant", "content": response.json()["content"]}]

    async def chunks():
        yield response.text

    message, error, saw_stop = asyncio.run(aggregate_anthropic_sse_to_message(chunks()))
    assert error is None and saw_stop, response.text
    return [{"role": "assistant", "content": message["content"]}]


@pytest.mark.parametrize("reject", ["first", "every"])
@pytest.mark.parametrize(
    "wire,streaming", [("messages", True), ("messages", False), ("responses", True)]
)
def test_rejected_reasoning_in_a_multi_turn_history_is_corrected(
    wire, streaming, reject
):
    turn = iter(TURNS)
    client, _ = _replies(lambda _sent: (200, _turn(next(turn))))
    history = []
    with client:
        for name in TURNS:
            history.append({"role": "user", "content": f"question {name}"})
            history += _saved(_post(client, wire, streaming, history), wire, streaming)
    history.append({"role": "user", "content": "continue"})

    def responder(sent):
        opaque = _opaque(sent[-1])
        if opaque and (len(sent) == 1 or reject == "every"):
            return 400, _rejection(opaque[0]["id"])
        return 200, _events_for("responses")

    client, sent = _replies(responder)
    with client:
        _saved(_post(client, wire, streaming, history), wire, streaming)

    first, second = _opaque(sent[0])
    assert len(sent) == (2 if reject == "first" else 3)
    assert first not in sent[-1]["input"]
    assert {"role": "assistant", "content": "Thought one."} in sent[-1]["input"]
    assert (second in sent[-1]["input"]) is (reject == "first")
