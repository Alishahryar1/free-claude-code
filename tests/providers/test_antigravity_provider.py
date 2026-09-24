"""End-to-end provider wiring over a deterministic Cloud Code double."""

from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest

from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.antigravity import provider as provider_module
from free_claude_code.providers.antigravity.auth import AntigravityAuthManager
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, make_provider_config

pytestmark = pytest.mark.asyncio


class FakeCloudCode:
    def __init__(self) -> None:
        self.envelopes: list[Mapping[str, Any]] = []
        self.closed = False

    async def load_code_assist(self) -> Mapping[str, Any]:
        return {"cloudaicompanionProject": "project-1"}

    async def fetch_available_models(self) -> Mapping[str, Any]:
        return {"models": {"gemini-test": {"displayName": "Gemini Test"}}}

    async def stream_generate_content(
        self, payload: Mapping[str, Any]
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.envelopes.append(payload)
        yield {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 1,
                },
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "hello"}]},
                    }
                ],
            }
        }

    async def close(self) -> None:
        self.closed = True


def _provider(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[provider_module.AntigravityProvider, FakeCloudCode]:
    fake = FakeCloudCode()

    def construct(**_: object) -> FakeCloudCode:
        return fake

    monkeypatch.setattr(provider_module, "AntigravityClient", construct)
    provider = provider_module.AntigravityProvider(
        make_provider_config(None, "https://example.test"),
        auth=cast(AntigravityAuthManager, object()),
        admission=immediate_admission(provider_name="ANTIGRAVITY", max_attempts=1),
    )
    return provider, fake


@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_provider_reuses_fcc_public_protocol_transports(
    monkeypatch: pytest.MonkeyPatch, wire: str
) -> None:
    provider, fake = _provider(monkeypatch)
    try:
        if wire == "messages":
            stream = provider.stream_messages(
                make_messages_request("gemini-test", thinking=None)
            )
        else:
            stream = provider.stream_responses(
                OpenAIResponsesRequest(model="gemini-test", input="hello")
            )

        output = "".join([event async for event in stream])

        assert "hello" in output
        assert len(fake.envelopes) == 1
        envelope = fake.envelopes[0]
        assert envelope["project"] == "project-1"
        assert envelope["model"] == "gemini-test"
    finally:
        await provider.cleanup()

    assert fake.closed


async def test_provider_discovers_models_from_antigravity_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _fake = _provider(monkeypatch)
    try:
        infos = await provider.list_model_infos()
        assert {info.model_id for info in infos} == {"gemini-test"}
    finally:
        await provider.cleanup()
