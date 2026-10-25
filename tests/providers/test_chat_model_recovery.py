"""Recovery follows the model serialized by the Chat SDK, not its public alias."""

import pytest

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.groq import GroqProvider
from tests.api.test_delivered_stream_recovery import text_events
from tests.providers.request_factory import make_messages_request
from tests.providers.support import make_provider_config
from tests.providers.test_history_transports import _harness


@pytest.mark.asyncio
async def test_allowed_model_override_uses_effective_wire_model_for_recovery():
    admission = ProviderAdmissionController(
        provider_name="groq",
        rate_limit=1000,
        max_attempts=1,
        base_delay=60,
        max_delay=60,
        jitter=0,
    )

    def reply(bodies):
        if bodies[-1]["model"] == "healthy":
            return 200, text_events("chat", "worked")
        return 503, {"type": "server_error", "message": "Unavailable"}

    async with _harness(
        "chat",
        reply,
        chat_provider_factory=lambda: GroqProvider(
            make_provider_config("fixture", "https://provider.invalid/v1"),
            admission=admission,
        ),
    ) as (_, bodies, provider):

        async def send(model, override, public_model):
            return [
                event
                async for event in provider.stream_messages(
                    make_messages_request(model, extra_body={"model": override}),
                    reasoning=ReasoningPolicy.provider_default(),
                    response_model=public_model,
                )
            ]

        with pytest.raises(ExecutionFailure):
            await send("target-a", "overloaded", "public-a")
        with pytest.raises(ExecutionFailure):
            await send("target-b", "overloaded", "public-b")
        assert [body["model"] for body in bodies] == ["overloaded"]
        events = await send("target-a", "healthy", "public-a")
        assert "worked" in "".join(events)
        assert [body["model"] for body in bodies] == ["overloaded", "healthy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_model", [None, 1, {"bad": "model"}])
async def test_invalid_override_keeps_existing_upstream_rejection(invalid_model):
    async with _harness(
        "chat",
        lambda _: (400, {"type": "invalid_request_error", "message": "Invalid model"}),
        chat_provider_factory=lambda: GroqProvider(
            make_provider_config("fixture", "https://provider.invalid/v1"),
            admission=ProviderAdmissionController(provider_name="groq"),
        ),
    ) as (_, bodies, provider):
        with pytest.raises(ExecutionFailure, match="Invalid model"):
            async for _ in provider.stream_messages(
                make_messages_request("target", extra_body={"model": invalid_model}),
                reasoning=ReasoningPolicy.provider_default(),
            ):
                pass
        assert len(bodies) == 1 and bodies[0]["model"] == invalid_model
