"""Native Gemini uses shared FCC budgets and preserves native client controls."""

from copy import deepcopy

import httpx
import pytest

from free_claude_code.config.provider_catalog import GEMINI_DEFAULT_BASE
from free_claude_code.core.gemini.models import ThinkingConfig
from free_claude_code.core.gemini.native_request import apply_reasoning
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.gemini import GeminiProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    stream_provider_request,
)
from tests.providers.test_gemini import _collect, _request, _sse
from tests.wire_payload import decode_object, object_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "chat", "gemini"])
@pytest.mark.parametrize(
    "model", ["models/gemini-2.5-flash", "models/gemini-3.5-flash-lite", "arbitrary"]
)
@pytest.mark.parametrize("effort", list(ReasoningEffort))
async def test_fcc_effort_uses_shared_budget_on_every_ingress(
    wire: str, model: str, effort: ReasoningEffort
):
    bodies = []

    def reply(request: httpx.Request):
        bodies.append(decode_object(request.content))
        return _sse([{"text": "answer"}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    request = _request(wire).model_copy(update={"model": model}, deep=True)
    original = request.model_dump()
    try:
        await _collect(
            wire,
            stream_provider_request(
                provider, wire, request, reasoning=ReasoningPolicy.on(effort=effort)
            ),
        )
        assert object_payload(bodies[0]["generationConfig"])["thinkingConfig"] == {
            "includeThoughts": True,
            "thinkingBudget": effort.budget_tokens,
        }
        assert request.model_dump() == original
    finally:
        await provider.cleanup()


@pytest.mark.parametrize(
    "model", ["models/gemini-2.5-pro", "models/gemini-3.5-flash-lite", "arbitrary"]
)
@pytest.mark.parametrize(
    "control",
    [
        {"thinkingLevel": "HIGH"},
        {"thinkingLevel": "MINIMAL"},
        {"thinkingBudget": 777},
        {"thinkingBudget": -1},
        {"thinkingBudget": 777, "thinkingLevel": "THINKING_LEVEL_UNSPECIFIED"},
        {},
    ],
)
@pytest.mark.parametrize("visible", [False, True])
def test_native_client_computation_and_visibility_are_preserved(
    model: str, control: JsonObject, visible: bool
):
    thinking = {"includeThoughts": visible, **control}
    body: JsonObject = {
        "model": model,
        "generationConfig": {"thinkingConfig": thinking, "temperature": 0.3},
    }
    before = deepcopy(body)
    apply_reasoning(body, ThinkingConfig.model_validate(thinking).policy(), native=True)
    assert body == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag, capability", [(False, "none"), (True, "unknown"), (None, "unknown")]
)
async def test_discovery_capability_drives_classifier_allowance(
    flag: bool | None, capability: str
):
    bodies = []

    def reply(request: httpx.Request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent"],
                            **({"thinking": flag} if flag is not None else {}),
                        }
                    ]
                },
            )
        bodies.append(decode_object(request.content))
        return _sse([{"text": "verdict"}])

    provider = GeminiProvider(
        make_provider_config(api_key="key", base_url=GEMINI_DEFAULT_BASE),
        admission=immediate_admission(),
        transport=httpx.MockTransport(reply),
    )
    try:
        info = next(iter(await provider.list_model_infos()))
        assert info.reasoning_capability.value == capability
        request = _request("messages").model_copy(
            update={"model": info.model_id, "max_tokens": 64}
        )
        await _collect(
            "messages",
            provider.stream_messages(
                request, reasoning=ReasoningPolicy.prefer_off(), model_info=info
            ),
        )
        config = bodies[0]["generationConfig"]
        if flag is False:
            assert object_payload(config)["maxOutputTokens"] == 64
            assert "thinkingBudget" not in object_payload(
                object_payload(config)["thinkingConfig"]
            )
        else:
            assert "maxOutputTokens" not in object_payload(config)
            assert (
                object_payload(object_payload(config)["thinkingConfig"])[
                    "thinkingBudget"
                ]
                == 0
            )
    finally:
        await provider.cleanup()
