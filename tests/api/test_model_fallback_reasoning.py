"""Each fallback resolves controls from the original request's reasoning intent."""

import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    execution_failure,
    fallback_client,
    messages_payload,
    responses_payload,
)


@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize("client_effort", [None, "high"])
def test_from_client_default_reaches_primary_and_inherited_fallback(
    wire_api, client_effort
):
    primary = ControlledFallbackProvider(failure=execution_failure("primary failed"))
    fallback = ControlledFallbackProvider(text="fallback worked")
    settings = Settings(
        MODEL="nvidia_nim/primary-model",
        MODEL_FALLBACKS=[{"model": "groq/fallback-model"}],
    )
    payload = (
        messages_payload(stream=True) if wire_api == "messages" else responses_payload()
    )
    if client_effort is not None:
        payload["output_config" if wire_api == "messages" else "reasoning"] = {
            "effort": client_effort
        }
    with fallback_client(primary, fallback, settings=settings) as client:
        response = client.post(f"/v1/{wire_api}", json=payload)
    assert response.status_code == 200 and "fallback worked" in response.text
    expected = ReasoningPolicy(
        effort=ReasoningEffort(client_effort) if client_effort is not None else None
    )
    assert primary.reasoning_policies == [expected]
    assert fallback.reasoning_policies == [expected]


@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize("root_effort", ["high", "off"])
@pytest.mark.parametrize("fallback_effort", ["inherit", "client"])
def test_from_client_bypasses_fcc_policy_while_inherit_keeps_it(
    wire_api, root_effort, fallback_effort
):
    primary = ControlledFallbackProvider(failure=execution_failure("primary failed"))
    fallback = ControlledFallbackProvider(text="fallback worked")
    settings = Settings(
        MODEL="nvidia_nim/primary-model",
        REASONING_POLICY=root_effort,
        MODEL_FALLBACKS=[
            {"model": "groq/fallback-model", "reasoning": fallback_effort}
        ],
    )
    if wire_api == "messages":
        payload = messages_payload(stream=True) | {
            "thinking": {"type": "enabled", "budget_tokens": 6000},
            "output_config": {"effort": "low"},
        }
        client_policy = ReasoningPolicy.on(
            effort=ReasoningEffort.LOW, budget_tokens=6000
        )
    else:
        payload = responses_payload() | {
            "reasoning": {"effort": "low", "summary": "detailed"}
        }
        client_policy = ReasoningPolicy(
            control=ReasoningControl.DEFAULT, effort=ReasoningEffort.LOW
        )
    configured = (
        ReasoningPolicy.off()
        if root_effort == "off"
        else ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    )
    with fallback_client(primary, fallback, settings=settings) as client:
        response = client.post(f"/v1/{wire_api}", json=payload)
    assert response.status_code == 200 and "fallback worked" in response.text
    assert primary.reasoning_policies == [configured]
    assert fallback.reasoning_policies == [
        client_policy if fallback_effort == "client" else configured
    ]
    first, second = primary.requests[0], fallback.requests[0]
    if isinstance(first, MessagesRequest):
        assert isinstance(second, MessagesRequest)
        assert (
            second.thinking == first.thinking
            and second.output_config == first.output_config
        )
    else:
        assert (
            isinstance(second, OpenAIResponsesRequest)
            and second.reasoning == first.reasoning
        )


@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize(
    "effort", ["inherit", "off", "low", "medium", "high", "xhigh", "max"]
)
def test_fallback_has_its_own_policy_and_replaces_client_controls(wire_api, effort):
    primary = ControlledFallbackProvider(failure=execution_failure("primary failed"))
    fallback = ControlledFallbackProvider(text="fallback worked")
    settings = Settings(
        MODEL="nvidia_nim/primary-model",
        MODEL_FALLBACKS=[{"model": "groq/fallback-model", "reasoning": effort}],
    )
    if wire_api == "messages":
        payload = messages_payload(stream=True) | {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 6000,
                "display": "omitted",
            },
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": {"type": "object"}},
            },
        }
        original = ReasoningPolicy.on(effort=ReasoningEffort.HIGH, budget_tokens=6000)
    else:
        payload = responses_payload() | {
            "reasoning": {
                "effort": "high",
                "summary": "detailed",
                "context": "all_turns",
            }
        }
        original = ReasoningPolicy(
            control=ReasoningControl.DEFAULT, effort=ReasoningEffort.HIGH
        )
    with fallback_client(primary, fallback, settings=settings) as client:
        response = client.post(f"/v1/{wire_api}", json=payload)
    assert response.status_code == 200 and "fallback worked" in response.text
    assert primary.reasoning_policies == [original]
    expected = (
        original
        if effort == "inherit"
        else ReasoningPolicy.off()
        if effort == "off"
        else ReasoningPolicy.on(effort=ReasoningEffort(effort))
    )
    assert fallback.reasoning_policies == [expected]
    first, second = primary.requests[0], fallback.requests[0]
    assert first is not second and second.model == "fallback-model"
    if isinstance(first, MessagesRequest):
        assert isinstance(second, MessagesRequest)
        assert first.thinking is not None and first.thinking.budget_tokens == 6000
        assert (
            first.output_config is not None and first.output_config["effort"] == "high"
        )
        assert second.messages == first.messages and second.output_config is not None
        assert second.output_config["format"] == first.output_config["format"]
        if effort == "inherit":
            assert (
                second.thinking == first.thinking
                and second.output_config == first.output_config
            )
        else:
            assert second.thinking is not None and second.thinking.budget_tokens is None
            assert second.thinking.type is None and second.thinking.display == "omitted"
            assert "effort" not in second.output_config
    else:
        assert isinstance(second, OpenAIResponsesRequest)
        assert first.reasoning == payload["reasoning"]
        assert second.input == first.input
        assert (
            second.reasoning == first.reasoning
            if effort == "inherit"
            else second.reasoning == {"summary": "detailed", "context": "all_turns"}
        )


@pytest.mark.parametrize("wire_api", ["messages", "responses"])
def test_later_inherit_returns_to_original_policy_after_overridden_fallback(wire_api):
    primary = ControlledFallbackProvider(failure=execution_failure("primary failed"))
    low = ControlledFallbackProvider(failure=execution_failure("low failed"))
    inherited = ControlledFallbackProvider(text="original intent restored")
    settings = Settings(
        MODEL="nvidia_nim/primary-model",
        MODEL_FALLBACKS=[
            {"model": "groq/fallback-model", "reasoning": "low"},
            {"model": "open_router/later", "reasoning": "inherit"},
        ],
    )
    payload = (
        messages_payload(stream=True)
        | {
            "thinking": {"type": "enabled", "budget_tokens": 6000},
            "output_config": {"effort": "high"},
        }
        if wire_api == "messages"
        else responses_payload() | {"reasoning": {"effort": "high"}}
    )
    with fallback_client(
        primary, low, settings=settings, extra_providers={"open_router": inherited}
    ) as client:
        response = client.post(f"/v1/{wire_api}", json=payload)
    assert response.status_code == 200 and "original intent restored" in response.text
    assert low.reasoning_policies == [ReasoningPolicy.on(effort=ReasoningEffort.LOW)]
    assert inherited.reasoning_policies == primary.reasoning_policies
    last = inherited.requests[0]
    if isinstance(last, MessagesRequest):
        assert last.thinking is not None and last.thinking.budget_tokens == 6000
    else:
        assert last.reasoning == {"effort": "high"}
