from typing import Any

import httpx
import pytest

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.stream_contracts import (
    SSEEvent,
    parse_sse_lines,
)
from smoke.lib.config import ProviderModel, SmokeConfig, auth_headers
from smoke.lib.e2e import (
    ConversationDriver,
    ProviderMatrixDriver,
    SmokeServerDriver,
    assert_native_thinking_stream,
    assert_product_stream,
    echo_tool_schema,
    tool_use_blocks,
)
from smoke.lib.http import collect_message_stream
from smoke.lib.skips import (
    skip_if_upstream_unavailable_events,
    skip_if_upstream_unavailable_exception,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("providers")]


def test_provider_matrix_presence_e2e(smoke_config: SmokeConfig) -> None:
    models = ProviderMatrixDriver(smoke_config).provider_smoke_models()
    assert models or smoke_config.provider_matrix == frozenset()


def test_model_mapping_matrix_e2e(smoke_config: SmokeConfig) -> None:
    models = ProviderMatrixDriver(smoke_config).configured_models()
    sources = {model.source for model in models}
    assert sources <= {
        "MODEL",
        "MODEL_FABLE",
        "MODEL_OPUS",
        "MODEL_SONNET",
        "MODEL_HAIKU",
    }
    for model in models:
        assert model.provider
        assert model.model_name


def test_provider_text_multiturn_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    _run_provider_scenario(smoke_config, provider_model, _scenario_text_multiturn)


def test_provider_adaptive_thinking_history_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    _run_provider_scenario(
        smoke_config, provider_model, _scenario_adaptive_thinking_history
    )


def test_provider_interleaved_thinking_tool_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    _run_provider_scenario(smoke_config, provider_model, _scenario_interleaved_history)


@pytest.mark.smoke_target("tools")
def test_provider_tool_use_then_text_history_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    """OpenAI-compatible path: history with tool_use + assistant text after tool (issue #206)."""
    _run_provider_scenario(
        smoke_config, provider_model, _scenario_tool_use_then_text_in_history
    )


@pytest.mark.smoke_target("tools")
def test_provider_tool_result_continuation_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    _run_provider_scenario(
        smoke_config, provider_model, _scenario_tool_result_continuation
    )


@pytest.mark.smoke_target("tools")
def test_provider_interrupted_tool_turn_resume_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    _run_provider_scenario(
        smoke_config, provider_model, _scenario_interrupted_tool_turn_resume
    )


@pytest.mark.smoke_target("tools")
def test_gemini_thought_signature_tool_continuation_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    if provider_model.provider != "gemini":
        pytest.skip("gemini-specific smoke scenario")
    _run_provider_scenario(
        smoke_config,
        provider_model,
        _scenario_gemini_thought_signature_tool_continuation,
    )


@pytest.mark.smoke_target("tools")
def test_provider_reasoning_tool_continuation_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    if not _provider_smoke_reasoning_enabled(smoke_config):
        pytest.skip("the configured Claude route disables reasoning")
    _run_provider_scenario(
        smoke_config, provider_model, _scenario_reasoning_tool_continuation
    )


def test_mistral_native_reasoning_model_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = smoke_config.mistral_reasoning_smoke_model()
    if provider_model is None:
        pytest.skip("missing_env: mistral is not configured")

    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Reply with one short sentence."}],
        "thinking": {"type": "adaptive"},
    }
    try:
        with _server_for_provider(
            smoke_config, provider_model, "mistral-native-reasoning"
        ) as server:
            events = collect_message_stream(server, payload, smoke_config)
    except Exception as exc:
        skip_if_upstream_unavailable_exception(exc)
        raise

    assert_native_thinking_stream(
        events,
        context=f"{provider_model.source}={provider_model.full_model}",
    )


@pytest.mark.smoke_target("rate_limit")
def test_provider_disconnect_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    _run_provider_scenario(smoke_config, provider_model, _scenario_disconnect)


def test_provider_error_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    broken_model = f"{provider_model.provider}/fcc-smoke-missing-model"
    with (
        SmokeServerDriver(
            smoke_config,
            name=f"product-provider-error-{provider_model.provider}",
            env_overrides={"MODEL": broken_model, "MESSAGING_PLATFORM": "none"},
        ).run() as server,
        httpx.Client(timeout=smoke_config.timeout_s) as client,
    ):
        response = client.post(
            f"{server.base_url}/v1/messages",
            headers=auth_headers(),
            json={
                "model": "fcc-smoke-default",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code >= 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"]
    assert payload["error"]["message"]
    assert payload["request_id"] == response.headers["request-id"]


def test_provider_codex_responses_text_e2e(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    try:
        with (
            _server_for_provider(
                smoke_config, provider_model, "codex-responses"
            ) as server,
            httpx.stream(
                "POST",
                f"{server.base_url}/v1/responses",
                headers=_openai_auth_headers(smoke_config),
                json={
                    "model": provider_model.full_model,
                    "input": smoke_config.prompt,
                    "max_output_tokens": 1024,
                    "stream": True,
                },
                timeout=smoke_config.timeout_s,
            ) as response,
        ):
            assert response.status_code == 200, response.read()
            events = parse_sse_lines(response.iter_lines())
    except Exception as exc:
        skip_if_upstream_unavailable_exception(exc)
        raise

    skip_if_upstream_unavailable_events(events)
    names = [event.event for event in events]
    assert names[0] == "response.created", names
    assert names[-1] == "response.completed", names
    assert any(event.event == "response.output_text.delta" for event in events), names


def test_openrouter_native_e2e(smoke_config: SmokeConfig) -> None:
    models = [
        model
        for model in ProviderMatrixDriver(smoke_config).provider_smoke_models()
        if model.provider == "open_router"
    ]
    if not models:
        pytest.skip("missing_env: open_router is not configured")

    provider_model = models[0]
    with SmokeServerDriver(
        smoke_config,
        name="product-openrouter-native",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(
            {
                "model": "claude-opus-4-7",
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with one short sentence.",
                    }
                ],
                "thinking": {"type": "adaptive", "budget_tokens": 1024},
            }
        )
    _assert_provider_product_stream(turn.events)


def _run_provider_scenario(
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    scenario,
) -> None:
    try:
        scenario(smoke_config, provider_model)
    except Exception as exc:
        skip_if_upstream_unavailable_exception(exc)
        raise AssertionError(
            f"{provider_model.source}={provider_model.full_model}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _assert_provider_product_stream(events: list[SSEEvent]) -> None:
    skip_if_upstream_unavailable_events(events)
    assert_product_stream(events)


def _tool_use_blocks_or_skip(
    events: list[SSEEvent], message: str
) -> list[dict[str, Any]]:
    skip_if_upstream_unavailable_events(events)
    blocks = tool_use_blocks(events)
    assert blocks, message
    return blocks


def _provider_smoke_reasoning_enabled(smoke_config: SmokeConfig) -> bool:
    return (
        ModelRouter(smoke_config.settings)
        .resolve("claude-sonnet-4-5-20250929")
        .reasoning_preference
        is not ReasoningPreference.OFF
    )


def _scenario_text_multiturn(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    with _server_for_provider(smoke_config, provider_model, "text") as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.ask("Reply with one short sentence.")
        second = driver.ask("Reply with a different short sentence.")
    _assert_provider_product_stream(first.events)
    _assert_provider_product_stream(second.events)


def _scenario_adaptive_thinking_history(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "unsigned hidden thought"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "Hello."},
                ],
            },
            {"role": "user", "content": "Reply with one short sentence."},
        ],
        "thinking": {"type": "adaptive", "budget_tokens": 1024},
    }
    with _server_for_provider(smoke_config, provider_model, "adaptive") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    _assert_provider_product_stream(turn.events)


def _scenario_interleaved_history(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Use the tool."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect first."},
                    {"type": "text", "text": "I will call the tool."},
                    {
                        "type": "tool_use",
                        "id": "toolu_interleaved",
                        "name": "echo_smoke",
                        "input": {"value": "FCC_INTERLEAVED"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_interleaved",
                        "content": "FCC_INTERLEAVED",
                    }
                ],
            },
        ],
        "tools": [echo_tool_schema()],
        "thinking": {"type": "adaptive"},
    }
    with _server_for_provider(smoke_config, provider_model, "interleaved") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    _assert_provider_product_stream(turn.events)


def _scenario_tool_use_then_text_in_history(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    tool_id = "toolu_206_smoke"
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "We will use echo_smoke once in this session."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "echo_smoke",
                        "input": {"value": "FCC_206_SMOKE"},
                    },
                    {
                        "type": "text",
                        "text": "Narration after the tool call (issue #206 shape).",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "FCC_206_SMOKE",
                    },
                ],
            },
            {
                "role": "user",
                "content": "Reply in one short sentence: did you see the echo value?",
            },
        ],
        "tools": [echo_tool_schema()],
    }
    with _server_for_provider(smoke_config, provider_model, "tool-206") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    _assert_provider_product_stream(turn.events)


def _scenario_tool_result_continuation(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    first_payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Use echo_smoke once with value FCC_TOOL."}
        ],
        "tools": [echo_tool_schema()],
        "tool_choice": {"type": "tool", "name": "echo_smoke"},
        "thinking": {"type": "adaptive"},
    }
    with _server_for_provider(smoke_config, provider_model, "tool") as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.stream(first_payload)
        tool_uses = _tool_use_blocks_or_skip(
            first.events, "provider did not emit a tool_use block"
        )
        tool_use = tool_uses[0]
        second_payload = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 256,
            "messages": [
                first_payload["messages"][0],
                {"role": "assistant", "content": first.assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "FCC_TOOL",
                        }
                    ],
                },
            ],
            "tools": [echo_tool_schema()],
        }
        second = driver.stream(second_payload)
    _assert_provider_product_stream(first.events)
    _assert_provider_product_stream(second.events)


def _scenario_interrupted_tool_turn_resume(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    tool_id = "toolu_interrupted123456789"
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Use echo_smoke with value FCC_INTERRUPTED."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "echo_smoke",
                        "input": {"value": "FCC_INTERRUPTED"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "FCC_INTERRUPTED",
                    }
                ],
            },
            {
                "role": "user",
                "content": "Reply exactly OK without calling tools.",
            },
        ],
        "tools": [echo_tool_schema()],
    }
    with _server_for_provider(
        smoke_config, provider_model, "interrupted-tool-turn"
    ) as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    _assert_provider_product_stream(turn.events)


def _scenario_gemini_thought_signature_tool_continuation(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    first_payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Use echo_smoke once with value FCC_TOOL."}
        ],
        "tools": [echo_tool_schema()],
        "tool_choice": {"type": "tool", "name": "echo_smoke"},
        "thinking": {"type": "adaptive", "budget_tokens": 1024},
    }
    with _server_for_provider(
        smoke_config, provider_model, "gemini-signature"
    ) as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.stream(first_payload)
        tool_uses = _tool_use_blocks_or_skip(
            first.events, "gemini did not emit a tool_use block"
        )
        tool_use = tool_uses[0]
        signature = _gemini_tool_thought_signature(tool_use)
        assert signature, (
            "gemini tool_use did not preserve extra_content.google.thought_signature"
        )
        second_payload = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 256,
            "messages": [
                first_payload["messages"][0],
                {"role": "assistant", "content": first.assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "FCC_TOOL",
                        }
                    ],
                },
            ],
            "tools": [echo_tool_schema()],
            "thinking": {"type": "adaptive", "budget_tokens": 1024},
        }
        second = driver.stream(second_payload)
    _assert_provider_product_stream(first.events)
    _assert_provider_product_stream(second.events)


def _gemini_tool_thought_signature(tool_use: dict[str, Any]) -> str | None:
    extra_content = tool_use.get("extra_content")
    if not isinstance(extra_content, dict):
        return None
    google = extra_content.get("google")
    if not isinstance(google, dict):
        return None
    signature = google.get("thought_signature")
    return signature if isinstance(signature, str) and signature else None


def _scenario_reasoning_tool_continuation(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Use echo_smoke once with value FCC_TOOL."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to return the echo result."},
                    {
                        "type": "tool_use",
                        "id": "toolu_reasoning_smoke",
                        "name": "echo_smoke",
                        "input": {"value": "FCC_TOOL"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_reasoning_smoke",
                        "content": "FCC_TOOL",
                    }
                ],
            },
        ],
        "tools": [echo_tool_schema()],
        "thinking": {"type": "adaptive"},
    }
    with _server_for_provider(smoke_config, provider_model, "reasoning-tool") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    _assert_provider_product_stream(turn.events)


def _scenario_disconnect(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    with _server_for_provider(smoke_config, provider_model, "disconnect") as server:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/messages",
            headers=auth_headers(),
            json={
                "model": "fcc-smoke-default",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": smoke_config.prompt}],
            },
            timeout=smoke_config.timeout_s,
        ) as response:
            assert response.status_code == 200, response.read()
            for _line in response.iter_lines():
                break
        health = httpx.get(f"{server.base_url}/health", timeout=5)
        assert health.status_code == 200
        followup = ConversationDriver(server, smoke_config).ask(
            "Reply with one short sentence."
        )
    _assert_provider_product_stream(followup.events)


def _server_for_provider(
    smoke_config: SmokeConfig, provider_model: ProviderModel, name: str
):
    return SmokeServerDriver(
        smoke_config,
        name=f"product-provider-{provider_model.provider}-{name}",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run()


def _openai_auth_headers(smoke_config: SmokeConfig) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    token = smoke_config.settings.proxy_auth_token
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers
