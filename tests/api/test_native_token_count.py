"""Native histories remain countable without contacting their provider."""

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import get_token_count
from free_claude_code.core.anthropic.tokens import CountMessages
from free_claude_code.core.json_types import JsonObject
from tests.api.support import create_test_app, provider_manager_for_app
from tests.wire_payload import (
    object_array,
    object_payload,
)


def count_body() -> JsonObject:
    return {
        "model": "claude-sonnet-client-alias",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "bash_code_execution_tool_result",
                        "tool_use_id": "srvtoolu_test",
                        "content": {
                            "type": "bash_code_execution_result",
                            "stdout": "hello",
                            "stderr": "",
                            "return_code": 0,
                            "content": [],
                        },
                    },
                    {"type": "future_result", "opaque": {"keep": [1, 2]}},
                ],
            }
        ],
        "system": [{"type": "future_system", "instructions": "be helpful"}],
        "tools": [{"type": "web_search_future", "name": "web_search", "max_uses": 2}],
        "thinking": {"type": "future_thinking", "option": True},
    }


@pytest.mark.parametrize("field", ["messages", "system", "tools"])
def test_native_count_accepts_and_estimates_opaque_payload_without_provider(
    field: str,
) -> None:
    app = create_test_app(Settings(MODEL="anthropic/selected"))
    body = count_body()
    expanded = deepcopy(body)
    extra = "a substantially longer native payload " * 100
    if field == "messages":
        message = object_array(expanded[field])[0]
        opaque = object_array(message["content"])[1]["opaque"]
        object_payload(opaque)["extra"] = extra
    else:
        object_array(expanded[field])[0]["extra"] = extra
    original = deepcopy(body)
    with (
        patch(
            "free_claude_code.runtime.provider_manager.ProviderGenerationLease.resolve_provider",
            side_effect=AssertionError("Counting must not resolve a provider"),
        ),
        TestClient(app) as client,
    ):
        baseline = client.post("/v1/messages/count_tokens", json=body)
        larger = client.post("/v1/messages/count_tokens", json=expanded)
    assert baseline.status_code == 200, baseline.text
    assert larger.status_code == 200, larger.text
    assert larger.json()["input_tokens"] > baseline.json()["input_tokens"] > 0
    assert body == original


def test_compatibility_primary_counts_future_blocks_and_unnamed_tools_with_native_fallback() -> (
    None
):
    app = create_test_app(
        Settings(
            MODEL="nvidia_nim/selected",
            MODEL_FALLBACKS=[ModelFallback(model="anthropic/backup")],
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={**count_body(), "tools": [{"type": "future_hosted"}]},
        )
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0


@pytest.mark.parametrize(
    ("update", "status"),
    [
        ({"messages": []}, 400),
        ({"messages": "invalid"}, 422),
        ({"messages": [{"role": "assistant", "content": [42]}]}, 422),
        ({"system": 42}, 422),
        ({"tools": [42]}, 422),
        ({"model": 42}, 422),
        ({"model": " "}, 400),
    ],
)
def test_native_count_rejects_bad_shapes_and_releases_lease(
    update: JsonObject, status: int
) -> None:
    app = create_test_app(Settings(MODEL="anthropic/selected"))
    manager = provider_manager_for_app(app)
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages/count_tokens", json={**count_body(), **update}
        )
    assert response.status_code == status, response.text
    assert manager._current.active_leases == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "\ud800"])
def test_native_count_rejects_non_finite_or_non_utf8_json(value: float | str) -> None:
    app = create_test_app(Settings(MODEL="anthropic/selected"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages/count_tokens",
            content=json.dumps({**count_body(), "extension": value}),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert provider_manager_for_app(app)._current.active_leases == 0


def test_native_count_proxy_auth_and_local_estimator_failure_release() -> None:
    app = create_test_app(
        Settings(
            MODEL="anthropic/selected",
            PROXY_AUTH_ENABLED=True,
            ANTHROPIC_AUTH_TOKEN="proxy-secret",
        )
    )
    with TestClient(app) as client:
        denied = client.post("/v1/messages/count_tokens", json=count_body())
        accepted = client.post(
            "/v1/messages/count_tokens",
            json=count_body(),
            headers={"Authorization": "Bearer proxy-secret"},
        )
        with (
            patch(
                "free_claude_code.api.routes.get_token_count",
                side_effect=RuntimeError("private-estimate-token"),
            ),
            patch("free_claude_code.api.request_errors.logger.error") as error_log,
        ):
            failed = client.post(
                "/v1/messages/count_tokens",
                json=count_body(),
                headers={"Authorization": "Bearer proxy-secret"},
            )
    assert denied.status_code in {401, 403}
    assert accepted.status_code == 200, accepted.text
    assert failed.status_code == 500, failed.text
    assert "private-estimate-token" not in str(error_log.call_args_list)
    assert provider_manager_for_app(app)._current.active_leases == 0


def test_native_count_estimator_receives_payload_without_mutation_and_traces_alias() -> (
    None
):
    from free_claude_code.application.routing import ModelRouter

    body = count_body()
    object_array(body["tools"])[0]["api_key"] = "private-tool-option"
    original = deepcopy(body)
    captured = []
    resolutions = []
    resolve = ModelRouter.resolve

    def record_resolve(router: ModelRouter, model: str):
        resolutions.append(model)
        return resolve(router, model)

    def count(
        messages: CountMessages, system: list[JsonObject], tools: list[JsonObject]
    ):
        captured.append(
            (
                [message.model_dump() for message in messages],
                deepcopy(system),
                deepcopy(tools),
            )
        )
        before = deepcopy(captured[-1])
        result = get_token_count(messages, system, tools)
        assert ([message.model_dump() for message in messages], system, tools) == before
        return result

    app = create_test_app(Settings(MODEL="anthropic/selected"))
    with (
        patch.object(ModelRouter, "resolve", record_resolve),
        patch("free_claude_code.api.routes.get_token_count", count),
        patch("free_claude_code.api.handlers.token_count.trace_event") as trace,
        TestClient(app) as client,
    ):
        response = client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200, response.text
    assert resolutions == [body["model"]]
    assert captured == [(body["messages"], body["system"], body["tools"])]
    assert body == original
    route = next(
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs["stage"] == "routing"
    )
    assert route["provider_model_ref"] == "anthropic/selected"
    assert route["gateway_model"] == body["model"]
    render: Callable[[], JsonObject] = next(
        call.args[0] for call in trace.call_args_list if call.args
    )
    snapshot = object_payload(render()["snapshot"])
    assert object_array(snapshot["tools"])[0]["api_key"] == "<redacted>"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_native_count_retains_generation_while_waiting_for_encoder(
    cancel: bool,
) -> None:
    from free_claude_code.runtime.provider_manager import ProviderGenerationLease

    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def wait(lease: object):
        entered.set()
        await proceed.wait()

    async def commit():
        pass

    app = create_test_app(Settings(MODEL="anthropic/selected"))
    manager = provider_manager_for_app(app)
    old_generation = manager._current
    try:
        with patch.object(ProviderGenerationLease, "wait_for_token_estimation", wait):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(
                    client.post("/v1/messages/count_tokens", json=count_body())
                )
                await asyncio.wait_for(entered.wait(), 2)
                await manager.replace(Settings(MODEL="nvidia_nim/other"), commit=commit)
                assert old_generation.active_leases == 1
                if cancel:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    proceed.set()
                    response = await asyncio.wait_for(task, 2)
                    assert response.status_code == 200, response.text
        assert old_generation.active_leases == 0
        assert manager._current.active_leases == 0
    finally:
        proceed.set()
        await manager.close()
