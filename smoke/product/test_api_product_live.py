from typing import Any

import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import (
    ConversationDriver,
    ProviderMatrixDriver,
    SmokeServerDriver,
    assert_product_stream,
    echo_tool_schema,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("api")]


def test_api_basic_conversation_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-api-basic",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).ask(
            "Reply with one short sentence."
        )

    assert_product_stream(turn.events)
    assert turn.text.strip()


def test_api_count_tokens_full_payload_e2e(
    smoke_server,
    smoke_config: SmokeConfig,
    smoke_headers: dict[str, str],
) -> None:
    payload: dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Use the image and tool."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need the tool."},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {
                        "type": "tool_use",
                        "id": "toolu_smoke",
                        "name": "echo_smoke",
                        "input": {"value": "FCC"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_smoke",
                        "content": "FCC",
                    }
                ],
            },
        ],
        "tools": [echo_tool_schema()],
        "thinking": {"type": "adaptive", "budget_tokens": 1024},
    }
    response = httpx.post(
        f"{smoke_server.base_url}/v1/messages/count_tokens",
        headers=smoke_headers,
        json=payload,
        timeout=smoke_config.timeout_s,
    )
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0


def test_api_request_optimizations_e2e(
    smoke_server,
    smoke_config: SmokeConfig,
    smoke_headers: dict[str, str],
) -> None:
    cases: tuple[tuple[str, dict[str, Any], str], ...] = (
        (
            "quota",
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "quota"}],
            },
            "Quota check passed.",
        ),
        (
            "title",
            {
                "model": "claude-3-5-sonnet-20241022",
                "system": (
                    "Generate a concise, sentence-case title (3-7 words). "
                    'Return JSON with a single "title" field.'
                ),
                "messages": [{"role": "user", "content": "hello"}],
            },
            "Conversation",
        ),
        (
            "prefix",
            {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<policy_spec>extract command</policy_spec>\n"
                            "Command: git status --short"
                        ),
                    }
                ],
            },
            "git",
        ),
        (
            "suggestion",
            {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "[SUGGESTION MODE: next]"}],
            },
            "",
        ),
        (
            "filepath",
            {
                "model": "claude-3-5-sonnet-20241022",
                "system": "Extract any file paths that this command output contains.",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Command: cat smoke/product/test_api_product_live.py\n"
                            "Output: file contents\n<filepaths>"
                        ),
                    }
                ],
            },
            "smoke/product/test_api_product_live.py",
        ),
    )
    for name, payload, expected_text in cases:
        response = httpx.post(
            f"{smoke_server.base_url}/v1/messages",
            headers=smoke_headers,
            json=payload,
            timeout=smoke_config.timeout_s,
        )
        assert response.status_code == 200, (name, response.text)
        text = response.json()["content"][0]["text"]
        assert expected_text in text


def test_api_error_shape_e2e(
    smoke_server,
    smoke_config: SmokeConfig,
    smoke_headers: dict[str, str],
) -> None:
    response = httpx.post(
        f"{smoke_server.base_url}/v1/messages",
        headers=smoke_headers,
        json={"model": "claude-3-5-sonnet-20241022", "messages": []},
        timeout=smoke_config.timeout_s,
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"


def test_api_stop_e2e(smoke_server, smoke_headers: dict[str, str]) -> None:
    response = httpx.post(
        f"{smoke_server.base_url}/stop",
        headers=smoke_headers,
        timeout=5,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Messaging system not initialized"
