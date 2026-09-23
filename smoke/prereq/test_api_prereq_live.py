from typing import Any

import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.http import post_json
from smoke.lib.server import RunningServer

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("api")]


def test_probe_and_models_routes(
    smoke_server: RunningServer, smoke_headers: dict[str, str]
) -> None:
    with httpx.Client(base_url=smoke_server.base_url, headers=smoke_headers) as client:
        assert client.get("/health").json()["status"] == "healthy"

        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["status"] == "ok"

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"]

        for path in (
            "/",
            "/health",
            "/v1/messages",
            "/v1/responses",
            "/v1/messages/count_tokens",
        ):
            head = client.head(path)
            assert head.status_code == 204, (path, head.status_code, head.text)
            options = client.options(path)
            assert options.status_code == 204, (path, options.status_code, options.text)


def test_count_tokens_accepts_thinking_tools_and_results(
    smoke_server: RunningServer,
    smoke_config: SmokeConfig,
    smoke_headers: dict[str, str],
) -> None:
    payload: dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "Use the tool."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect the file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_smoke",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_smoke",
                        "content": "Free Claude Code",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
    }
    response = post_json(
        smoke_server,
        "/v1/messages/count_tokens",
        payload,
        smoke_config,
        headers=smoke_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0


def test_optimization_fast_paths_do_not_need_provider(
    smoke_server: RunningServer,
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
                        "content": "<policy_spec>extract command</policy_spec>\nCommand: git status --short",
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
                        "content": "Command: cat smoke/test_api_live.py\nOutput: file contents\n<filepaths>",
                    }
                ],
            },
            "smoke/test_api_live.py",
        ),
    )
    for name, payload, expected_text in cases:
        response = post_json(
            smoke_server, "/v1/messages", payload, smoke_config, headers=smoke_headers
        )
        assert response.status_code == 200, (name, response.text)
        text = response.json()["content"][0]["text"]
        assert expected_text in text


def test_invalid_messages_returns_anthropic_error(
    smoke_server: RunningServer,
    smoke_config: SmokeConfig,
    smoke_headers: dict[str, str],
) -> None:
    response = post_json(
        smoke_server,
        "/v1/messages",
        {"model": "claude-3-5-sonnet-20241022", "messages": []},
        smoke_config,
        headers=smoke_headers,
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"


def test_stop_endpoint_reports_no_messaging(
    smoke_server: RunningServer, smoke_headers: dict[str, str]
) -> None:
    response = httpx.post(
        f"{smoke_server.base_url}/stop",
        headers=smoke_headers,
        timeout=5,
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Messaging system not initialized"
