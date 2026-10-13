"""Grok's native configuration and connection-option placement."""

import json

import pytest

from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["grok", "--disable-web-search", "--no-leader"]),
        (
            ["-p", "hello"],
            ["grok", "--disable-web-search", "--no-leader", "-p", "hello"],
        ),
        (
            ["agent", "stdio"],
            ["grok", "--disable-web-search", "agent", "--no-leader", "stdio"],
        ),
        (
            ["--", "agent"],
            ["grok", "--disable-web-search", "--no-leader", "--", "agent"],
        ),
    ],
)
def test_grok_connection_options_use_native_scopes(
    args, expected, launch_capture: LaunchCapture
) -> None:
    launch("grok", args)
    assert launch_capture.commands == [expected]
    env = launch_capture.environments[0]
    assert env["GROK_XAI_API_BASE_URL"] == "http://127.0.0.1:8182/v1"
    assert (
        env["GROK_MODELS_LIST_URL"] == "http://127.0.0.1:8182/v1/models?view=responses"
    )
    assert env["XAI_API_KEY"] == "launcher-test-token"
    assert json.loads(env["GROK_CONFIG"])["models"]["allowed_models"] == [
        "nvidia_nim/catalog-model:variant"
    ]


@pytest.mark.parametrize("key", ("GROK_CONFIG", "GROK_CONFIG_PATH"))
def test_existing_grok_process_config_is_not_replaced(
    key: str, launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(key, "user-owned-config")
    launch("grok", [], exit_code=1)
    assert not launch_capture.commands
