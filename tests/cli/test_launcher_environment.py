"""Native state and process-local proxy settings survive shared launch setup."""

import os

import pytest

from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import HARNESSES, launch


@pytest.mark.parametrize("name", HARNESSES)
def test_native_state_and_unrelated_environment_are_preserved(
    name: str, launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "CODEX_HOME": "native-codex-home",
        "DSH_HOME": "native-dsh-home",
        "GROK_HOME": "native-grok-home",
        "HERMES_HOME": "native-hermes-home",
        "MUSE_HOME": "native-muse-home",
        "FCC_UNRELATED": "keep",
        "HTTP_PROXY": "http://machine-proxy.invalid:8080",
        "NO_PROXY": "internal.example",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    parent = dict(os.environ)
    launch(name, [])
    child = launch_capture.environments[0]
    for key, value in inherited.items():
        if key != "NO_PROXY":
            assert child[key] == value
    assert "internal.example" in child["NO_PROXY"].split(",")
    assert "127.0.0.1" in child["NO_PROXY"].split(",")
    assert child["NO_PROXY"] == child["no_proxy"]
    assert dict(os.environ) == parent
