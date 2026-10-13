"""Muse receives connection options in native parser scope."""

import pytest

from free_claude_code.cli.launchers.muse import SPEC
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["exec", "--model", "native-model", "hello"],
        ["resume", "session-id", "--model=native-model"],
        ["--", "exec", "--model"],
        ["literal exec text"],
    ],
)
def test_muse_connection_placement_keeps_native_model_options(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch("muse", args)
    connection = ["--provider", "meta", "--base-url", "http://127.0.0.1:8182/v1"]
    expected = (
        ["muse", args[0], *connection, *args[1:]]
        if args and args[0] in {"exec", "resume"}
        else ["muse", *connection, *args]
    )
    assert launch_capture.commands == [expected]
    env = launch_capture.environments[0]
    assert env["META_API_KEY"] == "launcher-test-token"
    assert env["MUSE_MODEL"] == "nvidia_nim/catalog-model:variant"


def test_muse_install_hint_covers_windows_and_posix() -> None:
    assert "install-muse.ps1" in SPEC.install_hint
    assert "https://dev.meta.ai/install.sh" in SPEC.install_hint
