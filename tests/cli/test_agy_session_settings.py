"""FCC configures Antigravity for the child session without a saved connection."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from free_claude_code.harnesses import antigravity
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch
from tests.wire_payload import decode_object, object_payload, string_payload


def test_fresh_launch_uses_temporary_settings(launch_capture: LaunchCapture) -> None:
    path = antigravity.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"modelProvider":"", "model":"personal"}')
    before = path.read_bytes()
    launch_capture.versions["agy"] = "1.5.9"
    files = []

    def inspect(command: list[str], env: Mapping[str, str]):
        path = Path(command[command.index("--settings") + 1])
        files.append(path)
        settings = decode_object(path.read_text())
        assert settings["modelProvider"] == "gemini"
        model = string_payload(launch_capture.catalog["default_model_id"])
        assert object_payload(
            object_payload(
                object_payload(settings["customModelsConfig"])["customModels"]
            )[model]
        )["modelName"] == ("gemini-api://models/" + string_payload(model))
        assert env["GEMINI_API_KEY"] not in path.read_text()
        assert antigravity.config_path().read_bytes() == before
        assert not antigravity.state_path().exists()

    launch_capture.on_start = inspect
    launch("agy", ["--print", "hello"])
    assert files and all(not path.exists() for path in files)


@pytest.mark.parametrize(
    "args",
    [
        ["--settings", "{}"],
        ["--settings={}"],
        ["-settings", "{}"],
        ["-settings={}"],
        ["--settings", "my session settings.json"],
        ["--settings", "{}", "--settings", "another.json"],
        ["--model", "print", "--settings", "{}"],
        ["--log-file", "--print", "--settings", "{}", "--print", "hello"],
        ["--print", "--settings"],
        ["--print=--settings"],
    ],
)
def test_user_arguments_follow_fcc_defaults_without_interpretation(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch("agy", args)
    command = launch_capture.commands[0]
    assert command[:2] == ["agy", "--settings"]
    assert command[3:5] == ["--model", launch_capture.catalog["default_model_id"]]
    assert command[5:] == args
