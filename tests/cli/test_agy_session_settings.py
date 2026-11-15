"""FCC configures Antigravity for the child session without a saved connection."""

import json
from pathlib import Path

import pytest

from free_claude_code.harnesses import antigravity
from tests.cli.test_launcher_workflow import launch


def test_fresh_launch_uses_temporary_settings(launch_capture):
    path = antigravity.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"modelProvider":"", "model":"personal"}')
    before = path.read_bytes()
    launch_capture.versions["agy"] = "1.5.9"
    files = []

    def inspect(command, env):
        path = Path(command[command.index("--settings") + 1])
        files.append(path)
        settings = json.loads(path.read_text())
        assert settings["modelProvider"] == "gemini"
        model = launch_capture.catalog["default_model_id"]
        assert settings["customModelsConfig"]["customModels"][model]["modelName"] == (
            "gemini-api://models/" + model
        )
        assert env["GEMINI_API_KEY"] not in path.read_text()
        assert antigravity.config_path().read_bytes() == before
        assert not antigravity.state_path().exists()

    launch_capture.on_start = inspect
    launch("agy", ["--print", "hello"])
    assert files and all(not path.exists() for path in files)


@pytest.mark.parametrize(
    "arg", ["--settings", "--settings={}", "-settings", "-settings={}"]
)
def test_settings_are_reserved_before_any_server_work(arg, launch_capture, capsys):
    launch("agy", [arg, "{}"], exit_code=1)
    assert not launch_capture.requests and not launch_capture.probes
    assert not launch_capture.commands
    assert "settings" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag",
    [
        "--print",
        "-print",
        "--p",
        "-p",
        "--prompt",
        "-prompt",
        "--prompt-interactive",
        "-prompt-interactive",
        "--i",
        "-i",
    ],
)
def test_prompt_can_literally_name_the_reserved_option(flag, launch_capture):
    launch("agy", [flag, "--settings"])
    assert launch_capture.commands[0][-2:] == [flag, "--settings"]


@pytest.mark.parametrize("model", ["print", "prompt", "prompt-interactive", "p", "i"])
@pytest.mark.parametrize(
    "option", ["--settings", "--settings={}", "-settings", "-settings={}"]
)
def test_model_value_cannot_hide_reserved_settings(model, option, launch_capture):
    launch("agy", ["--model", model, option, "{}"], exit_code=1)
    assert not launch_capture.probes
    assert not launch_capture.requests
    assert not launch_capture.commands
