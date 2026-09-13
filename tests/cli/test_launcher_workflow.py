"""Regression tests for shared startup and native command ownership."""

import importlib
import json
import tomllib
from pathlib import Path
from urllib.error import URLError

import pytest

from free_claude_code.cli.launchers.aider_config import build_aider_config
from free_claude_code.config.paths import codex_model_catalog_path
from free_claude_code.harnesses.model_catalog import ClientModel
from tests.cli.conftest import LaunchCapture

HARNESSES = (
    "claude",
    "codex",
    "pi",
    "opencode",
    "cline",
    "hermes",
    "dsh",
    "grok",
    "muse",
    "aider",
)


def launch(name: str, args: list[str], exit_code: int = 23) -> None:
    module = importlib.import_module(f"free_claude_code.cli.launchers.{name}")
    with pytest.raises(SystemExit) as exc:
        module.launch(args)
    assert exc.value.code == exit_code


@pytest.mark.parametrize("name", HARNESSES)
def test_help_receives_normal_fcc_setup(
    name: str, launch_capture: LaunchCapture
) -> None:
    launch(name, ["--help"])
    assert launch_capture.requests[0].full_url == "http://127.0.0.1:8182/health"
    assert launch_capture.commands[0][-1] == "--help"
    if name in {"claude", "pi"}:
        assert len(launch_capture.requests) == 1
    else:
        view = "messages" if name == "aider" else "responses"
        assert [request.full_url for request in launch_capture.requests] == [
            "http://127.0.0.1:8182/health",
            f"http://127.0.0.1:8182/v1/models?view={view}",
        ]


@pytest.mark.parametrize("name", HARNESSES)
def test_help_does_not_bypass_failed_setup(
    name: str, launch_capture: LaunchCapture
) -> None:
    launch_capture.health_error = URLError("test server is stopped")
    launch(name, ["--help"], exit_code=1)
    assert not launch_capture.commands


@pytest.mark.parametrize("name", HARNESSES)
def test_native_inputs_are_forwarded_without_fcc_model_or_command_policy(
    name: str, launch_capture: LaunchCapture
) -> None:
    args = [
        "--model",
        "future/model",
        "--model=another/model",
        "--future-option",
        "resume",
        "--",
        "--help",
        "--provider",
        "literal text",
    ]
    launch(name, args)
    assert launch_capture.commands[0][-len(args) :] == args


@pytest.mark.parametrize(
    "failure", [URLError("catalog unavailable"), ValueError("invalid catalog"), None]
)
def test_codex_required_catalog_failure_stops_launch(
    failure: Exception | None, launch_capture: LaunchCapture
) -> None:
    launch_capture.catalog_error = failure
    launch_capture.catalog = {"data": []}
    launch("codex", [], exit_code=1)
    assert not launch_capture.commands


def test_codex_cli_catalog_is_private_and_does_not_replace_external_catalog(
    launch_capture: LaunchCapture,
) -> None:
    stable = codex_model_catalog_path()
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_text('{"external":"unchanged"}')
    private_paths: list[Path] = []

    def inspect(command, env):
        settings = tomllib.loads(
            "\n".join(
                command[index + 1] for index, arg in enumerate(command) if arg == "-c"
            )
        )
        path = Path(settings["model_catalog_json"])
        assert path != stable
        assert (
            json.loads(path.read_text())["models"][0]["slug"]
            == "nvidia_nim/catalog-model:variant"
        )
        assert stable.read_text() == '{"external":"unchanged"}'
        private_paths.append(path)

    launch_capture.on_start = inspect
    launch("codex", [])
    launch("codex", [])
    assert private_paths[0] != private_paths[1]
    assert all(not path.exists() for path in private_paths)


def test_aider_native_settings_resolve_bare_and_transport_names_without_rewriting_args() -> (
    None
):
    wire_name = "nvidia_nim/catalog-model:variant"
    config = build_aider_config(
        (ClientModel(wire_name, wire_name, "Model", False),),
        messages_url="http://localhost:8182/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_TEST123",
        launch_id="launch-a",
    )
    entries = {entry["name"]: entry for entry in config.settings}
    for name in (wire_name, f"anthropic/{wire_name}"):
        entry = entries[name]
        extra_params = entry["extra_params"]
        assert isinstance(extra_params, dict)
        assert extra_params["model"] == f"anthropic/{wire_name}"
        assert entry["weak_model_name"] == name
        assert entry["editor_model_name"] == name
        assert name in config.metadata
