"""Observe configuration while children run and after launch scopes unwind."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from free_claude_code.cli.launchers import common
from free_claude_code.config.paths import launcher_temp_dir_path
from free_claude_code.core.json_types import JsonValue
from free_claude_code.harnesses import resources
from free_claude_code.harnesses.resources import LaunchResources
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch, launcher

FILE_HARNESSES = ("codex", "opencode", "cline", "hermes", "dsh", "aider")


@pytest.mark.parametrize("name", FILE_HARNESSES)
def test_private_files_live_until_child_exit(
    name: str, launch_capture: LaunchCapture
) -> None:
    paths: list[Path] = []

    def inspect(command: object, env: object):
        paths.extend(
            path for path in launcher_temp_dir_path().rglob("*") if path.is_file()
        )
        assert paths
        if os.name != "nt":
            assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
            assert all(path.parent.stat().st_mode & 0o777 == 0o700 for path in paths)

    launch_capture.on_start = inspect
    launch(name, [])
    assert all(not path.exists() for path in paths)
    assert list(launcher_temp_dir_path().iterdir()) == []


@pytest.mark.parametrize("name", FILE_HARNESSES)
@pytest.mark.parametrize(
    "failure", (PermissionError("cannot start"), KeyboardInterrupt())
)
def test_cleanup_on_process_start_failure(
    name: str, failure: BaseException, launch_capture: LaunchCapture
) -> None:
    def fail_start(command: object, env: object):
        assert any(launcher_temp_dir_path().rglob("*"))
        raise failure

    launch_capture.on_start = fail_start
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            launcher(name)([])
    else:
        launch(name, [], exit_code=1)
    assert list(launcher_temp_dir_path().iterdir()) == []


def test_cleanup_after_partial_configuration(launch_capture: LaunchCapture) -> None:
    original = resources.LaunchResources.write_json

    def fail_second(self: LaunchResources, filename: str, payload: JsonValue):
        if filename == "settings/models.json":
            raise PermissionError("failed second write")
        return original(self, filename, payload)

    with patch.object(resources.LaunchResources, "write_json", fail_second):
        launch("cline", [], exit_code=1)
    assert not launch_capture.commands
    assert list(launcher_temp_dir_path().iterdir()) == []


def test_cleanup_after_directory_creation_error(launch_capture: LaunchCapture) -> None:
    with patch.object(
        resources.tempfile,
        "TemporaryDirectory",
        side_effect=PermissionError("no directory"),
    ):
        launch("codex", [], exit_code=1)
    assert not launch_capture.commands


def test_nested_launches_keep_independent_live_catalogs(
    launch_capture: LaunchCapture,
) -> None:
    outer: list[Path] = []

    def nested(command: object, env: object):
        outer.extend(launcher_temp_dir_path().glob("*/model-catalog.json"))
        assert len(outer) == 1

        def inspect_inner(inner_command: object, inner_env: object):
            paths = list(launcher_temp_dir_path().glob("*/model-catalog.json"))
            assert len(paths) == 2
            assert outer[0].is_file()

        launch_capture.on_start = inspect_inner
        launch("codex", [])
        assert outer[0].is_file()

    launch_capture.on_start = nested
    launch("codex", [])
    assert not outer[0].exists()


def test_interrupt_kills_and_waits_for_registered_child_before_cleanup(
    launch_capture: LaunchCapture,
) -> None:
    process = Mock(pid=12345)
    process.wait.side_effect = (KeyboardInterrupt(), 0)
    events: list[str] = []

    def kill(pid: int):
        assert pid == 12345
        assert list(launcher_temp_dir_path().glob("*/model-catalog.json"))
        events.append("kill")

    def side_effect_2(pid: object):
        return events.append("register")

    def side_effect_3(pid: object):
        return events.append("unregister")

    with (
        patch.object(common.subprocess, "Popen", return_value=process),
        patch.object(common, "register_pid", side_effect=side_effect_2),
        patch.object(common, "kill_pid_tree_best_effort", side_effect=kill),
        patch.object(
            common,
            "unregister_pid",
            side_effect=side_effect_3,
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        launcher("codex")([])
    assert events == ["register", "kill", "unregister"]
    assert process.wait.call_count == 2
    assert list(launcher_temp_dir_path().iterdir()) == []


@pytest.mark.parametrize("name", ("claude", "pi", "grok", "muse"))
def test_environment_only_harnesses_allocate_no_directory(
    name: str, launch_capture: LaunchCapture
) -> None:
    launch(name, [])
    assert not launcher_temp_dir_path().exists()
