"""Native Aider configuration replaces FCC argument rewriting."""

import json
from pathlib import Path

import pytest

from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


def test_aider_receives_native_files_and_a_private_credential_reference(
    launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("fcc_aider_proxy_auth_stale", "stale-secret")
    monkeypatch.setenv("USER_SETTING", "preserved")
    key_names: list[str] = []
    launch_ids: list[str] = []

    def inspect(command, env):
        settings_path = Path(env["AIDER_MODEL_SETTINGS_FILE"])
        metadata_path = Path(env["AIDER_MODEL_METADATA_FILE"])
        settings = json.loads(settings_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        assert settings_path.parent == metadata_path.parent
        assert env["AIDER_MODEL"] == "nvidia_nim/catalog-model:variant"
        assert env["USER_SETTING"] == "preserved"
        assert "fcc_aider_proxy_auth_stale" not in env
        (key_name,) = [key for key in env if key.startswith("FCC_AIDER_PROXY_AUTH_")]
        assert key_name.upper() == key_name
        assert env[key_name] == "launcher-test-token"
        key_names.append(key_name)
        bare = next(entry for entry in settings if entry["name"] == env["AIDER_MODEL"])
        launch_id = bare["extra_params"]["extra_headers"]["x-fcc-launch-id"]
        assert launch_id
        launch_ids.append(launch_id)
        assert all(
            entry["extra_params"]["extra_headers"] == {"x-fcc-launch-id": launch_id}
            for entry in settings
        )
        assert bare["extra_params"] == {
            "model": "anthropic/nvidia_nim/catalog-model:variant",
            "api_base": "http://127.0.0.1:8182/v1/messages",
            "api_key": f"os.environ/{key_name}",
            "extra_headers": {"x-fcc-launch-id": launch_id},
        }
        assert env["AIDER_MODEL"] in metadata
        assert "launcher-test-token" not in settings_path.read_text()
        assert "launcher-test-token" not in metadata_path.read_text()
        assert command[1:3] == ["--set-env", "ANTHROPIC_API_KEY=fcc-local"]

    launch_capture.on_start = inspect
    args = [
        "--model",
        "native-selection",
        "--weak-model",
        "native-weak",
        "--editor-model",
        "native-editor",
        "--message",
        "hello",
    ]
    launch("aider", args)
    assert launch_capture.commands[0][-len(args) :] == args
    launch("aider", [])
    assert key_names[0] != key_names[1]
    assert launch_ids[0] != launch_ids[1]
