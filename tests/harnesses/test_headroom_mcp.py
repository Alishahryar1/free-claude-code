import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.headroom_mcp import (
    McpSetupError,
    Registration,
    configure_registrations,
    json_registration,
)


def test_registration_preserves_other_servers_and_rerun_is_a_noop(tmp_path: Path):
    path = tmp_path / "settings.json"
    unrelated = {"command": "other", "args": ["--existing"]}
    path.write_text(json.dumps({"mcpServers": {"other": unrelated}, "theme": "dark"}))
    target = json_registration("Muse", path, ("mcpServers",), tmp_path / "headroom")

    assert configure_registrations([target]) == ["Muse: configured"]
    first = path.read_bytes()
    assert json.loads(first)["mcpServers"]["other"] == unrelated
    assert json.loads(first)["theme"] == "dark"
    assert configure_registrations([target]) == ["Muse: already configured"]
    assert path.read_bytes() == first


def test_all_targets_are_checked_before_first_write(tmp_path: Path):
    first = json_registration(
        "First", tmp_path / "first.json", ("mcpServers",), tmp_path / "headroom"
    )
    conflict = tmp_path / "second.json"
    conflict.write_text('{"mcpServers":{"headroom":{"command":"custom","args":[]}}}')
    second = json_registration(
        "Second", conflict, ("mcpServers",), tmp_path / "headroom"
    )

    with pytest.raises(McpSetupError, match=r"Second.*headroom"):
        configure_registrations([first, second])

    assert not first.path.exists()
    assert (
        json.loads(conflict.read_text())["mcpServers"]["headroom"]["command"]
        == "custom"
    )


@pytest.mark.parametrize("disabled", [{"enabled": False}, {"disabled": True}])
def test_disabled_entry_is_not_reenabled(tmp_path: Path, disabled: dict[str, bool]):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"headroom": disabled}}))
    before = path.read_bytes()
    target = json_registration("Pi", path, ("mcpServers",), tmp_path / "headroom")

    assert configure_registrations([target]) == [
        "Pi: skipped (Headroom is disabled or blocked by policy)"
    ]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "source", ['{"mcpServers":[]}', '{"mcpServers":{"headroom":null}}', "{invalid"]
)
def test_bad_config_is_preserved(tmp_path: Path, source: str):
    path = tmp_path / "mcp.json"
    path.write_text(source)
    target = json_registration("Client", path, ("mcpServers",), tmp_path / "headroom")

    with pytest.raises(McpSetupError):
        configure_registrations([target])

    assert path.read_text() == source


def test_check_only_does_not_create_configuration(tmp_path: Path):
    path = tmp_path / "new" / "mcp.json"
    target = json_registration("Client", path, ("servers",), tmp_path / "headroom")

    assert configure_registrations([target], check_only=True) == ["Client: ready"]
    assert not path.parent.exists()


@pytest.mark.parametrize("transport", [{"type": "http"}, {"transport": "sse"}])
def test_existing_non_stdio_transport_is_a_conflict(tmp_path, transport):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "headroom": {
                        "command": str(tmp_path / "headroom"),
                        "args": ["mcp", "serve"],
                        **transport,
                    }
                }
            }
        )
    )
    target = json_registration("Client", path, ("mcpServers",), tmp_path / "headroom")
    with pytest.raises(McpSetupError, match="conflicting"):
        configure_registrations([target])


def test_native_command_success_requires_registration_readback(tmp_path: Path):
    target = Registration(
        label="Native",
        path=tmp_path / "config.json",
        expected={"command": str(tmp_path / "headroom"), "args": ["mcp", "serve"]},
        read=lambda: None,
        write=lambda: None,
    )

    with pytest.raises(McpSetupError, match=r"Native.*verify"):
        configure_registrations([target])


def test_rechecks_target_before_native_write(tmp_path: Path):
    reads: Iterator[JsonObject | None] = iter(
        [None, {"command": "new-custom", "args": []}]
    )
    written = []
    target = Registration(
        label="Native",
        path=tmp_path / "config.json",
        expected={"command": str(tmp_path / "headroom"), "args": ["mcp", "serve"]},
        read=lambda: next(reads),
        write=lambda: written.append(True),
    )

    with pytest.raises(McpSetupError, match=r"Native.*headroom"):
        configure_registrations([target])
    assert written == []
