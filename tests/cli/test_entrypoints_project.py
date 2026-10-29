"""Tests for the ``--project`` server flag and its global config slot."""

import os
import sys
from pathlib import Path

import pytest

from free_claude_code.cli import entrypoints
from free_claude_code.config import paths


@pytest.fixture
def isolated_fcc_home(monkeypatch):
    """Remove FCC_HOME before the test and scrub it afterwards."""
    monkeypatch.delenv("FCC_HOME", raising=False)
    yield
    os.environ.pop("FCC_HOME", None)


@pytest.fixture
def slot_path(monkeypatch, tmp_path):
    """Redirect project slots into the test sandbox (real slots live in ~/.fcc)."""
    slot = tmp_path / "slot-home"
    monkeypatch.setattr(paths, "project_slot_path", lambda project: slot)
    return slot


def test_project_flag_scaffolds_slot_with_env_template(
    isolated_fcc_home, slot_path, tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(entrypoints, "_port_is_listening", lambda port: False)
    project = tmp_path.resolve()
    entrypoints._apply_project_arg(["--project", str(tmp_path)])

    expected_env = (
        "# Free Claude Code configuration for:\n"
        f"#   {project}\n"
        "# Managed settings edited at /admin live in this slot; nothing\n"
        "# FCC-related is written into the project directory.\n"
        "FCC_CONFIG_SCHEMA=2\n"
        "\n# Stable port for this project's FCC instance, allocated once;\n"
        "# edit here or in /admin to choose a different one.\n"
        f"PORT={_derived_port(slot_path.name)}\n"
    )
    assert (slot_path / ".env").read_text(encoding="utf-8") == expected_env
    assert os.environ["FCC_HOME"] == str(slot_path)
    assert "project slot:" in capsys.readouterr().out


def test_project_flag_preserves_existing_slot_env(
    isolated_fcc_home, slot_path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(entrypoints, "_port_is_listening", lambda port: False)
    slot_path.mkdir(parents=True)
    existing = "FCC_CONFIG_SCHEMA=2\nMODEL=nvidia_nim/test-model\n"
    (slot_path / ".env").write_text(existing, encoding="utf-8")

    entrypoints._apply_project_arg(["--project", str(tmp_path)])

    text = (slot_path / ".env").read_text(encoding="utf-8")
    assert text.startswith(existing)
    assert f"PORT={_derived_port(slot_path.name)}\n" in text


def test_explicit_fcc_home_wins_over_project_flag(
    isolated_fcc_home, slot_path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FCC_HOME", "C:/explicit/wins")

    entrypoints._apply_project_arg(["--project", str(tmp_path)])

    assert os.environ["FCC_HOME"] == "C:/explicit/wins"
    assert slot_path.is_dir()


def test_project_flag_strips_pair_from_sys_argv(
    isolated_fcc_home, slot_path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["fcc-server", "--project", str(tmp_path), "--auto-started"]
    )

    entrypoints._apply_project_arg(None)

    assert sys.argv[1:] == ["--auto-started"]


def test_project_flag_does_not_mutate_caller_sequence(
    isolated_fcc_home, slot_path, tmp_path
) -> None:
    argv = ["--project", str(tmp_path)]

    entrypoints._apply_project_arg(argv)

    assert list(argv) == ["--project", str(tmp_path)]


def test_without_project_flag_nothing_happens(isolated_fcc_home) -> None:
    entrypoints._apply_project_arg(["--auto-started"])

    assert "FCC_HOME" not in os.environ


def test_project_flag_missing_value_exits_with_usage(isolated_fcc_home, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        entrypoints._apply_project_arg(["--project"])

    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().out


def test_serve_applies_project_arg_before_starting_the_server(
    isolated_fcc_home, monkeypatch, tmp_path
) -> None:
    from free_claude_code.cli import commands

    applied: list[list[str]] = []
    monkeypatch.setattr(
        entrypoints,
        "_apply_project_arg",
        lambda argv, program: applied.append(list(argv) if argv is not None else []),
    )
    monkeypatch.setattr(commands, "serve", lambda **kwargs: None)

    entrypoints.serve(["--project", str(tmp_path)])

    assert applied == [["--project", str(tmp_path)]]


# ---------------------------------------------------------------------------
# Stable project port allocation
# ---------------------------------------------------------------------------


@pytest.fixture
def sandboxed_home(monkeypatch, tmp_path):
    """Redirect the default config home so allocation scans stay in tmp."""
    default_home = tmp_path / "default-home"
    (default_home / "projects").mkdir(parents=True)
    monkeypatch.setattr(paths, "_default_config_dir_path", lambda: default_home)
    monkeypatch.setattr(entrypoints, "_port_is_listening", lambda port: False)
    return default_home


def _allocated_port(slot_path: Path) -> int:
    for line in (slot_path / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("PORT="):
            return int(line.split("=", 1)[1])
    raise AssertionError("slot .env has no PORT= line")


def _derived_port(name: str) -> int:
    import zlib

    return entrypoints.PROJECT_PORT_BASE + (
        zlib.crc32(name.encode("utf-8")) % entrypoints.PROJECT_PORT_RANGE
    )


def test_project_flag_allocates_stable_port(
    isolated_fcc_home, slot_path, sandboxed_home, tmp_path
) -> None:
    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")
    first = _allocated_port(slot_path)
    assert (
        entrypoints.PROJECT_PORT_BASE
        <= first
        < (entrypoints.PROJECT_PORT_BASE + entrypoints.PROJECT_PORT_RANGE)
    )

    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")
    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")
    assert _allocated_port(slot_path) == first


def test_project_flag_preserves_explicit_slot_port(
    isolated_fcc_home, slot_path, sandboxed_home, tmp_path
) -> None:
    slot_path.mkdir(parents=True)
    (slot_path / ".env").write_text(
        "FCC_CONFIG_SCHEMA=2\nPORT=9999\n", encoding="utf-8"
    )

    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")

    assert _allocated_port(slot_path) == 9999


def test_project_flag_port_is_deterministic_from_the_path(
    isolated_fcc_home, slot_path, sandboxed_home, tmp_path
) -> None:
    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")

    assert _allocated_port(slot_path) == _derived_port(slot_path.name)


def test_project_flag_port_avoids_other_slots(
    isolated_fcc_home, slot_path, sandboxed_home, tmp_path
) -> None:
    derived = _derived_port(slot_path.name)
    other = sandboxed_home / "projects" / "another-slot"
    other.mkdir(parents=True)
    (other / ".env").write_text(
        f"FCC_CONFIG_SCHEMA=2\nPORT={derived}\n", encoding="utf-8"
    )

    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")

    allocated = _allocated_port(slot_path)
    assert allocated != derived
    assert (
        entrypoints.PROJECT_PORT_BASE
        <= allocated
        < (entrypoints.PROJECT_PORT_BASE + entrypoints.PROJECT_PORT_RANGE)
    )


def test_project_flag_port_avoids_live_listeners(
    isolated_fcc_home, slot_path, sandboxed_home, tmp_path, monkeypatch
) -> None:
    derived = _derived_port(slot_path.name)
    claimed = set()

    def listening(port: int) -> bool:
        return port in claimed

    claimed.add(derived)
    monkeypatch.setattr(entrypoints, "_port_is_listening", listening)

    entrypoints._apply_project_arg(["--project", str(tmp_path)], program="fcc-claude")

    assert _allocated_port(slot_path) != derived


def test_launcher_strips_project_pair_and_resolves_slot(
    isolated_fcc_home, monkeypatch, tmp_path, launch_capture
) -> None:
    from tests.cli.test_launcher_workflow import launch

    slot = tmp_path / "prebuilt-slot"
    slot.mkdir(parents=True)
    (slot / ".env").write_text(
        "FCC_CONFIG_SCHEMA=2\nANTHROPIC_AUTH_TOKEN=launcher-test-token\nPORT=18201\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "project_slot_path", lambda project: slot)
    monkeypatch.delenv("PORT", raising=False)

    launch("claude", ["--project", str(tmp_path), "fix tests"])

    # The slot became this process's home, the flag pair never reached claude,
    # and the launch completed the FCC connection path (catalog + env build).
    assert os.environ.get("FCC_HOME") == str(slot)
    assert launch_capture.commands == [["claude", "fix tests"]]
    assert (
        launch_capture.environments[0]["ANTHROPIC_AUTH_TOKEN"] == "launcher-test-token"
    )
