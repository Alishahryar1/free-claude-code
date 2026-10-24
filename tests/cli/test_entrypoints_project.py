"""Tests for the ``--project`` server flag and its global config slot."""

import os
import sys

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
    isolated_fcc_home, slot_path, tmp_path, capsys
) -> None:
    project = tmp_path.resolve()
    entrypoints._apply_project_arg(["--project", str(tmp_path)])

    expected_env = (
        "# Free Claude Code configuration for:\n"
        f"#   {project}\n"
        "# Managed settings edited at /admin live in this slot; nothing\n"
        "# FCC-related is written into the project directory.\n"
        "FCC_CONFIG_SCHEMA=2\n"
    )
    assert (slot_path / ".env").read_text(encoding="utf-8") == expected_env
    assert os.environ["FCC_HOME"] == str(slot_path)
    assert "project slot:" in capsys.readouterr().out


def test_project_flag_preserves_existing_slot_env(
    isolated_fcc_home, slot_path, tmp_path
) -> None:
    slot_path.mkdir(parents=True)
    existing = "FCC_CONFIG_SCHEMA=2\nMODEL=nvidia_nim/test-model\n"
    (slot_path / ".env").write_text(existing, encoding="utf-8")

    entrypoints._apply_project_arg(["--project", str(tmp_path)])

    assert (slot_path / ".env").read_text(encoding="utf-8") == existing


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
        entrypoints, "_apply_project_arg", lambda argv: applied.append(list(argv))
    )
    monkeypatch.setattr(commands, "serve", lambda **kwargs: None)

    entrypoints.serve(["--project", str(tmp_path)])

    assert applied == [["--project", str(tmp_path)]]
