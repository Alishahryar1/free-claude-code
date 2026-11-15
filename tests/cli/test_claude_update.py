"""Installed Claude maintenance verifies the executable selected by PATH."""

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from free_claude_code.harnesses import claude_update
from free_claude_code.harnesses.claude import (
    CLAUDE_MINIMUM_VERSION_TEXT,
    claude_version,
    claude_version_supported,
)


@pytest.mark.parametrize(
    "output,supported",
    [
        ("2.1.288 (Claude Code)", False),
        ("2.1.309 (Claude Code)", False),
        ("2.1.310 (Claude Code)", True),
        (" 2.1.311+build.1 (Claude Code)\n", True),
        ("2.2.0", True),
        ("2.1.310-rc.1 (Claude Code)", False),
        ("2.1.999-beta", False),
        ("warning 2.1.310", False),
        ("2.1.310\n2.1.311", False),
        ("", False),
    ],
)
def test_supported_claude_versions(output, supported):
    assert claude_version_supported(output) is supported


@pytest.fixture
def commands(monkeypatch):
    resolve = Mock(return_value="/tools with spaces/claude")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "2.1.310 (Claude Code)"))
    monkeypatch.setattr(claude_update.shutil, "which", resolve)
    monkeypatch.setattr(claude_update.subprocess, "run", run)
    return resolve, run


def test_missing_claude_is_not_installed(commands):
    resolve, run = commands
    resolve.return_value = None
    assert claude_update.main([]) == 0
    run.assert_not_called()


def test_compatible_claude_is_not_updated(commands):
    _, run = commands
    assert claude_update.main([]) == 0
    assert [call.args[0] for call in run.call_args_list] == [
        ["/tools with spaces/claude", "--version"]
    ]


@pytest.mark.parametrize("binary", ["/native/claude", "C:/npm path/claude.cmd"])
def test_updates_selected_installation_and_rechecks(commands, binary):
    resolve, run = commands
    resolve.return_value = binary
    run.side_effect = [
        subprocess.CompletedProcess([], 0, "2.1.288 (Claude Code)"),
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0, "2.1.310 (Claude Code)"),
    ]
    assert claude_update.main([]) == 0
    assert [call.args[0] for call in run.call_args_list] == [
        [binary, "--version"],
        [binary, "update"],
        [binary, "--version"],
    ]
    assert resolve.call_count == 2


@pytest.mark.parametrize("version", ["2.1.288 (Claude Code)", "invalid"])
def test_successful_update_still_requires_supported_active_version(commands, version):
    resolve, run = commands
    resolve.side_effect = ["/native/claude", "/shadowed/claude"]
    run.side_effect = [
        subprocess.CompletedProcess([], 0, "2.1.288 (Claude Code)"),
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0, version),
    ]
    assert claude_update.main([]) == 1
    assert run.call_args.args[0] == ["/shadowed/claude", "--version"]


@pytest.mark.parametrize(
    "result",
    [
        subprocess.CompletedProcess([], 1, "2.1.310 (Claude Code)"),
        subprocess.CompletedProcess([], 0, "invalid"),
        OSError("cannot execute"),
        subprocess.TimeoutExpired("claude", 5),
    ],
)
def test_failed_probe_never_updates(commands, result):
    _, run = commands
    run.side_effect = [result]
    assert claude_update.main([]) == 1
    assert run.call_count == 1


def test_update_failure_is_not_reported_as_success(commands):
    _, run = commands
    run.side_effect = [
        subprocess.CompletedProcess([], 0, "2.1.288 (Claude Code)"),
        subprocess.CompletedProcess([], 17),
    ]
    assert claude_update.main([]) == 1
    assert run.call_count == 2


def test_dry_run_does_not_execute_claude(commands):
    _, run = commands
    assert claude_update.main(["--dry-run"]) == 0
    run.assert_not_called()


def test_version_parser_returns_numeric_components():
    assert claude_version("2.1.310 (Claude Code)") == (2, 1, 310)


def test_standalone_installer_minimums_match_runtime():
    root = Path(__file__).resolve().parents[2]
    assert (
        f'MIN_CLAUDE_VERSION="{CLAUDE_MINIMUM_VERSION_TEXT}"'
        in (root / "scripts/install.sh").read_text()
    )
    assert (
        f'$ClaudeMinimumVersion = "{CLAUDE_MINIMUM_VERSION_TEXT}"'
        in (root / "scripts/install.ps1").read_text()
    )
