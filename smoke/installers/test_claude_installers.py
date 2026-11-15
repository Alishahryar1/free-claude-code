"""Claude minimum compatibility through the real standalone installer functions."""

import os

import pytest

from smoke.installers.test_installers import _write_executable
from smoke.installers.test_installers import posix_harness as posix_harness
from smoke.installers.test_installers import powershell_harness as powershell_harness

if os.name == "nt":

    @pytest.fixture
    def claude_harness(powershell_harness):
        return powershell_harness
else:

    @pytest.fixture
    def claude_harness(posix_harness):
        return posix_harness


def _installed(harness, version):
    version_file = harness.root / "claude-version.txt"
    version_file.write_text(version)
    harness.env["FAKE_CLAUDE_VERSION_FILE"] = str(version_file)
    if os.name == "nt":
        command = harness.bin_dir / "claude.cmd"
        content = r"""@echo off
echo claude:%*>>"%CALL_LOG%"
if "%1"=="update" goto update
if "%FAIL_STEP%"=="claude-verify" exit /b 31
if "%1"=="--version" type "%FAKE_CLAUDE_VERSION_FILE%"
exit /b 0
:update
if "%FAIL_STEP%"=="claude-update" exit /b 32
if "%FAIL_STEP%"=="claude-no-change" exit /b 0
>"%FAKE_CLAUDE_VERSION_FILE%" echo 2.1.310
exit /b 0
"""
    else:
        command = harness.bin_dir / "claude"
        content = """#!/bin/sh
printf 'claude:%s\n' "$*" >> "$CALL_LOG"
if [ "$1" = update ]; then
    [ "$FAIL_STEP" != claude-update ] || exit 32
    [ "$FAIL_STEP" != claude-no-change ] || exit 0
    printf '2.1.310 (Claude Code)\n' > "$FAKE_CLAUDE_VERSION_FILE"
elif [ "$1" = --version ]; then
    [ "$FAIL_STEP" != claude-verify ] || exit 31
    cat "$FAKE_CLAUDE_VERSION_FILE"
    printf '\n'
fi
"""
    _write_executable(command, content)


def _ensure(harness, failure="", dry_run=False):
    if os.name == "nt":
        return harness.run_functions(
            "Ensure-ClaudeCode", *(["-DryRun"] if dry_run else []), fail_step=failure
        )
    harness.env["FAIL_STEP"] = failure
    return harness.run_functions(f"dry_run={int(dry_run)}\nensure_claude")


@pytest.mark.parametrize(
    "version",
    ["2.1.288 (Claude Code)", "2.1.310 (Claude Code)", "2.2.0+build.1 (Claude Code)"],
)
def test_preserves_supported_claude_and_updates_old_installation(
    claude_harness, version
):
    _installed(claude_harness, version)
    result = _ensure(claude_harness)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("claude:update" in claude_harness.calls()) == version.startswith("2.1.288")
    assert not any(call.startswith("download:") for call in claude_harness.calls())


@pytest.mark.parametrize("version", ["invalid", "2.1.310-beta", "2.1.310\n2.1.311"])
def test_invalid_claude_version_does_not_trigger_replacement(claude_harness, version):
    _installed(claude_harness, version)
    result = _ensure(claude_harness)
    assert result.returncode != 0
    assert claude_harness.calls() == ["claude:--version"]


@pytest.mark.parametrize(
    "failure", ["claude-update", "claude-no-change", "claude-verify"]
)
def test_failed_repair_is_not_success(claude_harness, failure):
    _installed(claude_harness, "2.1.288 (Claude Code)")
    result = _ensure(claude_harness, failure)
    assert result.returncode != 0
    assert "Verified Claude Code" not in result.stdout


def test_dry_run_does_not_execute_claude(claude_harness):
    _installed(claude_harness, "2.1.288 (Claude Code)")
    result = _ensure(claude_harness, dry_run=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not claude_harness.calls()


def test_missing_claude_uses_official_installer(claude_harness):
    result = _ensure(claude_harness)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "claude-install" in claude_harness.calls()
    assert "claude:--version" in claude_harness.calls()
