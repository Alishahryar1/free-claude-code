"""Exercise native AGY installation, reuse, and upgrade through real installer functions."""

import os
from pathlib import Path

import pytest

from smoke.installers.test_installers import CODING_AGENTS, _write_executable
from smoke.installers.test_installers import posix_harness as posix_harness
from smoke.installers.test_installers import powershell_harness as powershell_harness

if os.name == "nt":

    @pytest.fixture
    def agy_harness(powershell_harness):
        return powershell_harness
else:

    @pytest.fixture
    def agy_harness(posix_harness):
        return posix_harness


def _installed(harness, version):
    version_file = harness.root / "agy-version.txt"
    version_file.write_text(version)
    harness.env["FAKE_AGY_VERSION_FILE"] = str(version_file)
    if os.name == "nt":
        command = harness.bin_dir / "agy.cmd"
        content = r"""@echo off
echo agy:%*>>"%CALL_LOG%"
if "%FAIL_STEP%"=="agy-verify" exit /b 31
if "%1"=="update" (
    if "%FAIL_STEP%"=="agy-update" exit /b 32
    >"%FAKE_AGY_VERSION_FILE%" echo 1.4.7
    exit /b 0
)
if "%1"=="--version" type "%FAKE_AGY_VERSION_FILE%"
exit /b 0
"""
    else:
        command = harness.bin_dir / "agy"
        content = """#!/bin/sh
printf 'agy:%s\n' "$*" >> "$CALL_LOG"
[ "$FAIL_STEP" != agy-verify ] || exit 31
if [ "$1" = update ]; then
    [ "$FAIL_STEP" != agy-update ] || exit 32
    printf '1.4.7\n' > "$FAKE_AGY_VERSION_FILE"
elif [ "$1" = --version ]; then
    cat "$FAKE_AGY_VERSION_FILE"
    printf '\n'
fi
"""
    _write_executable(command, content)
    return command


def _ensure(harness, *, fail_step=""):
    if os.name == "nt":
        return harness.run_functions(
            "Add-KnownBinDirectories\nEnsure-Agy", fail_step=fail_step
        )
    harness.env["FAIL_STEP"] = fail_step
    return harness.run_functions("add_known_bin_directories\nensure_agy")


@pytest.mark.parametrize("version", ["1.4.7", "2.0.0", "1.4.6"])
def test_agy_keeps_supported_versions_and_updates_older_ones(agy_harness, version):
    _installed(agy_harness, version)
    result = _ensure(agy_harness)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("agy:update" in agy_harness.calls()) == (version == "1.4.6")
    assert not any(call.startswith("download:") for call in agy_harness.calls())


@pytest.mark.parametrize("version", ["unrecognized", "1.4", "1.4.7-beta.1"])
def test_agy_invalid_version_stops_before_other_installation(agy_harness, version):
    _installed(agy_harness, version)
    result = _ensure(agy_harness)
    assert result.returncode != 0
    assert not any(call.startswith("download:") for call in agy_harness.calls())


def test_agy_missing_binary_uses_official_installer(agy_harness):
    result = _ensure(agy_harness)
    assert result.returncode == 0, result.stdout + result.stderr
    suffix = "ps1" if os.name == "nt" else "sh"
    assert (
        f"download:https://antigravity.google/cli/install.{suffix}"
        in agy_harness.calls()
    )
    assert "agy-install" in agy_harness.calls()
    assert "agy:--version" in agy_harness.calls()


@pytest.mark.parametrize("failure", ["agy-download", "agy-install", "agy-verify"])
def test_agy_installation_failure_is_not_reported_as_success(agy_harness, failure):
    result = _ensure(agy_harness, fail_step=failure)
    assert result.returncode != 0
    assert "Verified Antigravity" not in result.stdout


def test_agy_update_failure_stops(agy_harness):
    _installed(agy_harness, "1.4.6")
    result = _ensure(agy_harness, fail_step="agy-update")
    assert result.returncode != 0 and "agy:update" in agy_harness.calls()


def test_agy_running_process_prevents_replacement(agy_harness):
    _installed(agy_harness, "1.4.6")
    agy_harness.env["FCC_RUNNING_COMMAND"] = "agy"
    agy_harness.env["FCC_RUNNING_PHASE"] = "early"
    result = _ensure(agy_harness)
    assert result.returncode != 0
    assert "Close Antigravity" in result.stdout + result.stderr
    assert "agy:update" not in agy_harness.calls()


def test_agy_dry_run_makes_no_native_changes(agy_harness):
    command = _installed(agy_harness, "1.4.6")
    before = command.read_bytes()
    result = (
        agy_harness.run_functions("Ensure-Agy", "-DryRun")
        if os.name == "nt"
        else agy_harness.run_functions("dry_run=1\nensure_agy")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not agy_harness.calls()
    assert command.read_bytes() == before


def test_agy_windows_default_install_directory_is_discovered(powershell_harness):
    source = _installed(powershell_harness, "1.4.7")
    target = Path(powershell_harness.env["LOCALAPPDATA"]) / "agy/bin/agy.cmd"
    target.parent.mkdir(parents=True)
    source.replace(target)
    result = _ensure(powershell_harness)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agy:update" not in powershell_harness.calls()
    assert not any(call.startswith("download:") for call in powershell_harness.calls())


@pytest.mark.parametrize("selected", [False, True])
def test_agy_selection_preserves_native_settings_and_verifies_fcc(
    agy_harness, selected
):
    windows = os.name == "nt"
    if selected:
        (agy_harness.bin_dir / ("opencode.cmd" if windows else "opencode")).unlink()
    home = Path(agy_harness.env["USERPROFILE" if windows else "HOME"])
    settings = home / ".gemini/antigravity-cli/settings.json"
    settings.parent.mkdir(parents=True)
    original = '{"modelProvider":"","model":"personal"}\n'
    settings.write_text(original)
    answers = [
        "y" if selected and command == "agy" else "n"
        for command in CODING_AGENTS
        if selected or command != "opencode"
    ] + ["n"]
    result = agy_harness.run_interactive(
        answers if windows else "\n".join(answers) + "\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = agy_harness.calls()
    assert ("agy-install" in calls) is selected
    assert ("agy:--version" in calls) is selected
    assert "fcc-server:--version" in calls
    assert settings.read_text() == original
    assert not (home / ".fcc/antigravity-integration.json").exists()
    if selected:
        assert (
            "Connect Antigravity CLI on FCC's Integrations page first." in result.stdout
        )
        assert "fcc-agy --connect" not in result.stdout
        assert not any(
            f"{command}:--version" in calls
            for command in CODING_AGENTS
            if command != "agy"
        )
