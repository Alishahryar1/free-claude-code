import json

import pytest

from free_claude_code.cli import doctor
from free_claude_code.config.settings import Settings
from free_claude_code.platform.termux import TermuxEnvironment
from free_claude_code.runtime import termux_diagnostics


def _termux_environment() -> TermuxEnvironment:
    from pathlib import Path

    return TermuxEnvironment(
        termux=True,
        android=True,
        bionic=True,
        arm64=True,
        prefix=Path("/data/data/com.termux/files/usr"),
        home=Path("/data/data/com.termux/files/home"),
        android_api_level=35,
    )


def _stub_healthy_termux(monkeypatch):
    monkeypatch.setattr(
        termux_diagnostics,
        "detect_termux_environment",
        _termux_environment,
    )
    monkeypatch.setattr(
        termux_diagnostics, "_termux_paths_available", lambda _env: True
    )
    monkeypatch.setattr(termux_diagnostics, "_config_is_writable", lambda: True)
    monkeypatch.setattr(
        termux_diagnostics,
        "_config_settings",
        lambda: (Settings(), "valid"),
    )
    monkeypatch.setattr(termux_diagnostics, "_localhost_round_trip", lambda: True)
    monkeypatch.setattr(termux_diagnostics, "_port_is_available", lambda _port: True)
    monkeypatch.setattr(
        termux_diagnostics.importlib,
        "import_module",
        lambda _name: object(),
    )
    monkeypatch.setattr(termux_diagnostics.shutil, "which", lambda _name: "/bin/tool")
    monkeypatch.setattr(termux_diagnostics.platform, "machine", lambda: "aarch64")


def test_termux_doctor_report_has_status_and_suggestions(monkeypatch):
    _stub_healthy_termux(monkeypatch)
    report = termux_diagnostics.collect_termux_diagnostics()

    assert report["platform"] == "Termux"
    assert report["summary"] == "PASS"
    assert {check["status"] for check in report["checks"]} == {"PASS"}
    assert all("name" in check and "detail" in check for check in report["checks"])


def test_termux_doctor_redacts_dependency_import_errors(monkeypatch):
    _stub_healthy_termux(monkeypatch)

    def fail_import(module_name):
        if module_name == "fastapi":
            raise ImportError("private-token")
        return object()

    monkeypatch.setattr(termux_diagnostics.importlib, "import_module", fail_import)
    report = termux_diagnostics.collect_termux_diagnostics()

    assert report["summary"] == "FAIL"
    assert "private-token" not in json.dumps(report)
    required_check = next(
        check for check in report["checks"] if check["name"] == "Core dependencies"
    )
    assert required_check["status"] == "FAIL"
    assert "suggested_fix" in required_check


def test_termux_doctor_command_does_not_attempt_clipboard(monkeypatch, capsys):
    report = {
        "platform": "Termux",
        "summary": "WARNING",
        "checks": [{"name": "Agent", "status": "WARNING", "detail": "Not installed."}],
    }
    monkeypatch.setattr(doctor, "collect_termux_diagnostics", lambda: report)
    monkeypatch.setattr(
        doctor,
        "copy_text",
        lambda _text: pytest.fail("Termux report should not use the clipboard"),
    )

    doctor.main(["--termux"])

    assert json.loads(capsys.readouterr().out) == report


def test_termux_doctor_returns_nonzero_when_a_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "collect_termux_diagnostics",
        lambda: {"platform": "Termux", "summary": "FAIL", "checks": []},
    )

    with pytest.raises(SystemExit) as exc_info:
        doctor.main(["--termux"])

    assert exc_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["summary"] == "FAIL"


def test_termux_doctor_rejects_conflicting_modes(capsys):
    with pytest.raises(SystemExit) as exc_info:
        doctor.main(["--termux", "--disconnect-integrations"])

    assert exc_info.value.code == 2
    assert "--termux cannot be combined" in capsys.readouterr().err
