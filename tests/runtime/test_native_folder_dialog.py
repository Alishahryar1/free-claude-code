import subprocess
from types import SimpleNamespace

import pytest

from free_claude_code.runtime import native_folder_dialog as native


@pytest.mark.parametrize(
    ("status", "output", "diagnostic", "expected"),
    [
        (0, "/tmp/project café \n\n", "", "/tmp/project café \n"),
        (1, "", "", None),
        (1, "", "Gtk-WARNING: optional theme is missing", None),
    ],
)
def test_linux_selection_preserves_paths_and_normal_cancel(
    monkeypatch, status, output, diagnostic, expected
):
    monkeypatch.setattr(native.shutil, "which", lambda _tool: "/usr/bin/zenity")
    monkeypatch.setenv("ZENITY_CANCEL", "42")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, status, output.encode("utf-8"), diagnostic.encode("utf-8")
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    assert native._linux("/tmp/start") == expected
    assert "--filename=/tmp/start/" in calls[0][0]
    assert "ZENITY_CANCEL" not in calls[0][1]["env"]


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Gtk-WARNING: cannot open display",
        "Failed to open display",
        "Could not connect to display",
    ],
)
def test_linux_display_failure_is_not_user_cancellation(monkeypatch, diagnostic):
    monkeypatch.setattr(native.shutil, "which", lambda _tool: "/usr/bin/zenity")
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, b"", diagnostic.encode()
        ),
    )
    with pytest.raises(RuntimeError, match=diagnostic):
        native._linux(None)


def test_linux_uses_installed_kdialog_or_reports_missing_tools(monkeypatch):
    monkeypatch.setattr(
        native.shutil,
        "which",
        lambda tool: "/usr/bin/kdialog" if tool == "kdialog" else None,
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, b"/tmp/project\n", b"")

    monkeypatch.setattr(native.subprocess, "run", run)
    assert native._linux("/tmp") == "/tmp/project"
    assert calls[0][-2:] == ["--getexistingdirectory", "/tmp"]
    monkeypatch.setattr(native.shutil, "which", lambda _tool: None)
    with pytest.raises(RuntimeError, match="No desktop folder picker"):
        native._linux(None)


def test_macos_passes_the_hint_as_data_and_preserves_path_characters(monkeypatch):
    hint = '/tmp/quotes " and spaces'

    def run(command, **_kwargs):
        assert command[-1] == hint
        assert hint not in command[-2]
        return subprocess.CompletedProcess(command, 0, b"/tmp/project \n\n", b"")

    monkeypatch.setattr(native.subprocess, "run", run)
    assert native._macos(hint) == "/tmp/project \n"


@pytest.mark.parametrize("selection", ["", "C:/Projects/café"])
def test_windows_destroys_hidden_parent_after_selection(monkeypatch, selection):
    calls = []
    monkeypatch.setattr(native, "enable_dpi_awareness", lambda: calls.append("scaling"))
    root = SimpleNamespace(
        withdraw=lambda: calls.append("hidden"),
        attributes=lambda *_args: None,
        destroy=lambda: calls.append("destroyed"),
    )

    def create_parent():
        calls.append("created")
        return root

    module = SimpleNamespace(
        Tk=create_parent,
        filedialog=SimpleNamespace(askdirectory=lambda **_kwargs: selection),
    )
    monkeypatch.setitem(native.sys.modules, "tkinter", module)
    assert native._windows(None) == (selection or None)
    assert calls == ["scaling", "created", "hidden", "destroyed"]


def test_folder_hint_is_optional_and_checked_in_helper(tmp_path):
    assert native._initial_directory(str(tmp_path)) == str(tmp_path.resolve())
    assert native._initial_directory(str(tmp_path / "missing")) is None
    assert native._initial_directory("\0") is None
    assert native._initial_directory("") is None
