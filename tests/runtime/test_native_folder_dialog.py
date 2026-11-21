import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from free_claude_code.runtime import native_folder_dialog as native


@pytest.mark.parametrize("platform", ["macos", "linux"])
def test_posix_dialog_replaces_helper_without_spawning_child(
    monkeypatch: pytest.MonkeyPatch, platform: str
):
    calls = []

    def exec_command(path: str, arguments: list[str], *env: object):
        calls.append((path, arguments, env))
        raise SystemExit(0)

    def nested_child(*args: object, **kwargs: object):
        pytest.fail("Native dialog must replace the owned helper process")

    monkeypatch.setattr(native.os, "execv", exec_command)
    monkeypatch.setattr(native.os, "execve", exec_command)
    monkeypatch.setattr(subprocess, "run", nested_child)

    def replace_which(_: object):
        return "/usr/bin/zenity"

    monkeypatch.setattr(native.shutil, "which", replace_which)
    with pytest.raises(SystemExit):
        getattr(native, "_" + platform)("/tmp/start")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("platform", "status", "output", "diagnostic", "expected"),
    [
        ("linux", 0, "/tmp/project café \n\n", "", "/tmp/project café \n"),
        ("linux", 1, "", "", None),
        ("linux", 1, "", "Gtk-WARNING: optional theme is missing", None),
        (
            "linux",
            0,
            "/tmp/project\n",
            "optional component cannot open display",
            "/tmp/project",
        ),
        ("darwin", 0, "/tmp/project café \n\n", "", "/tmp/project café \n"),
        ("darwin", 0, "\n", "", None),
    ],
)
def test_native_selection_preserves_paths_and_normal_cancel(
    platform: str, status: int, output: str, diagnostic: str, expected: str | None
):
    assert (
        native.decode_result(
            output.encode(), diagnostic.encode(), status, platform=platform
        )
        == expected
    )


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Gtk-WARNING: cannot open display",
        "Failed to open display",
        "Could not connect to display",
    ],
)
def test_linux_display_failure_is_not_user_cancellation(diagnostic: str):
    with pytest.raises(RuntimeError, match=diagnostic):
        native.decode_result(b"", diagnostic.encode(), 1, platform="linux")


def test_linux_uses_installed_tool_arguments_and_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def execve(path: str, command: list[str], env: dict[str, str]) -> None:
        calls.append((path, command, env))

    monkeypatch.setenv("ZENITY_CANCEL", "42")
    monkeypatch.setattr(native.os, "execve", execve)

    def replace_which(_: object):
        return "/usr/bin/zenity"

    monkeypatch.setattr(native.shutil, "which", replace_which)
    native._linux("/tmp/start")
    assert calls[0][0] == "/usr/bin/zenity"
    assert "--filename=/tmp/start/" in calls[0][1]
    assert "ZENITY_CANCEL" not in calls[0][2]

    def replace_which_2(tool: str):
        return "/usr/bin/kdialog" if tool == "kdialog" else None

    monkeypatch.setattr(
        native.shutil,
        "which",
        replace_which_2,
    )
    native._linux("/tmp")
    assert calls[1][0] == "/usr/bin/kdialog"
    assert calls[1][1][-2:] == ["--getexistingdirectory", "/tmp"]

    def replace_which_3(_: object):
        return None

    monkeypatch.setattr(native.shutil, "which", replace_which_3)
    with pytest.raises(RuntimeError, match="No desktop folder picker"):
        native._linux(None)


def test_macos_passes_the_hint_as_data(monkeypatch: pytest.MonkeyPatch):
    hint = '/tmp/quotes " and spaces'
    calls: list[tuple[str, list[str]]] = []

    def execv(path: str, command: list[str]) -> None:
        calls.append((path, command))

    monkeypatch.setattr(native.os, "execv", execv)
    native._macos(hint)
    assert calls[0][0] == "/usr/bin/osascript"
    assert calls[0][1][-1] == hint
    assert hint not in calls[0][1][-2]


@pytest.mark.parametrize("failure", ["missing_tool", "exec_error"])
def test_helper_setup_failure_cannot_be_mistaken_for_cancel(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
):
    monkeypatch.setattr(native.sys, "platform", "linux")

    def replace_which(_: object):
        return None if failure == "missing_tool" else "/missing/zenity"

    monkeypatch.setattr(
        native.shutil,
        "which",
        replace_which,
    )

    def fail(*args: object):
        raise FileNotFoundError("Native executable disappeared")

    monkeypatch.setattr(native.os, "execve", fail)
    assert native.main() == 2
    captured = capsys.readouterr()
    assert captured.err
    with pytest.raises(RuntimeError):
        native.decode_result(
            captured.out.encode(), captured.err.encode(), 2, platform="linux"
        )


@pytest.mark.parametrize("selection", ["", "C:/Projects/café"])
def test_windows_destroys_hidden_parent_after_selection(
    monkeypatch: pytest.MonkeyPatch, selection: str
):
    calls = []
    monkeypatch.setattr(native, "enable_dpi_awareness", lambda: calls.append("scaling"))

    def callback(*_args: object):
        return None

    root = SimpleNamespace(
        withdraw=lambda: calls.append("hidden"),
        attributes=callback,
        destroy=lambda: calls.append("destroyed"),
    )

    def create_parent():
        calls.append("created")
        return root

    def callback_2(**_kwargs: object):
        return selection

    module = SimpleNamespace(
        Tk=create_parent,
        filedialog=SimpleNamespace(askdirectory=callback_2),
    )
    monkeypatch.setitem(native.sys.modules, "tkinter", module)
    assert native._windows(None) == (selection or None)
    assert calls == ["scaling", "created", "hidden", "destroyed"]


def test_folder_hint_is_optional_and_checked_in_helper(tmp_path: Path):
    assert native._initial_directory(str(tmp_path)) == str(tmp_path.resolve())
    assert native._initial_directory(str(tmp_path / "missing")) is None
    assert native._initial_directory("\0") is None
    assert native._initial_directory("") is None
