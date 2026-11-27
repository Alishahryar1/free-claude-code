"""macOS runtime discovery and transactional application installation."""

import os
import plistlib
import subprocess
from pathlib import Path

import pytest

from free_claude_code.cli import macos_app
from free_claude_code.core.json_types import JsonObject


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path / "managed python"
    library = base / "lib/libpython3.14.dylib"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"library")
    executable = tmp_path / "venv's 日本語/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    executable.chmod(0o755)
    monkeypatch.setattr(macos_app.sys, "executable", str(executable))
    monkeypatch.setattr(macos_app.sys, "base_prefix", str(base))
    monkeypatch.setattr(macos_app.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(macos_app.platform, "python_version", lambda: "3.14.7")
    monkeypatch.setattr(macos_app.platform, "machine", lambda: "x86_64")
    variables = {"LDLIBRARY": library.name, "Py_GIL_DISABLED": 0, "Py_DEBUG": 0}
    monkeypatch.setattr(macos_app.sysconfig, "get_config_var", variables.get)

    def replace_metadata(name: object):
        return {"Requires-Python": "==3.14.7"}

    monkeypatch.setattr(macos_app, "metadata", replace_metadata)
    return executable, library, variables


def test_runtime_uses_venv_executable_and_interpreter_architecture(
    runtime: tuple[Path, Path, dict[str, int | str | None]],
) -> None:
    executable, library, _variables = runtime
    settings, architecture = macos_app.runtime_settings()
    assert settings == {
        "PythonExecutable": str(executable),
        "PythonLibrary": str(library),
        "PythonVersion": "3.14.7",
    }
    assert architecture == "x86_64"


def test_runtime_keeps_symlinked_venv_identity(
    runtime: tuple[Path, Path, dict[str, int | str | None]],
) -> None:
    executable, library, _variables = runtime
    base_executable = library.parent / "python"
    executable.rename(base_executable)
    try:
        executable.symlink_to(base_executable)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    settings, _architecture = macos_app.runtime_settings()
    assert settings["PythonExecutable"] == str(executable)
    assert Path(settings["PythonExecutable"]).resolve() == base_executable


def test_unsupported_interpreter_architecture_is_rejected(
    runtime: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(macos_app.platform, "machine", lambda: "armv7l")
    with pytest.raises(ValueError, match="architecture"):
        macos_app.runtime_settings()


@pytest.mark.parametrize("variable", ["Py_GIL_DISABLED", "Py_DEBUG"])
def test_unsupported_runtime_build_is_rejected(
    runtime: tuple[Path, Path, dict[str, int | str | None]], variable: str
) -> None:
    runtime[2][variable] = 1
    with pytest.raises(ValueError, match="release"):
        macos_app.runtime_settings()


def test_missing_runtime_library_is_rejected(
    runtime: tuple[Path, Path, dict[str, int | str | None]],
) -> None:
    runtime[1].unlink()
    with pytest.raises(ValueError, match="library"):
        macos_app.runtime_settings()


def test_wrong_python_version_is_rejected(
    runtime: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(macos_app.platform, "python_version", lambda: "3.15.0")
    with pytest.raises(ValueError, match="version"):
        macos_app.runtime_settings()


@pytest.fixture
def installer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: object):
    home = tmp_path / "home's 日本語"
    home.mkdir()
    resources = tmp_path / "package"
    templates = resources / "assets/macos"
    templates.mkdir(parents=True)
    (templates / "fcc-desktop-x86_64").write_bytes(b"intel template")
    (templates / "fcc-desktop-arm64").write_bytes(b"arm template")

    def replace_files(package: object):
        return resources

    monkeypatch.setattr(macos_app, "files", replace_files)
    monkeypatch.setattr(macos_app.Path, "home", lambda: home)
    monkeypatch.setattr(macos_app.sys, "platform", "darwin")
    if os.name == "nt":
        monkeypatch.setattr(
            Path,
            "symlink_to",
            lambda self, target, target_is_directory=False: self.mkdir(),
        )
    calls = []

    def sign(command: list[str], **kwargs: object):
        bundle = Path(command[-1])
        assert (bundle / "Contents/Resources/runtime.plist").is_file()
        assert (bundle / "Contents/Resources/.free-claude-code-owner").is_file()
        assert (bundle / "Contents/MacOS/fcc-desktop").read_bytes() == b"intel template"
        calls.append(command)

    monkeypatch.setattr(macos_app.subprocess, "run", sign)
    return home, resources, calls


def test_complete_bundle_is_signed_before_publication(
    installer: tuple[Path, Path, list[list[str]]],
    runtime: tuple[Path, Path, dict[str, int | str | None]],
) -> None:
    home, _resources, calls = installer
    app = macos_app.install()
    assert app == home / "Applications/Free Claude Code.app"
    contents = app / "Contents"
    with (contents / "Info.plist").open("rb") as stream:
        info: JsonObject = plistlib.load(stream)
    assert info["CFBundleIdentifier"] == macos_app.BUNDLE_ID
    assert info["CFBundleExecutable"] == "fcc-desktop"
    assert info["LSUIElement"] and info["LSMultipleInstancesProhibited"]
    with (contents / "Resources/runtime.plist").open("rb") as stream:
        assert plistlib.load(stream)["PythonExecutable"] == str(runtime[0])
    desktop_app = home / "Desktop/Free Claude Code.app"
    if os.name == "nt":
        assert desktop_app.is_dir()
    else:
        assert desktop_app.resolve() == app.resolve()
    assert [call[:-1] for call in calls] == [
        ["/usr/bin/codesign", "--force", "--sign", "-"],
        ["/usr/bin/codesign", "--verify", "--strict"],
    ]
    if os.name != "nt":
        assert (contents / "MacOS/fcc-desktop").stat().st_mode & 0o111 == 0o111


def owned_app(home: Path):
    app = home / "Applications/Free Claude Code.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents/.free-claude-code-owner").write_text(macos_app.BUNDLE_ID)
    (app / "old").write_text("preserve on failure")
    return app


def test_upgrade_replaces_legacy_owned_bundle_and_updates_runtime(
    installer: tuple[Path, Path, list[list[str]]],
    runtime: tuple[Path, Path, dict[str, int | str | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, _resources, _calls = installer
    app = owned_app(home)
    macos_app.install()
    assert not (app / "old").exists()
    new_python = runtime[0].with_name("python-new")
    new_python.write_bytes(b"python")
    new_python.chmod(0o755)
    monkeypatch.setattr(macos_app.sys, "executable", str(new_python))
    macos_app.install()
    with (app / "Contents/Resources/runtime.plist").open("rb") as stream:
        assert plistlib.load(stream)["PythonExecutable"] == str(new_python)
    assert list(app.parent.iterdir()) == [app]


@pytest.mark.parametrize("failure", ["sign", "publish", "resource"])
def test_failure_preserves_existing_app(
    installer: tuple[Path, Path, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    home, resources, _calls = installer
    app = owned_app(home)
    if failure == "sign":

        def fail(*args: object, **kwargs: object):
            raise subprocess.CalledProcessError(1, "codesign")

        monkeypatch.setattr(macos_app.subprocess, "run", fail)
    elif failure == "publish":
        rename = Path.rename

        def fail(path: Path, destination: Path):
            if path.name == "Free Claude Code.app" and path != app:
                raise OSError("publication failed")
            return rename(path, destination)

        monkeypatch.setattr(Path, "rename", fail)
    else:
        (resources / "assets/macos/fcc-desktop-x86_64").unlink()
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        macos_app.install()
    assert (app / "old").read_text() == "preserve on failure"
    assert list(app.parent.iterdir()) == [app]


def test_unowned_app_is_preserved(
    installer: tuple[Path, Path, list[list[str]]],
) -> None:
    home, _resources, calls = installer
    app = home / "Applications/Free Claude Code.app"
    app.mkdir(parents=True)
    with pytest.raises(ValueError, match="not managed"):
        macos_app.install()
    assert app.is_dir() and not calls


def test_failed_restore_keeps_backup_for_recovery(
    installer: tuple[Path, Path, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _resources, _calls = installer
    app = owned_app(home)
    rename = Path.rename

    def fail(path: Path, destination: Path):
        if destination == app:
            raise OSError("destination unavailable")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError):
        macos_app.install()
    backups = list(app.parent.glob(".fcc-app-*/previous.app/old"))
    assert len(backups) == 1
    assert backups[0].read_text() == "preserve on failure"


def test_app_symlink_is_not_treated_as_owned(
    installer: tuple[Path, Path, list[list[str]]],
) -> None:
    home, _resources, _calls = installer
    app = owned_app(home)
    real = app.with_name("Other.app")
    app.rename(real)
    try:
        os.symlink(real, app, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    assert app.is_symlink()
    with pytest.raises(ValueError, match="not managed"):
        macos_app.install()
    assert (real / "old").read_text() == "preserve on failure"


def test_unrelated_desktop_item_is_preserved(
    installer: tuple[Path, Path, list[list[str]]],
) -> None:
    home, _resources, _calls = installer
    desktop = home / "Desktop"
    desktop.mkdir()
    item = desktop / "Free Claude Code.app"
    item.write_text("unrelated")
    macos_app.install()
    assert item.read_text() == "unrelated"
