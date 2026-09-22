import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from free_claude_code.cli import update


@pytest.fixture
def installation(tmp_path, monkeypatch):
    root = tmp_path / "tools" / "free-claude-code"
    root.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / ("fcc-update.cmd" if update.os.name == "nt" else "fcc-update")
    launcher.touch()
    module = root / "lib" / "update.py"
    module.parent.mkdir()
    module.touch()
    monkeypatch.setattr(update.sys, "prefix", str(root))
    monkeypatch.setattr(update, "__file__", str(module))
    monkeypatch.setattr(update, "running_fcc", lambda: [])
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")

    def capture(args):
        if args[1:] == ["--version"]:
            return "uv 0.12.13"
        if args[1:] == ["tool", "dir"]:
            return str(root.parent)
        if args[1:] == ["tool", "dir", "--bin"]:
            return str(bin_dir)
        raise AssertionError(args)

    monkeypatch.setattr(update, "capture", capture)

    def receipt(extras=(), source=None):
        (root / "uv-receipt.toml").write_text(
            '[tool]\npython = "3.14.0"\nrequirements = [{name = "free-claude-code", url = '
            + json.dumps(source or update.ARCHIVE_URL)
            + ", extras = "
            + json.dumps(list(extras))
            + "}]\nentrypoints = [{ name = "
            + json.dumps(launcher.name)
            + ', from = "free-claude-code", install-path = '
            + json.dumps(str(launcher))
            + "}]\n",
            encoding="utf-8",
        )

    receipt()
    return root, launcher, receipt


@pytest.mark.parametrize(
    "extras", [(), ("voice",), ("voice-local",), ("voice", "voice-local")]
)
def test_prepare_preserves_install_choices(installation, monkeypatch, extras):
    _, launcher, receipt = installation
    receipt(extras)
    monkeypatch.setattr(update, "torch_backend", lambda uv, options: "cu130")
    result = update.prepare(Path("uv"), launcher)
    args = result["arguments"]
    assert args[:4] == ["tool", "install", "--force", "--refresh-package"]
    assert args[4:7] == ["free-claude-code", "--python", "3.14.0"]
    assert ("--torch-backend" in args) == ("voice-local" in extras)
    assert (
        args[-1]
        == "free-claude-code"
        + ("[" + ",".join(sorted(extras)) + "]" if extras else "")
        + " @ "
        + update.ARCHIVE_URL
    )


def test_foreign_source_refused_before_update(installation):
    _, launcher, receipt = installation
    receipt(source="https://example.com/foreign.zip")
    with pytest.raises(ValueError, match="official"):
        update.prepare(Path("uv"), launcher)


def test_checkout_cannot_update_installed_tool(installation, monkeypatch, tmp_path):
    _, launcher, _ = installation
    monkeypatch.setattr(update, "__file__", str(tmp_path / "checkout" / "update.py"))
    with pytest.raises(ValueError, match="installation"):
        update.prepare(Path("uv"), launcher)


def test_wrong_launcher_refused(installation, tmp_path):
    other = tmp_path / "other.cmd"
    other.touch()
    with pytest.raises(ValueError, match="launcher"):
        update.prepare(Path("uv"), other)


def test_running_fcc_refused(installation, monkeypatch):
    _, launcher, _ = installation
    monkeypatch.setattr(update, "running_fcc", lambda: ["fcc-server (PID 123)"])
    with pytest.raises(ValueError, match=r"Stop.*fcc-server"):
        update.prepare(Path("uv"), launcher)


@pytest.mark.parametrize(
    "build,expected",
    [
        ("2.14.0+cpu", "cpu"),
        ("2.14.0+cu130", "cu130"),
        ("2.14.0+rocm6.2.4", "rocm6.2.4"),
        ("2.14.0+xpu", "xpu"),
    ],
)
def test_torch_backend_uses_build_not_hardware(monkeypatch, build, expected):
    monkeypatch.setattr(update, "torch_build_version", lambda: build)
    calls = []
    monkeypatch.setattr(update, "capture", lambda args: calls.append(args) or "")
    assert update.torch_backend(Path("uv"), {}) == expected
    assert calls == [["uv", "tool", "install", "--torch-backend", expected, "--help"]]


def test_unknown_torch_backend_is_not_silently_changed(monkeypatch):
    monkeypatch.setattr(update, "torch_build_version", lambda: "2.14.0+custom")
    with pytest.raises(ValueError, match="voice installer"):
        update.torch_backend(Path("uv"), {})


def test_recorded_backend_is_preserved_without_inspecting_build(monkeypatch):
    calls = []
    monkeypatch.setattr(update, "capture", lambda args: calls.append(args) or "")
    monkeypatch.setattr(
        update, "torch_build_version", lambda: pytest.fail("must use recorded backend")
    )
    assert update.torch_backend(Path("uv"), {"torch-backend": "auto"}) == "auto"


def test_torch_build_metadata_is_read_without_loading_code(tmp_path, monkeypatch):
    version_file = tmp_path / "version.py"
    version_file.write_text(
        "raise RuntimeError('must not execute')\n__version__: str = '2.14.0+cu130'\n"
    )
    monkeypatch.setattr(
        update.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(version="2.14.0", locate_file=lambda _: version_file),
    )
    assert update.torch_build_version() == "2.14.0+cu130"


def test_process_guard_finds_fcc_without_matching_unrelated_apps(monkeypatch):
    if update.os.name == "nt":
        output = (
            '"fcc-server.exe","123","Console"\n"not-fcc-server.exe","124","Console"'
        )
    else:
        output = "123 /some/bin/fcc-server\n124 /some/bin/not-fcc-server\n"
    monkeypatch.setattr(update, "capture", lambda _: output)
    found = update.running_fcc()
    assert len(found) == 1
    assert "123" in found[0]


def test_no_server_imports_in_isolated_preparation_module():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from free_claude_code.cli import update; import sys; assert not any(k.startswith(('free_claude_code.config', 'free_claude_code.runtime', 'free_claude_code.providers')) for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
