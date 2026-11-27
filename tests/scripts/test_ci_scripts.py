import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_text(name: str) -> str:
    return (_repo_root() / "scripts" / name).read_text(encoding="utf-8")


def _braced_body(text: str, declaration: str) -> str:
    start = text.index(declaration)
    brace_start = text.index("{", start)
    depth = 0

    for index, char in enumerate(text[brace_start:], start=brace_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index]

    raise AssertionError(f"Unclosed function body for {declaration}")


def _path_without_uv() -> str:
    uv_names = ("uv", "uv.exe", "uv.cmd", "uv.bat")
    entries = []
    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_entry:
            continue
        entry = Path(raw_entry)
        if any((entry / name).exists() for name in uv_names):
            continue
        entries.append(raw_entry)
    return os.pathsep.join(entries)


def _shell_interpreter() -> str:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("sh is not available on this platform")
    return sh


def _powershell_interpreter() -> str:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("PowerShell is not available on this platform")
    return pwsh


@pytest.mark.parametrize("script_name", ["ci.sh", "ci.ps1"])
@pytest.mark.parametrize("suffix", [".py", ".pyi"])
@pytest.mark.parametrize(
    ("before", "after", "directive"),
    [
        (" ", " ", "ignore[bad-assignment]"),
        (" ", " ", "ignore-errors"),
        ("", "", "ignore"),
    ],
)
def test_ci_rejects_pyrefly_suppressions(
    tmp_path: Path,
    script_name: str,
    suffix: str,
    before: str,
    after: str,
    directive: str,
) -> None:
    comment = f"#{before}pyrefly:{after}{directive}"
    assignment = "x: int = 'invalid'"
    source = (
        f"{comment}\n{assignment}\n"
        if directive == "ignore-errors"
        else f"{assignment}  {comment}\n"
    )
    (tmp_path / f"sample{suffix}").write_text(source, encoding="utf-8")
    script = str(_repo_root() / "scripts" / script_name)
    if script_name == "ci.sh":
        command = [_shell_interpreter(), script, "--only", "suppressions"]
    else:
        command = [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            script,
            "-Only",
            "suppressions",
        ]
    result = subprocess.run(
        command, cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert comment in result.stdout


@pytest.mark.parametrize("suffix", [".py", ".pyi"])
@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        (
            "from typing import cast\nx: str\ny = cast(str, x)\n",
            "redundant-cast",
        ),
        (
            "from typing import Any\npayload: Any\nvalue = payload.get('name')\n",
            "explicit-any-use",
        ),
        ("x: int = 'invalid'  # type:ignore\n", "bad-assignment"),
        ("x: int = 'invalid'  # pyrefly:\u0085ignore\n", "bad-assignment"),
        (f"# {'pyrefly: ignore-errors'}\nx: int = 'invalid'\n", "bad-assignment"),
    ],
)
def test_pyrefly_check_enforces_project_diagnostics(
    tmp_path: Path, suffix: str, source: str, diagnostic: str
) -> None:
    sample = tmp_path / f"sample{suffix}"
    sample.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--config",
            str(_repo_root() / "pyproject.toml"),
            "--output-format",
            "min-text",
            str(sample),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert f"[{diagnostic}]" in result.stdout


def test_ci_sh_runs_ci_checks_in_order() -> None:
    text = _script_text("ci.sh")
    legacy_future_import = "from __future__ import " + "annotations"

    assert (
        'CHECK_ORDER="suppressions ruff-format ruff-check pyrefly pytest playwright macos-desktop"'
        in text
    )
    assert "grep -rE" in text
    assert "Fix the underlying type/import issue instead" in text
    assert legacy_future_import in text
    assert "legacy future annotations are not allowed" in text
    assert "--exclude-dir=.venv" in text
    assert "--exclude-dir=.git" in text
    assert "uv run ruff format" in text
    assert "uv run ruff format --check" not in text
    assert "uv run ruff check --fix" in text
    assert "uv run pyrefly check" in text
    assert "uv run pytest -v --tb=short" in text
    assert "uv run playwright install chromium" in text
    assert "uv run pytest e2e -v --tb=short" in text
    assert "--only" in text
    assert "--skip" in text
    assert "--dry-run" in text
    assert "uv is required but was not found on PATH" in text
    assert "npm" not in text
    assert "--confcutdir=smoke/desktop" in text
    assert "uv self update" not in text


def test_ci_sh_dry_run_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _shell_interpreter(),
            str(_repo_root() / "scripts" / "ci.sh"),
            "--only",
            "pytest",
            "--dry-run",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "+ uv run pytest -v --tb=short" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_sh_playwright_dry_run_prints_browser_and_test_commands() -> None:
    result = subprocess.run(
        [
            _shell_interpreter(),
            str(_repo_root() / "scripts" / "ci.sh"),
            "--only",
            "playwright",
            "--dry-run",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "+ uv run playwright install chromium" in result.stdout
    assert "+ uv run pytest e2e -v --tb=short" in result.stdout
    assert "uv is required" not in result.stderr


@pytest.mark.parametrize(
    ("check_id", "command"),
    [
        ("ruff-format", "+ uv run ruff format"),
        ("ruff-check", "+ uv run ruff check --fix"),
        ("macos-desktop", "+ uv run --locked python scripts/build_macos_launcher.py"),
    ],
)
def test_ci_sh_dry_run_prints_local_ruff_repair_commands(
    check_id: str, command: str
) -> None:
    result = subprocess.run(
        [
            _shell_interpreter(),
            str(_repo_root() / "scripts" / "ci.sh"),
            "--only",
            check_id,
            "--dry-run",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert command in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_sh_suppression_only_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _shell_interpreter(),
            str(_repo_root() / "scripts" / "ci.sh"),
            "--only",
            "suppressions",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Ban suppressions and legacy annotations" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_sh_is_tracked_executable() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "scripts/ci.sh"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.startswith("100755 ")


def test_ci_sh_fail_fast_runs_checks_sequentially() -> None:
    text = _script_text("ci.sh")
    main = text[text.index('parse_args "$@"') :]

    suppress_index = text.index("run_suppressions()")
    ruff_format_index = text.index("run_ruff_format()")
    ruff_check_index = text.index("run_ruff_check()")
    pyrefly_index = text.index("run_pyrefly()")
    pytest_index = text.index("run_pytest()")
    playwright_index = text.index("run_playwright()")

    assert (
        suppress_index
        < ruff_format_index
        < ruff_check_index
        < pyrefly_index
        < pytest_index
        < playwright_index
    )
    assert "for check_id in $CHECK_ORDER" in main


def test_ci_ps1_runs_ci_checks_in_order() -> None:
    text = _script_text("ci.ps1")
    legacy_future_import = "from __future__ import " + "annotations"

    assert '"suppressions"' in text
    assert '"ruff-format"' in text
    assert '"ruff-check"' in text
    assert '"pyrefly"' in text
    assert '"pytest"' in text
    assert '"playwright"' in text
    assert "Select-String -Pattern" in text
    assert "Fix the underlying type/import issue instead" in text
    assert legacy_future_import in text
    assert "legacy future annotations are not allowed" in text
    assert ".venv" in text
    assert ".git" in text
    assert '"run", "ruff", "format"' in text
    assert '"format", "--check"' not in text
    assert '"run", "ruff", "check", "--fix"' in text
    assert '"-v", "--tb=short"' in text
    assert '"run", "playwright", "install", "chromium"' in text
    assert '"run", "pytest", "e2e", "-v", "--tb=short"' in text
    assert "-Only" in text
    assert "-Skip" in text
    assert "-DryRun" in text
    assert "uv is required but was not found on PATH" in text
    assert "npm" not in text
    assert "--confcutdir=smoke/desktop" in text
    assert "uv self update" not in text


def test_ci_ps1_dry_run_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            str(_repo_root() / "scripts" / "ci.ps1"),
            "-Only",
            "pytest",
            "-DryRun",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "+ uv run pytest -v --tb=short" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_ps1_playwright_dry_run_prints_browser_and_test_commands() -> None:
    result = subprocess.run(
        [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            str(_repo_root() / "scripts" / "ci.ps1"),
            "-Only",
            "playwright",
            "-DryRun",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "+ uv run playwright install chromium" in result.stdout
    assert "+ uv run pytest e2e -v --tb=short" in result.stdout
    assert "uv is required" not in result.stderr


@pytest.mark.parametrize(
    ("check_id", "command"),
    [
        ("ruff-format", "+ uv run ruff format"),
        ("ruff-check", "+ uv run ruff check --fix"),
    ],
)
def test_ci_ps1_dry_run_prints_local_ruff_repair_commands(
    check_id: str, command: str
) -> None:
    result = subprocess.run(
        [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            str(_repo_root() / "scripts" / "ci.ps1"),
            "-Only",
            check_id,
            "-DryRun",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert command in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_ps1_suppression_only_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            str(_repo_root() / "scripts" / "ci.ps1"),
            "-Only",
            "suppressions",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Ban suppressions and legacy annotations" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_ps1_fail_fast_runs_checks_sequentially() -> None:
    text = _script_text("ci.ps1")

    assert "foreach ($checkId in $CheckOrder)" in text
    assert "Invoke-SuppressionsCheck" in text
    assert "Invoke-RuffFormatCheck" in text
    assert "Invoke-RuffLintCheck" in text
    assert "Invoke-PyreflyCheck" in text
    assert "Invoke-PytestCheck" in text
    assert "Invoke-PlaywrightCheck" in text

    suppress_index = text.index("function Invoke-SuppressionsCheck")
    ruff_format_index = text.index("function Invoke-RuffFormatCheck")
    ruff_check_index = text.index("function Invoke-RuffLintCheck")
    pyrefly_index = text.index("function Invoke-PyreflyCheck")
    pytest_index = text.index("function Invoke-PytestCheck")
    playwright_index = text.index("function Invoke-PlaywrightCheck")

    assert (
        suppress_index
        < ruff_format_index
        < ruff_check_index
        < pyrefly_index
        < pytest_index
        < playwright_index
    )


def _run_install_with_stubbed_uv(
    tmp_path: Path, uv_system_certs: str | None
) -> list[str]:
    """Run configure_uv_environment and install_free_claude_code against stubs.

    Returns one line per recorded ``uv`` call: its arguments and the
    UV_SYSTEM_CERTS value that call inherited.
    """
    text = _script_text("install.sh")
    functions_only = text[: text.index('\nparse_args "$@"\n')]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "uv-calls.log"
    stubs = {
        "uv": f'#!/bin/sh\necho "$* UV_SYSTEM_CERTS=${{UV_SYSTEM_CERTS-unset}}" >> "{calls}"\n',
        "uname": "#!/bin/sh\necho Darwin\n",
        "pgrep": "#!/bin/sh\nexit 1\n",
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8", newline="\n")
        stub.chmod(0o755)

    harness = tmp_path / "harness.sh"
    harness.write_text(
        functions_only
        + '\ntrap - EXIT INT HUP TERM\nPYTHON_VERSION=3.14.10\ninstall_log=""\n'
        + "configure_uv_environment\ninstall_free_claude_code\n",
        encoding="utf-8",
        newline="\n",
    )

    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": str(tmp_path),
    }
    if uv_system_certs is not None:
        env["UV_SYSTEM_CERTS"] = uv_system_certs
    subprocess.run(["sh", str(harness)], env=env, check=True, capture_output=True)
    return calls.read_text(encoding="utf-8").splitlines()


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="requires a POSIX shell and POSIX paths",
)
def test_install_sh_installs_managed_python_before_pinned_tool_install(
    tmp_path: Path,
) -> None:
    calls = _run_install_with_stubbed_uv(tmp_path, None)

    assert len(calls) == 2
    assert calls[0].startswith("python install --managed-python 3.14.10 ")
    assert calls[1].startswith(
        "tool install --force --refresh-package free-claude-code"
    )
    assert "--python 3.14.10+gil" in calls[1]


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="requires a POSIX shell and POSIX paths",
)
def test_install_sh_defaults_uv_system_certs_when_unset(tmp_path: Path) -> None:
    calls = _run_install_with_stubbed_uv(tmp_path, None)

    assert calls
    assert all(call.endswith("UV_SYSTEM_CERTS=1") for call in calls)


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="requires a POSIX shell and POSIX paths",
)
def test_install_sh_keeps_explicit_uv_system_certs_opt_out(tmp_path: Path) -> None:
    calls = _run_install_with_stubbed_uv(tmp_path, "0")

    assert calls
    assert all(call.endswith("UV_SYSTEM_CERTS=0") for call in calls)
