import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

UV_MINIMUM = "0.12.13"
CI_SETUP = Path(".github/actions/ci-environment/action.yml")
UV_WORKFLOWS = (
    Path(".github/workflows/tests.yml"),
    Path(".github/workflows/post-merge.yml"),
    Path(".github/workflows/version-policy.yml"),
)
REQUIRED_TEST_RUNNERS = {
    "Linux": "ubuntu-latest",
    "Windows": "windows-latest",
    "macOS": "macos-latest",
}


@pytest.mark.parametrize(
    "requirement, valid",
    [
        ('"==3.14.7"', True),
        ("'==3.14.7'", True),
        ('"==3.14.7" # Python runtime', True),
        ('">=3.14.7"', False),
    ],
)
def test_ci_python_identity_parses_toml(
    tmp_path: Path, requirement: str, valid: bool
) -> None:
    setup = yaml.safe_load(CI_SETUP.read_text(encoding="utf-8"))
    step = next(step for step in setup["runs"]["steps"] if step.get("id") == "identity")
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nrequires-python = {requirement}\n", encoding="utf-8"
    )
    bash = shutil.which("bash")
    if sys.platform == "win32":
        git = shutil.which("git")
        assert git is not None
        bash = str(Path(git).resolve().parent.parent / "bin" / "bash.exe")
    assert bash is not None
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "CI_TEMP": tmp_path.as_posix(),
        "CI_RUNNER": "test-runner",
        "CI_OS": "test-os",
        "CI_ARCH": "X64",
        "CI_UV": UV_MINIMUM,
        "GITHUB_ENV": "github-env",
        "GITHUB_OUTPUT": "github-output",
    }
    result = subprocess.run(
        [bash, "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if valid:
        assert result.returncode == 0, result.stderr
        output = (tmp_path / "github-output").read_text(encoding="utf-8")
        assert "python-version=3.14.7\n" in output
        assert f"py-3.14.7-uv-{UV_MINIMUM}" in output
    else:
        assert result.returncode != 0
        assert not (tmp_path / "github-output").exists()


def test_installer_python_requests_match_package_requirement() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    requirement = project["project"]["requires-python"]
    match = re.fullmatch(r"==([0-9]+\.[0-9]+\.[0-9]+)", requirement)
    assert match is not None
    version = match[1]
    shell = Path("scripts/install.sh").read_text(encoding="utf-8")
    powershell = Path("scripts/install.ps1").read_text(encoding="utf-8")
    shell_pin = re.search(r'^PYTHON_VERSION="([^"]+)"', shell, re.MULTILINE)
    windows_pin = re.search(r'^\$PythonRequest = "([^"]+)"', powershell, re.MULTILINE)
    assert shell_pin is not None and windows_pin is not None
    assert shell_pin[1] == version
    assert windows_pin[1] == f"cpython-{version}-windows-x86_64-none"
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"] == requirement


def _assert_required_test_matrix(workflow: dict, suite: str) -> None:
    job = workflow["jobs"][suite]
    assert job["name"] == f"{suite} (${{{{ matrix.os }}}})"
    assert job["runs-on"] == "${{ matrix.runner }}"
    matrix = job["strategy"]["matrix"]
    assert not matrix.get("exclude")
    entries = matrix["include"]
    assert len(entries) == len(REQUIRED_TEST_RUNNERS)
    assert {(entry["os"], entry["runner"]) for entry in entries} == set(
        REQUIRED_TEST_RUNNERS.items()
    )


@pytest.mark.parametrize("suite", ("pytest", "playwright"))
def test_required_test_matrix_covers_supported_platforms(suite: str) -> None:
    workflow = yaml.safe_load(UV_WORKFLOWS[0].read_text(encoding="utf-8"))
    _assert_required_test_matrix(workflow, suite)


@pytest.mark.parametrize("suite", ("pytest", "playwright"))
@pytest.mark.parametrize("os_name", REQUIRED_TEST_RUNNERS)
def test_required_test_matrix_rejects_missing_platform(
    suite: str, os_name: str
) -> None:
    workflow = yaml.safe_load(UV_WORKFLOWS[0].read_text(encoding="utf-8"))
    entries = workflow["jobs"][suite]["strategy"]["matrix"]["include"]
    entries[:] = [entry for entry in entries if entry["os"] != os_name]
    with pytest.raises(AssertionError):
        _assert_required_test_matrix(workflow, suite)


def test_supported_uv_minimum_is_consistent() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    install_sh = Path("scripts/install.sh").read_text(encoding="utf-8")
    install_ps1 = Path("scripts/install.ps1").read_text(encoding="utf-8")

    assert pyproject["tool"]["uv"]["required-version"] == f">={UV_MINIMUM}"
    assert f'MIN_UV_VERSION="{UV_MINIMUM}"' in install_sh
    assert f'$MinUvVersion = "{UV_MINIMUM}"' in install_ps1
    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert f'CI_UV_VERSION: "{UV_MINIMUM}"' in workflow
        assert "uv-version: ${{ env.CI_UV_VERSION }}" in workflow


def test_every_uv_workflow_inherits_malware_check() -> None:
    malware_policy = '  UV_MALWARE_CHECK: "1"\n  UV_PREVIEW_FEATURES: "malware-check"\n'

    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert workflow.count(malware_policy) == 1
        assert workflow.index(malware_policy) < workflow.index("jobs:\n")


def test_only_trusted_main_workflow_writes_caches() -> None:
    pull_request_workflow = UV_WORKFLOWS[0].read_text(encoding="utf-8")
    main_workflow = UV_WORKFLOWS[1].read_text(encoding="utf-8")

    assert "actions/cache/save@" not in pull_request_workflow
    assert "actions/cache/save@" not in CI_SETUP.read_text(encoding="utf-8")
    assert main_workflow.count("actions/cache/save@") == 2
    assert "if: github.ref == 'refs/heads/main'" in main_workflow


def test_uv_workflows_share_managed_python_cache_policy() -> None:
    setup = CI_SETUP.read_text(encoding="utf-8")
    assert "UV_PYTHON_PREFERENCE=only-managed" in setup
    assert "runner.temp" in setup
    assert "sha256sum" not in setup
    assert "--no-emit-workspace" in setup
    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "uses: ./.github/actions/ci-environment" in workflow
        assert "actions/setup-python@" not in workflow
        assert "/tmp/" not in workflow
        assert "uv-deps-v" not in workflow
        assert "uv-python-v" not in workflow


def test_shared_setup_installs_dependencies_except_for_post_merge_jobs() -> None:
    setup = yaml.safe_load(CI_SETUP.read_text(encoding="utf-8"))
    assert setup["inputs"]["sync-dependencies"]["default"] == "true"
    sync = next(
        step
        for step in setup["runs"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    assert sync["run"] == "uv sync --locked --group dev"
    assert sync["if"] == "inputs.sync-dependencies == 'true'"
    for path in UV_WORKFLOWS:
        jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
        for job in jobs.values():
            for step in job["steps"]:
                if step.get("uses") == "./.github/actions/ci-environment":
                    assert step["with"].get("sync-dependencies", "true") == (
                        "false" if path.name == "post-merge.yml" else "true"
                    )
                assert step.get("run") != "uv sync --locked --group dev"
