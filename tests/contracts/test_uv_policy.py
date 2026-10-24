import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts.ci_shards import matrix

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
    "requirement, uv_requirement, python_version, uv_version",
    [
        ('"==3.14.7"', '">=0.13.5"', "3.14.7", "0.13.5"),
        ("'==3.15.0' # runtime", "'>=0.14.0' # minimum", "3.15.0", "0.14.0"),
        ('">=3.14.7"', '">=0.13.5"', None, None),
        ('"==3.14.7"', '"==0.13.5"', None, None),
        ('"==3.14.7"', '">=0.13.5,<1"', None, None),
    ],
)
def test_ci_python_identity_parses_toml(
    tmp_path: Path,
    requirement: str,
    uv_requirement: str,
    python_version: str | None,
    uv_version: str | None,
) -> None:
    setup = yaml.safe_load(CI_SETUP.read_text(encoding="utf-8"))
    step = next(step for step in setup["runs"]["steps"] if step.get("id") == "identity")
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nrequires-python = {requirement}\n"
        f"[tool.uv]\nrequired-version = {uv_requirement}\n",
        encoding="utf-8",
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
    if python_version is not None:
        assert result.returncode == 0, result.stderr
        output = (tmp_path / "github-output").read_text(encoding="utf-8")
        assert f"python-version={python_version}\n" in output
        assert f"uv-version={uv_version}\n" in output
        assert f"py-{python_version}-uv-{uv_version}" in output
    else:
        assert result.returncode != 0
        assert not (tmp_path / "github-output").exists()


def test_python_lock_matches_package_requirement() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    requirement = project["project"]["requires-python"]
    match = re.fullmatch(r"==([0-9]+\.[0-9]+\.[0-9]+)", requirement)
    assert match is not None
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"] == requirement


def _assert_required_test_matrix(workflow: dict, suite: str, entries: list) -> None:
    job = workflow["jobs"][f"{suite}-shards"]
    assert job["runs-on"] == "${{ matrix.runner }}"
    assert job["strategy"]["matrix"] == (
        "${{ fromJSON(needs.prepare-shards.outputs." + suite + "_matrix) }}"
    )
    count = int(workflow["env"][f"CI_{suite.upper()}_SHARDS"])
    assert len(entries) == len(REQUIRED_TEST_RUNNERS) * count
    assert {(entry["os"], entry["runner"]) for entry in entries} == set(
        REQUIRED_TEST_RUNNERS.items()
    )


@pytest.mark.parametrize("suite", ("pytest", "playwright"))
def test_required_test_matrix_covers_supported_platforms(suite: str) -> None:
    workflow = yaml.safe_load(UV_WORKFLOWS[0].read_text(encoding="utf-8"))
    count = int(workflow["env"][f"CI_{suite.upper()}_SHARDS"])
    _assert_required_test_matrix(workflow, suite, matrix(count)["include"])


@pytest.mark.parametrize("suite", ("pytest", "playwright"))
@pytest.mark.parametrize("os_name", REQUIRED_TEST_RUNNERS)
def test_required_test_matrix_rejects_missing_platform(
    suite: str, os_name: str
) -> None:
    workflow = yaml.safe_load(UV_WORKFLOWS[0].read_text(encoding="utf-8"))
    count = int(workflow["env"][f"CI_{suite.upper()}_SHARDS"])
    entries = matrix(count)["include"]
    entries[:] = [entry for entry in entries if entry["os"] != os_name]
    with pytest.raises(AssertionError):
        _assert_required_test_matrix(workflow, suite, entries)


def test_ci_uses_project_uv_minimum() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert re.fullmatch(
        r">=[0-9]+\.[0-9]+\.[0-9]+", pyproject["tool"]["uv"]["required-version"]
    )
    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "CI_UV_VERSION" not in workflow
        for job in yaml.safe_load(workflow)["jobs"].values():
            for step in job["steps"]:
                if step.get("uses") == "./.github/actions/ci-environment":
                    assert "uv-version" not in step.get("with", {})
    setup = yaml.safe_load(CI_SETUP.read_text(encoding="utf-8"))
    uv_setup = next(
        step for step in setup["runs"]["steps"] if step.get("name") == "Install uv"
    )
    assert uv_setup["with"]["version"] == "${{ steps.identity.outputs.uv-version }}"


def test_every_uv_workflow_inherits_malware_check() -> None:
    malware_policy = '  UV_MALWARE_CHECK: "1"\n  UV_PREVIEW_FEATURES: "malware-check"\n'

    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert workflow.count(malware_policy) == 1
        assert workflow.index(malware_policy) < workflow.index("\njobs:\n")


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


def test_shared_setup_installs_dependencies_for_jobs_running_project_code() -> None:
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
