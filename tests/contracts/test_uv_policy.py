import tomllib
from pathlib import Path

import pytest
import yaml

UV_MINIMUM = "0.12.13"
CI_SETUP = Path(".github/actions/ci-environment/action.yml")
UV_WORKFLOWS = (
    Path(".github/workflows/tests.yml"),
    Path(".github/workflows/post-merge.yml"),
)
REQUIRED_TEST_RUNNERS = {
    "Linux": "ubuntu-latest",
    "Windows": "windows-latest",
    "macOS": "macos-latest",
}


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
