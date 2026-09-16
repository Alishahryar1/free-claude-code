import tomllib
from pathlib import Path

UV_MINIMUM = "0.12.13"
UV_WORKFLOWS = (
    Path(".github/workflows/tests.yml"),
    Path(".github/workflows/dependency-cache.yml"),
)


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
        assert "version: ${{ env.CI_UV_VERSION }}" in workflow


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
    assert main_workflow.count("actions/cache/save@") == 2


def test_uv_workflows_share_managed_python_cache_policy() -> None:
    python_policy = (
        "  UV_PYTHON_INSTALL_DIR: /tmp/uv-python\n"
        "  UV_PYTHON_PREFERENCE: only-managed\n"
    )

    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert workflow.count(python_policy) == 1
        assert "actions/setup-python@" not in workflow
