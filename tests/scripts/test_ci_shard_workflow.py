"""Test sharding and required summaries are wired into the CI workflow."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci_shards import shard_count

ROOT = Path(__file__).resolve().parents[2]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())


@pytest.mark.parametrize("suite", ["pytest", "playwright"])
def test_shard_jobs_use_generated_matrices_and_existing_parallelism(suite):
    config = workflow()
    prepare = config["jobs"]["prepare-shards"]
    assert f"{suite}_matrix" in prepare["outputs"]
    generate = next(step for step in prepare["steps"] if step.get("id") == "matrix")
    for os_name in ("linux", "windows", "macos"):
        variable = f"CI_{suite.upper()}_{os_name.upper()}_SHARDS"
        shard_count(config["env"][variable])
        assert f'--{suite}-{os_name}-shards "${variable}"' in generate["run"]
    job = config["jobs"][f"{suite}-shards"]
    assert job["needs"] == "prepare-shards"
    assert job["strategy"]["matrix"] == (
        "${{ fromJSON(needs.prepare-shards.outputs." + suite + "_matrix) }}"
    )
    assert job["strategy"]["fail-fast"] is False
    assert "max-parallel" not in job["strategy"]
    assert "continue-on-error" not in job
    assert job["runs-on"] == "${{ matrix.runner }}"
    assert "${{ matrix.shard }}/${{ matrix.shards }}" in job["name"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("run", "").startswith("uv run --locked pytest ")
    )
    assert "--splits ${{ matrix.shards }}" in command
    assert "--group ${{ matrix.shard }}" in command
    assert "--splitting-algorithm=least_duration" in command
    assert "-n " not in command
    assert "--dist=" not in command
    assert ("pytest e2e " in command) == (suite == "playwright")
    assert "||" not in command


def test_workflow_uses_three_shards_except_two_for_macos_pytest():
    config = workflow()
    assert {
        key: value for key, value in config["env"].items() if key.endswith("_SHARDS")
    } == {
        "CI_PYTEST_LINUX_SHARDS": "3",
        "CI_PYTEST_WINDOWS_SHARDS": "3",
        "CI_PYTEST_MACOS_SHARDS": "2",
        "CI_PLAYWRIGHT_LINUX_SHARDS": "3",
        "CI_PLAYWRIGHT_WINDOWS_SHARDS": "3",
        "CI_PLAYWRIGHT_MACOS_SHARDS": "3",
    }


@pytest.mark.parametrize("suite", ["pytest", "playwright"])
def test_external_timing_path_preserves_config_and_collection_scope(tmp_path, suite):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', encoding="utf-8"
    )
    for directory in ("tests", "e2e", "smoke"):
        path = checkout / directory
        path.mkdir()
        (path / f"test_{directory}.py").write_text(
            'raise RuntimeError("Live smoke must not be collected")\n'
            if directory == "smoke"
            else f"def test_{directory}(): pass\n",
            encoding="utf-8",
        )
    runner_temp = tmp_path / "runner temp"
    timing_path = runner_temp / "fcc-ci-timings/working/durations.json"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_text("{}", encoding="utf-8")
    job = workflow()["jobs"][f"{suite}-shards"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == ("Run pytest" if suite == "pytest" else "Run Playwright")
    )
    command = (
        command.replace("${{ runner.temp }}", runner_temp.as_posix())
        .replace("${{ matrix.shards }}", "1")
        .replace("${{ matrix.shard }}", "1")
    )
    result = subprocess.run(
        [sys.executable, "-m", *shlex.split(command)[3:], "--collect-only"],
        cwd=checkout,
        env={**os.environ, "PYTEST_ADDOPTS": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "configfile: pyproject.toml" in result.stdout
    assert "collected 1 item" in result.stdout
    target = "tests" if suite == "pytest" else "e2e"
    assert f"test_{target}.py" in result.stdout


def test_browser_artifacts_identify_each_shard_and_run_attempt():
    job = workflow()["jobs"]["playwright-shards"]
    upload = next(
        step
        for step in job["steps"]
        if step.get("name") == "Upload Playwright failure artifacts"
    )
    name = upload["with"]["name"]
    for field in (
        "runner.os",
        "runner.arch",
        "matrix.shard",
        "matrix.shards",
        "github.run_id",
        "github.run_attempt",
    ):
        assert "${{ " + field + " }}" in name
    assert upload["if"] == "failure()"
    assert upload["with"]["path"] == "test-results/"
    assert upload["with"]["retention-days"] == 7


def test_timing_preparation_freezes_one_input_and_reuses_it_on_reruns():
    prepare = workflow()["jobs"]["prepare-shards"]
    assert prepare["permissions"] == {"contents": "read", "actions": "read"}
    steps = {step.get("id"): step for step in prepare["steps"]}
    assert steps["timings"]["run"] == "python -m scripts.ci_timings prepare"
    assert (
        steps["timings"]["env"]["CI_PYTEST_MATRIX"]
        == "${{ steps.matrix.outputs.pytest_matrix }}"
    )
    assert (
        steps["timings"]["env"]["CI_PLAYWRIGHT_MATRIX"]
        == "${{ steps.matrix.outputs.playwright_matrix }}"
    )
    freeze = steps["freeze"]
    assert freeze["if"] == "steps.timings.outputs.artifact_id == ''"
    assert freeze["with"]["name"] == "ci-timings-input-v1"
    assert freeze["with"]["retention-days"] == 30
    assert freeze["with"].get("overwrite", False) is False
    assert (
        prepare["outputs"]["timings_input_id"]
        == "${{ steps.timings.outputs.artifact_id || steps.freeze.outputs.artifact-id }}"
    )


@pytest.mark.parametrize("suite", ["pytest", "playwright"])
def test_timing_shards_share_the_frozen_input_and_upload_only_after_success(suite):
    job = workflow()["jobs"][f"{suite}-shards"]
    assert job["permissions"] == {"contents": "read"}
    assert job["env"]["CI_TIMINGS_SUITE"] == suite
    assert (
        job["env"]["CI_TIMINGS_INPUT_ID"]
        == "${{ needs.prepare-shards.outputs.timings_input_id }}"
    )
    steps = job["steps"]
    restore = next(
        step for step in steps if step.get("name") == "Restore frozen timing input"
    )
    assert restore["with"]["artifact-ids"] == job["env"]["CI_TIMINGS_INPUT_ID"]
    command = next(
        step
        for step in steps
        if step.get("run", "").startswith("uv run --locked pytest ")
    )
    assert "--store-durations --clean-durations" in command["run"]
    assert (
        '--durations-path "${{ runner.temp }}/fcc-ci-timings/working/durations.json"'
        in command["run"]
    )
    record = next(step for step in steps if step.get("name") == "Record shard timings")
    upload = next(step for step in steps if step.get("name") == "Upload shard timings")
    assert (
        steps.index(restore)
        < steps.index(command)
        < steps.index(record)
        < steps.index(upload)
    )
    assert (
        "if" not in record and "if" not in upload
    )  # Actions' default success() guard.
    assert "${{ github.run_attempt }}" in upload["with"]["name"]
    assert upload["with"]["retention-days"] == 30
    assert upload["with"]["if-no-files-found"] == "error"


def test_timing_collector_publishes_only_complete_current_attempts():
    job = workflow()["jobs"]["collect-test-timings"]
    assert set(job["needs"]) == {
        "prepare-shards",
        "pytest-summary",
        "playwright-summary",
    }
    assert "if" not in job
    assert job["permissions"] == {"contents": "read"}
    download = next(
        step
        for step in job["steps"]
        if step.get("name") == "Download current-attempt shard timings"
    )
    assert (
        download["with"]["pattern"]
        == "ci-timings-shard-v1-*-attempt-${{ github.run_attempt }}"
    )
    assert not download["with"].get("merge-multiple", False)
    upload = next(
        step for step in job["steps"] if step.get("name") == "Publish complete timings"
    )
    assert upload["if"] == "steps.merge.outputs.complete == 'true'"
    assert upload["with"]["retention-days"] == 30


@pytest.mark.parametrize(
    "suite,preparation,shards",
    [
        (suite, preparation, shards)
        for suite in ("pytest", "playwright")
        for preparation in ("success", "failure", "cancelled", "skipped")
        for shards in ("success", "failure", "cancelled", "skipped")
    ],
)
def test_summary_only_passes_when_preparation_and_all_shards_succeed(
    suite, preparation, shards
):
    job = workflow()["jobs"][f"{suite}-summary"]
    assert job["name"] == suite
    assert set(job["needs"]) == {"prepare-shards", f"{suite}-shards"}
    assert job["if"] == "${{ always() }}"
    assert "continue-on-error" not in job
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    expected_env = {
        "PREPARATION_RESULT": "${{ needs.prepare-shards.result }}",
        "SHARDS_RESULT": "${{ needs." + suite + "-shards.result }}",
    }
    assert step["env"] == expected_env
    bash = shutil.which("bash")
    if os.name == "nt":
        # WSL's bash.exe does not inherit arbitrary Windows environment variables.
        git = shutil.which("git")
        bash = (
            shutil.which("bash", path=str(Path(git).parents[1] / "bin"))
            if git
            else None
        )
    if bash is None:
        pytest.skip("bash is required to execute the GitHub summary check")
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "PREPARATION_RESULT": preparation,
            "SHARDS_RESULT": shards,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode == 0) == (preparation == shards == "success"), (
        result.stderr
    )
