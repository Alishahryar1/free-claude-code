"""Exercise version policy against real, isolated Git history."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "scripts/check_version_policy.py"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
    )
    return result.stdout.strip()


def write(repo: Path, path: str, content: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def version(repo: Path, value: str, lock: str | None = None) -> None:
    write(
        repo,
        "pyproject.toml",
        f'[project]\nname = "free-claude-code"\nversion = "{value}"\n',
    )
    write(
        repo,
        "uv.lock",
        f'[[package]]\nname = "free-claude-code"\nversion = "{lock or value}"\nsource = {{ editable = "." }}\n',
    )


def commit(repo: Path) -> str:
    git(repo, "add", ".")
    git(repo, "commit", "--no-gpg-sign", "-m", "test change")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "core.autocrlf", "false")
    version(tmp_path, "1.2.3")
    write(tmp_path, "src/file.py", "original\n")
    return tmp_path, commit(tmp_path)


def check(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--base", base, "--head", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "path",
    [
        "assets/a.svg",
        "scripts/a.sh",
        "src/a.py",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    ],
)
@pytest.mark.parametrize("bump", [False, True])
def test_release_changes_require_increase(history, path, bump):
    repo, base = history
    if bump:
        version(repo, "1.2.4")
    if path in {"pyproject.toml", "uv.lock"}:
        with (repo / path).open("a", encoding="utf-8") as stream:
            stream.write("# non-version change\n")
    else:
        write(repo, path, "changed\n")
    commit(repo)
    result = check(repo, base)
    assert (result.returncode == 0) is bump, result.stdout + result.stderr
    assert path in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "tests/test_example.py",
        ".github/workflows/ci.yml",
        "scripts-other/a.py",
    ],
)
@pytest.mark.parametrize("bump", [False, True])
def test_excluded_only_changes_forbid_increase(history, path, bump):
    repo, base = history
    if bump:
        version(repo, "1.2.4")
    write(repo, path, "documentation\n")
    commit(repo)
    result = check(repo, base)
    assert (result.returncode == 0) is not bump, result.stdout + result.stderr


def test_version_only_change_does_not_justify_itself(history):
    repo, base = history
    version(repo, "1.2.4")
    commit(repo)
    result = check(repo, base)
    assert result.returncode != 0
    assert "without release changes" in result.stdout


@pytest.mark.parametrize(
    "value,allowed",
    [
        ("1.2.4", True),
        ("1.3.0", True),
        ("2.0.0", True),
        ("1.2.5", False),
        ("1.4.0", False),
        ("3.0.0", False),
        ("1.3.1", False),
        ("2.1.0", False),
        ("2.0.1", False),
        ("1.2.4rc1", False),
        ("1.2.4.post1", False),
        ("1.2.4.dev1", False),
        ("1.2.4+local", False),
        ("1!1.2.4", False),
        ("1.2.4.0", False),
        ("v1.2.4", False),
        ("01.2.4", False),
    ],
)
def test_release_requires_exactly_one_version_increment(history, value, allowed):
    repo, base = history
    version(repo, value)
    write(repo, "src/file.py", "changed\n")
    commit(repo)
    result = check(repo, base)
    assert (result.returncode == 0) is allowed, result.stdout + result.stderr


@pytest.mark.parametrize("final_version,allowed", [("1.2.4", True), ("1.2.5", False)])
def test_multiple_pr_commits_share_one_bump(history, final_version, allowed):
    repo, base = history
    version(repo, "1.2.4")
    write(repo, "src/file.py", "first change\n")
    commit(repo)
    version(repo, final_version)
    write(repo, "src/file.py", "second change\n")
    commit(repo)
    result = check(repo, base)
    assert (result.returncode == 0) is allowed, result.stdout + result.stderr


@pytest.mark.parametrize(
    "value,lock", [("1.2.2", None), ("1.2.4", "1.2.3"), ("invalid", None)]
)
def test_invalid_or_inconsistent_versions_fail(history, value, lock):
    repo, base = history
    version(repo, value, lock)
    write(repo, "src/file.py", "changed\n")
    commit(repo)
    assert check(repo, base).returncode != 0


@pytest.mark.parametrize("move", [False, True])
def test_deleting_or_moving_out_of_release_directory_requires_bump(history, move):
    repo, base = history
    if move:
        (repo / "src/file.py").rename(repo / "file with spaces.py")
    else:
        (repo / "src/file.py").unlink()
    commit(repo)
    result = check(repo, base)
    assert result.returncode != 0
    assert "src/file.py" in result.stdout


def test_other_lock_package_version_is_a_release_change(history):
    repo, _ = history
    with (repo / "uv.lock").open("a", encoding="utf-8") as stream:
        stream.write('[[package]]\nname = "other"\nversion = "1.0"\n')
    base = commit(repo)
    lock = repo / "uv.lock"
    lock.write_text(lock.read_text().replace('version = "1.0"', 'version = "2.0"'))
    commit(repo)
    result = check(repo, base)
    assert result.returncode != 0
    assert "uv.lock" in result.stdout


def test_stale_release_cannot_reuse_new_main_version(history):
    repo, original = history
    version(repo, "1.2.4")
    write(repo, "src/main.py", "main change\n")
    base = commit(repo)
    git(repo, "checkout", "-b", "pr", original)
    version(repo, "1.2.4")
    write(repo, "src/pr.py", "pr change\n")
    commit(repo)
    result = check(repo, base)
    assert result.returncode != 0
    assert "must increase" in result.stdout


def test_unrelated_main_changes_are_not_attributed_to_docs_pr(history):
    repo, original = history
    write(repo, "src/main.py", "main change\n")
    base = commit(repo)
    git(repo, "checkout", "-b", "pr", original)
    write(repo, "README.md", "docs\n")
    commit(repo)
    result = check(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr


def test_checker_reads_commits_not_uncommitted_files(history):
    repo, base = history
    write(repo, "README.md", "docs\n")
    commit(repo)
    version(repo, "9.9.9")
    write(repo, "src/file.py", "uncommitted\n")
    result = check(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
