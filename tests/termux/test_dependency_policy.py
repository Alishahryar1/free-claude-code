import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def _selected(requirement_text: str, platform_release: str) -> bool:
    requirement = Requirement(requirement_text)
    if requirement.marker is None:
        return True
    return requirement.marker.evaluate(
        environment={"platform_release": platform_release}
    )


def test_python_policy_accepts_termux_python_patch_releases():
    assert _project()["requires-python"] == ">=3.14,<3.15"


def test_android_does_not_resolve_desktop_server_extras_or_riva():
    dependencies = _project()["dependencies"]
    android_release = "5.10.210-android13-8"

    assert not _selected(
        next(dep for dep in dependencies if "fastapi[standard]" in dep),
        android_release,
    )
    assert _selected(
        next(
            dep
            for dep in dependencies
            if dep.startswith("fastapi>=") and "[standard]" not in dep
        ),
        android_release,
    )
    assert not _selected(
        next(dep for dep in dependencies if dep.startswith("nvidia-riva-client")),
        android_release,
    )
    assert not _selected(
        next(dep for dep in dependencies if dep.startswith("tiktoken")),
        android_release,
    )


def test_non_android_dependency_set_preserves_riva_and_fastapi_extras():
    dependencies = _project()["dependencies"]
    linux_release = "6.8.0-generic"

    assert _selected(
        next(dep for dep in dependencies if "fastapi[standard]" in dep),
        linux_release,
    )
    assert not _selected(
        next(
            dep
            for dep in dependencies
            if dep.startswith("fastapi>=") and "[standard]" not in dep
        ),
        linux_release,
    )
    assert _selected(
        next(dep for dep in dependencies if dep.startswith("nvidia-riva-client")),
        linux_release,
    )
    assert _selected(
        next(dep for dep in dependencies if dep.startswith("tiktoken")),
        linux_release,
    )
