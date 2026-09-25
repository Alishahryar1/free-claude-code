from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

import pytest

import free_claude_code.core.version as version_module


def test_package_version_uses_installed_distribution_metadata() -> None:
    assert version_module.package_version() == distribution_version("free-claude-code")


def test_package_version_has_explicit_uninstalled_source_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_distribution_name: str) -> str:
        raise PackageNotFoundError("free-claude-code")

    monkeypatch.setattr(version_module, "distribution_version", missing)

    assert version_module.package_version() == "0+unknown"


def test_package_version_does_not_hide_invalid_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid(_distribution_name: str) -> str:
        raise ValueError("invalid metadata")

    monkeypatch.setattr(version_module, "distribution_version", invalid)

    with pytest.raises(ValueError, match="invalid metadata"):
        version_module.package_version()
