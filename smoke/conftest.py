from collections.abc import Iterator
from typing import Any

import pytest

from smoke.lib.config import ProviderModel, SmokeConfig, auth_headers
from smoke.lib.report import SmokeReport
from smoke.lib.server import RunningServer, start_server

DISABLED_PROVIDER_MODEL = ProviderModel(
    provider="smoke_disabled",
    full_model="smoke_disabled/smoke-disabled",
    source="smoke_disabled",
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "provider_model" in metafunc.fixturenames:
        config = SmokeConfig.load()
        metafunc.parametrize("provider_model", provider_model_params(config))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if SmokeConfig.load().live:
        return
    skip = pytest.mark.skip(reason="set FCC_LIVE_SMOKE=1 to run local smoke tests")
    for item in items:
        item.add_marker(skip)


def pytest_configure(config: pytest.Config) -> None:
    global _REPORT
    smoke_config = SmokeConfig.load()
    _REPORT = SmokeReport(smoke_config)


def pytest_runtest_setup(item: pytest.Item) -> None:
    config = SmokeConfig.load()
    target_marks = list(item.iter_markers("smoke_target"))
    if not target_marks:
        return
    targets = [str(mark.args[0]) for mark in target_marks if mark.args]
    if targets and not any(config.target_enabled(target) for target in targets):
        pytest.skip(f"smoke target disabled: {', '.join(targets)}")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "setup" and not report.skipped:
        return
    if report.when == "teardown" and not report.failed:
        return
    if _REPORT is None:
        return
    markers = sorted(
        str(name) for name in report.keywords if str(name).startswith("smoke_")
    )
    detail = "" if report.longrepr is None else str(report.longrepr)
    _REPORT.add(
        nodeid=report.nodeid,
        outcome=report.outcome,
        duration_s=report.duration,
        markers=markers,
        detail=detail,
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _REPORT is not None:
        _REPORT.write()


@pytest.fixture(scope="session")
def smoke_config() -> SmokeConfig:
    return SmokeConfig.load()


@pytest.fixture
def smoke_server(smoke_config: SmokeConfig) -> Iterator[RunningServer]:
    with start_server(smoke_config) as server:
        yield server


@pytest.fixture
def smoke_headers() -> dict[str, str]:
    return auth_headers()


def provider_model_params(config: SmokeConfig) -> list[Any]:
    """Return provider params grouped for pytest-xdist ``--dist=loadgroup``."""
    if not config.live:
        return [_disabled_provider_param("set FCC_LIVE_SMOKE=1 to run provider smoke")]

    models = config.provider_smoke_models()
    if not models:
        return [_disabled_provider_param("missing_env: no configured provider smoke")]

    return [
        pytest.param(
            model,
            id=provider_model_id(model),
            marks=pytest.mark.xdist_group(provider_xdist_group(model)),
        )
        for model in models
    ]


def _disabled_provider_param(reason: str) -> Any:
    return pytest.param(
        DISABLED_PROVIDER_MODEL,
        id=provider_model_id(DISABLED_PROVIDER_MODEL),
        marks=(
            pytest.mark.skip(reason=reason),
            pytest.mark.xdist_group(provider_xdist_group(DISABLED_PROVIDER_MODEL)),
        ),
    )


def provider_model_id(provider_model: ProviderModel) -> str:
    return provider_model.provider


def provider_xdist_group(provider_model: ProviderModel) -> str:
    return f"provider:{provider_model.provider}"


_REPORT: SmokeReport | None = None
