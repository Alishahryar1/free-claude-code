"""The real Playwright recorder retains requested evidence on every close path."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pytest_playwright.pytest_playwright import ArtifactsRecorder

from e2e.artifacts import discard_passed_artifacts


@pytest.mark.parametrize(
    "outcome,tracing,screenshot,zip_count,png_count",
    [
        ("passed", "retain-on-failure", "only-on-failure", 0, 0),
        ("failed", "retain-on-failure", "only-on-failure", 1, 1),
        ("setup-failed", "retain-on-failure", "only-on-failure", 1, 1),
        ("closed-before-failure", "retain-on-failure", "only-on-failure", 1, 1),
        ("passed", "on", "on", 1, 1),
        ("passed", "on", "only-on-failure", 1, 0),
        ("passed", "retain-on-failure", "on", 0, 1),
        ("failed", "off", "off", 0, 0),
    ],
)
def test_artifact_retention(
    tmp_path, outcome, tracing, screenshot, zip_count, png_count
):
    options = {
        "--tracing": tracing,
        "--screenshot": screenshot,
        "--full-page-screenshot": True,
        "--video": "off",
    }
    config = Mock()
    config.getoption.side_effect = options.__getitem__
    node = SimpleNamespace(nodeid="example")
    if outcome in {"passed", "failed"}:
        node.rep_call = SimpleNamespace(passed=outcome == "passed")
    request = Mock(config=config, node=node)
    output = tmp_path / "output"
    context = Mock()
    page = Mock()
    context.pages = [page]

    def save_trace(*, path=None):
        if path is not None:
            Path(path).write_bytes(b"trace")

    def save_screenshot(*, path, **_kwargs):
        Path(path).write_bytes(b"screenshot")

    context.tracing.stop.side_effect = save_trace
    page.screenshot.side_effect = save_screenshot
    staging = TemporaryDirectory(dir=tmp_path)
    with staging:
        recorder = ArtifactsRecorder(config, request, str(output), Mock(), staging)
        recorder = discard_passed_artifacts(recorder, request)
        recorder.on_did_create_browser_context(context)
        recorder.on_will_close_browser_context(context)
        recorder.did_finish_test(failed=outcome != "passed")

        assert len(list(output.glob("*.zip"))) == zip_count
        assert len(list(output.glob("*.png"))) == png_count
        assert list(Path(staging.name).iterdir()) == []
        if outcome == "passed" and tracing != "on" and screenshot != "on":
            context.tracing.stop.assert_called_once_with()
            page.screenshot.assert_not_called()
