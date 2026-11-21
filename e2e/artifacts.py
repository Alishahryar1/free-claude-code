"""Keep failure evidence without serializing artifacts already due for deletion."""

from unittest.mock import Mock

import pytest
from playwright.sync_api import BrowserContext
from pytest_playwright.pytest_playwright import ArtifactsRecorder


def discard_passed_artifacts(
    recorder: ArtifactsRecorder,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> ArtifactsRecorder:
    record_context = recorder.on_will_close_browser_context

    def close_context(context: BrowserContext | Mock):
        report: object = getattr(request.node, "rep_call", None)
        passed: object = getattr(report, "passed", None)
        if (
            passed is True
            and request.config.getoption("--tracing") != "on"
            and request.config.getoption("--screenshot") != "on"
        ):
            # Stop recording without building a ZIP or PNG that the plugin
            # would immediately delete. Its normal video cleanup still runs.
            context.tracing.stop()
        else:
            # A context closed during setup or the test may precede a failure.
            record_context(context)

    monkeypatch.setattr(recorder, "on_will_close_browser_context", close_context)
    return recorder


@pytest.fixture(name="_artifacts_recorder")
def failure_artifacts_recorder(
    _artifacts_recorder: ArtifactsRecorder,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    return discard_passed_artifacts(_artifacts_recorder, request, monkeypatch)
