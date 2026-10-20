"""Keep failure evidence without serializing artifacts already due for deletion."""

import pytest


def discard_passed_artifacts(recorder, request):
    record_context = recorder.on_will_close_browser_context

    def close_context(context):
        report = getattr(request.node, "rep_call", None)
        if (
            report is not None
            and report.passed
            and request.config.getoption("--tracing") != "on"
            and request.config.getoption("--screenshot") != "on"
        ):
            # Stop recording without building a ZIP or PNG that the plugin
            # would immediately delete. Its normal video cleanup still runs.
            context.tracing.stop()
        else:
            # A context closed during setup or the test may precede a failure.
            record_context(context)

    recorder.on_will_close_browser_context = close_context
    return recorder


@pytest.fixture(name="_artifacts_recorder")
def failure_artifacts_recorder(_artifacts_recorder, request):
    return discard_passed_artifacts(_artifacts_recorder, request)
