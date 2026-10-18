import pytest

from free_claude_code.config.loader import compose_settings_snapshot


@pytest.mark.parametrize(
    "key",
    [
        "FAST_PREFIX_DETECTION",
        "ENABLE_NETWORK_PROBE_MOCK",
        "ENABLE_TITLE_GENERATION_SKIP",
        "ENABLE_SUGGESTION_MODE_SKIP",
        "ENABLE_FILEPATH_EXTRACTION_MOCK",
    ],
)
@pytest.mark.parametrize("source", ["managed", "process", "both"])
def test_retired_request_setting_is_ignored(key, source):
    managed = {key: "true"} if source in {"managed", "both"} else {}
    process = {key: "obsolete-invalid-boolean"} if source in {"process", "both"} else {}

    snapshot = compose_settings_snapshot(managed, process)

    assert key.lower() not in snapshot.settings.model_dump()
    assert key.lower() not in snapshot.sources
