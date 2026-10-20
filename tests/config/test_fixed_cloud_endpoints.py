import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.providers.runtime.config import build_provider_config
from tests.api.support import create_test_app


@pytest.mark.parametrize(
    "provider_id, retired_key",
    [
        ("tokenrouter", "TOKENROUTER_BASE_URL"),
        ("nararoute", "NARAROUTE_BASE_URL"),
        ("lightning", "LIGHTNING_BASE_URL"),
        ("experiential", "EXPLABS_BASE_URL"),
    ],
)
@pytest.mark.parametrize("source", ["managed", "process", "both"])
def test_retired_url_cannot_override_cloud_endpoint(provider_id, retired_key, source):
    descriptor = PROVIDER_CATALOG[provider_id]
    assert descriptor.credential_env is not None
    managed = {descriptor.credential_env: "test-key"}
    process = {}
    if source in {"managed", "both"}:
        managed[retired_key] = "https://saved-custom.example/v1"
    if source in {"process", "both"}:
        process[retired_key] = "https://process-custom.example/v1"
    settings = compose_settings_snapshot(managed, process).settings
    config = build_provider_config(descriptor, settings)
    assert config.base_url == descriptor.default_base_url
    assert config.api_key == "test-key"


def test_admin_exposes_only_resource_region_and_local_base_urls():
    with TestClient(
        create_test_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        response = client.get("/admin/api/config")
    assert response.status_code == 200
    keys = {field["key"] for field in response.json()["fields"]}
    assert not keys.intersection(
        {
            "TOKENROUTER_BASE_URL",
            "NARAROUTE_BASE_URL",
            "LIGHTNING_BASE_URL",
            "EXPLABS_BASE_URL",
        }
    )
    assert {
        "AZURE_OPENAI_BASE_URL",
        "BEDROCK_BASE_URL",
        "LM_STUDIO_BASE_URL",
        "LLAMACPP_BASE_URL",
        "OLLAMA_BASE_URL",
    } <= keys
