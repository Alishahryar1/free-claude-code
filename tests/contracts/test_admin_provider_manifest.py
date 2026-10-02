"""Ensure admin UI manifest exposes every catalog credential/proxy binding."""

from free_claude_code.config.admin.manifest import FIELD_BY_KEY, FIELDS
from free_claude_code.config.admin.state import ConfigValueState
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject


def _test_value(value: str) -> ConfigValueState:
    return ConfigValueState(value=value, source="test")


def test_provider_catalog_remote_credentials_in_admin_manifest() -> None:
    missing: list[str] = []
    wrong_attr: list[str] = []

    for provider_id, desc in PROVIDER_CATALOG.items():
        if desc.credential_env is None:
            continue
        if desc.credential_attr is None:
            missing.append(
                f"{provider_id}: credential_env set but credential_attr missing"
            )
            continue
        entry = FIELD_BY_KEY.get(desc.credential_env)
        if entry is None:
            missing.append(
                f"{provider_id}: {desc.credential_env} not in admin FIELD_BY_KEY"
            )
            continue
        if entry.settings_attr != desc.credential_attr:
            wrong_attr.append(
                f"{provider_id}: {desc.credential_env} maps settings_attr="
                f"{entry.settings_attr!r}, catalog expects "
                f"{desc.credential_attr!r}"
            )

    assert not missing and not wrong_attr, "\n".join(missing + wrong_attr)


def test_provider_catalog_base_urls_in_admin_manifest() -> None:
    missing_key: list[str] = []
    wrong_attr: list[str] = []

    for provider_id, desc in PROVIDER_CATALOG.items():
        if desc.base_url_attr is None:
            continue
        mf = Settings.model_fields[desc.base_url_attr]
        alias = mf.validation_alias
        if alias is None:
            missing_key.append(
                f"{provider_id}: {desc.base_url_attr} has no validation_alias "
                "(admin manifest expects env-backed base URL)"
            )
            continue
        env_key = str(alias)
        entry = FIELD_BY_KEY.get(env_key)
        if entry is None:
            missing_key.append(
                f"{provider_id}: base URL env {env_key} not in FIELD_BY_KEY"
            )
            continue
        if entry.settings_attr != desc.base_url_attr:
            wrong_attr.append(
                f"{provider_id}: {env_key} maps settings_attr="
                f"{entry.settings_attr!r}, catalog expects {desc.base_url_attr!r}"
            )

    assert not missing_key and not wrong_attr, "\n".join(missing_key + wrong_attr)


def test_provider_catalog_proxy_attrs_in_admin_manifest() -> None:
    missing_key: list[str] = []
    wrong_attr: list[str] = []

    for provider_id, desc in PROVIDER_CATALOG.items():
        if desc.proxy_attr is None:
            continue
        mf = Settings.model_fields[desc.proxy_attr]
        alias = mf.validation_alias
        if alias is None:
            missing_key.append(
                f"{provider_id}: {desc.proxy_attr} has no validation_alias "
                "(admin manifest expects env-backed proxy)"
            )
            continue
        env_key = str(alias)
        entry = FIELD_BY_KEY.get(env_key)
        if entry is None:
            missing_key.append(
                f"{provider_id}: proxy env {env_key} not in FIELD_BY_KEY"
            )
            continue
        if entry.settings_attr != desc.proxy_attr:
            wrong_attr.append(
                f"{provider_id}: {env_key} maps settings_attr="
                f"{entry.settings_attr!r}, catalog expects {desc.proxy_attr!r}"
            )

    assert not missing_key and not wrong_attr, "\n".join(missing_key + wrong_attr)


def test_openai_proxy_override_applies_to_catalog_proxy_field() -> None:
    entry = FIELD_BY_KEY["OPENAI_PROXY"]

    assert entry.settings_attr == "openai_proxy"
    assert entry.restart_required is True
    assert "restarts FCC" in entry.description


def test_provider_catalog_display_names_are_admin_status_source() -> None:
    from free_claude_code.config.admin.status import provider_config_status
    from free_claude_code.config.admin.values import load_value_state

    status_by_provider = {
        entry["provider_id"]: entry
        for entry in provider_config_status(load_value_state())
    }

    assert set(status_by_provider) == set(PROVIDER_CATALOG)
    for provider_id, desc in PROVIDER_CATALOG.items():
        assert status_by_provider[provider_id]["display_name"] == desc.display_name
        expected_kind = (
            "connected_account"
            if desc.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT
            else "local"
            if desc.local
            else "remote"
        )
        assert status_by_provider[provider_id]["kind"] == expected_kind


def test_cloudflare_account_id_is_admin_provider_field() -> None:
    entry = FIELD_BY_KEY["CLOUDFLARE_ACCOUNT_ID"]

    assert entry.settings_attr == "cloudflare_account_id"
    assert entry.section_id == "providers"
    assert entry.secret is False


def test_vertex_project_and_location_are_admin_provider_fields() -> None:
    project = FIELD_BY_KEY["VERTEX_PROJECT_ID"]
    location = FIELD_BY_KEY["VERTEX_LOCATION"]

    assert project.settings_attr == "vertex_project_id"
    assert project.section_id == "providers"
    assert project.secret is False
    assert location.settings_attr == "vertex_location"
    assert location.resolved_default() == "global"


def test_qwencloud_coding_key_is_a_distinct_admin_provider_field() -> None:
    entry = FIELD_BY_KEY["QWENCLOUD_CODING_API_KEY"]

    assert entry.label == "QwenCloud Coding Plan API Key"
    assert entry.settings_attr == "qwencloud_coding_api_key"
    assert entry.section_id == "providers"
    assert entry.secret is True
    assert "separate endpoints" in entry.description


def test_cline_pass_admin_fields_use_programmatic_key_and_fixed_endpoint() -> None:
    from free_claude_code.config.admin.status import provider_config_status

    key = FIELD_BY_KEY["CLINE_API_KEY"]
    proxy = FIELD_BY_KEY["CLINE_PASS_PROXY"]
    status = next(
        item
        for item in provider_config_status(
            {"CLINE_API_KEY": _test_value("cline-programmatic-key")}
        )
        if item["provider_id"] == "cline_pass"
    )

    assert key.label == "Cline API Key"
    assert key.settings_attr == "cline_api_key"
    assert key.section_id == "providers"
    assert key.secret is True
    assert "Subscribe to ClinePass" in key.description
    assert "Settings > API Keys" in key.description
    assert "not the Cline CLI's managed account token" in key.description
    assert proxy.settings_attr == "cline_pass_proxy"
    assert proxy.secret is True
    assert "CLINE_BASE_URL" not in FIELD_BY_KEY
    assert status["display_name"] == "ClinePass"
    assert status["status"] == "configured"


def test_zai_shared_key_configures_both_distinct_provider_surfaces() -> None:
    from free_claude_code.config.admin.status import provider_config_status

    entry = FIELD_BY_KEY["ZAI_API_KEY"]
    statuses = {
        status["provider_id"]: status
        for status in provider_config_status(
            {"ZAI_API_KEY": _test_value("shared-zai-key")}
        )
    }

    assert sum(field.key == "ZAI_API_KEY" for field in FIELDS) == 1
    assert entry.settings_attr == "zai_api_key"
    assert "Coding Plan" in entry.description
    assert "pay-as-you-go" in entry.description
    assert statuses["zai"]["display_name"] == "Z.ai Coding Plan"
    assert statuses["zai"]["status"] == "configured"
    assert statuses["zai_api"]["display_name"] == "Z.ai API"
    assert statuses["zai_api"]["status"] == "configured"
    assert FIELD_BY_KEY["ZAI_API_PROXY"].settings_attr == "zai_api_proxy"


def test_vertex_admin_status_uses_project_configuration_not_an_api_key() -> None:
    from free_claude_code.config.admin.status import provider_config_status

    def vertex_status(project_id: str) -> JsonObject:
        statuses = provider_config_status(
            {
                "VERTEX_PROJECT_ID": _test_value(project_id),
                "VERTEX_LOCATION": _test_value("global"),
            }
        )
        return next(status for status in statuses if status["provider_id"] == "vertex")

    assert vertex_status("")["status"] == "missing_config"
    assert vertex_status("")["label"] == "Missing configuration"
    assert vertex_status("")["configuration"] == "VERTEX_PROJECT_ID"
    assert vertex_status("vertex-project")["status"] == "configured"


def test_azure_openai_admin_status_distinguishes_key_and_url() -> None:
    from free_claude_code.config.admin.status import provider_config_status

    def azure_status(api_key: str, base_url: str) -> JsonObject:
        statuses = provider_config_status(
            {
                "AZURE_OPENAI_API_KEY": _test_value(api_key),
                "AZURE_OPENAI_BASE_URL": _test_value(base_url),
            }
        )
        return next(
            status for status in statuses if status["provider_id"] == "azure_openai"
        )

    missing_key = azure_status(
        "",
        "https://resource.openai.azure.com/openai/v1/",
    )
    assert missing_key["status"] == "missing_key"
    assert missing_key["label"] == "Missing key"

    missing_url = azure_status("azure-key", "")
    assert missing_url["status"] == "missing_config"
    assert missing_url["label"] == "Missing configuration"
    assert missing_url["configuration"] == (
        "AZURE_OPENAI_API_KEY + AZURE_OPENAI_BASE_URL"
    )

    assert (
        azure_status(
            "azure-key",
            "https://resource.openai.azure.com/openai/v1/",
        )["status"]
        == "configured"
    )
