from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.state import ConfigValueState
from free_claude_code.config.admin.status import provider_config_status
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.core.json_types import JsonObject


def _value(value: str | None) -> ConfigValueState:
    return ConfigValueState(value=value, source="test")


def _provider_status(
    provider_id: str,
    state: dict[str, ConfigValueState],
) -> JsonObject:
    return next(
        status
        for status in provider_config_status(state)
        if status["provider_id"] == provider_id
    )


def test_remote_status_exposes_ordered_configuration_targets() -> None:
    missing = _provider_status(
        "nvidia_nim",
        {"NVIDIA_NIM_API_KEY": _value(None)},
    )
    configured = _provider_status(
        "nvidia_nim",
        {"NVIDIA_NIM_API_KEY": _value("configured")},
    )

    assert missing == {
        "provider_id": "nvidia_nim",
        "display_name": "NVIDIA NIM",
        "website_url": "https://build.nvidia.com/",
        "logo_filename": "nvidia-color.svg",
        "kind": "remote",
        "status": "missing_key",
        "label": "Missing key",
        "configuration_keys": ["NVIDIA_NIM_API_KEY"],
        "missing_configuration_keys": ["NVIDIA_NIM_API_KEY"],
        "settings_keys": ["NVIDIA_NIM_API_KEY", "NVIDIA_NIM_PROXY"],
    }
    assert configured["status"] == "configured"
    assert configured["label"] == "Configured"
    assert configured["configuration_keys"] == ["NVIDIA_NIM_API_KEY"]
    assert configured["missing_configuration_keys"] == []
    assert "configuration" not in configured


def test_multi_field_status_preserves_catalog_order_and_first_missing_target() -> None:
    status = _provider_status(
        "cloudflare",
        {
            "CLOUDFLARE_API_TOKEN": _value("configured"),
            "CLOUDFLARE_ACCOUNT_ID": _value(None),
        },
    )

    assert status["status"] == "missing_config"
    assert status["configuration_keys"] == [
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
    ]
    assert status["missing_configuration_keys"] == ["CLOUDFLARE_ACCOUNT_ID"]


def test_local_status_keeps_configuration_separate_from_reachability() -> None:
    missing = _provider_status(
        "lmstudio",
        {"LM_STUDIO_BASE_URL": _value(None)},
    )
    configured = _provider_status(
        "lmstudio",
        {"LM_STUDIO_BASE_URL": _value("http://127.0.0.1:1234/v1")},
    )

    assert missing["status"] == "missing_url"
    assert missing["label"] == "Missing URL"
    assert missing["missing_configuration_keys"] == ["LM_STUDIO_BASE_URL"]
    assert configured["status"] == "configured"
    assert configured["label"] == "Configured"
    assert configured["configuration_keys"] == ["LM_STUDIO_BASE_URL"]
    assert configured["missing_configuration_keys"] == []


def test_connected_accounts_have_no_configuration_navigation_contract() -> None:
    status = _provider_status("openai", {})

    assert status["kind"] == "connected_account"
    assert "configuration_keys" not in status
    assert "missing_configuration_keys" not in status


def test_every_catalog_configuration_attribute_has_an_admin_field() -> None:
    for descriptor in PROVIDER_CATALOG.values():
        for settings_attr in descriptor.configuration_attrs():
            assert any(
                field.settings_attr == settings_attr for field in FIELD_BY_KEY.values()
            ), settings_attr


def test_provider_modals_include_optional_settings_and_shared_credentials() -> None:
    statuses = {status["provider_id"]: status for status in provider_config_status({})}
    assert statuses["azure_openai"]["settings_keys"] == [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_PROXY",
    ]
    assert statuses["vertex"]["settings_keys"] == [
        "VERTEX_PROJECT_ID",
        "VERTEX_LOCATION",
        "VERTEX_PROXY",
    ]
    assert statuses["openai"]["settings_keys"] == ["OPENAI_PROXY"]
    keys_by_provider = {}
    for provider_id, status in statuses.items():
        keys = status["settings_keys"]
        assert isinstance(keys, list)
        keys_by_provider[provider_id] = keys
    for provider_id in ("opencode_zen", "opencode_go"):
        assert "OPENCODE_API_KEY" in keys_by_provider[provider_id]
    for provider_id in ("zai", "zai_api"):
        assert "ZAI_API_KEY" in keys_by_provider[provider_id]
    for field in FIELD_BY_KEY.values():
        owners = [keys for keys in keys_by_provider.values() if field.key in keys]
        assert bool(owners) == (field.section_id == "providers"), field.key
