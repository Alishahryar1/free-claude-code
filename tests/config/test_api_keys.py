"""Ordered credentials keep environment ownership and select only the first key."""

import json

import pytest
from pydantic import SecretStr, ValidationError

from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.env_migrations import quote_env_value
from free_claude_code.config.loader import (
    ConfigSource,
    ManagedConfigStore,
    compose_settings_snapshot,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.runtime.config import (
    build_custom_provider_config,
    build_provider_config,
)


@pytest.mark.parametrize(
    "descriptor",
    [provider for provider in PROVIDER_CATALOG.values() if provider.credential_env],
)
def test_every_catalog_owner_loads_an_ordered_secret_collection(descriptor):
    settings = Settings.model_validate(
        {descriptor.credential_env: '["primary", "extra"]'}
    )
    assert getattr(settings, descriptor.credential_attr) == (
        SecretStr("primary"),
        SecretStr("extra"),
    )


@pytest.mark.parametrize("value", [" first ", '[" first "]'])
def test_single_key_input_normalizes_to_one_secret(value):
    settings = Settings(OPENAI_API_KEY=value)
    assert settings.openai_api_keys == (SecretStr("first"),)
    assert "first" not in repr(settings)


def test_array_keeps_order_and_selects_only_first_at_provider_boundary():
    settings = Settings(OPENAI_API_KEY='["first", "second", "third"]')
    assert settings.openai_api_keys == tuple(
        map(SecretStr, ("first", "second", "third"))
    )
    assert (
        build_provider_config(PROVIDER_CATALOG["openai_api"], settings).api_key
        == "first"
    )


@pytest.mark.parametrize(
    "value", ['["first", ""]', '["first", " first "]', '["first", 3]', '["first"']
)
def test_invalid_array_is_rejected_without_disclosing_inputs(value):
    with pytest.raises(ValidationError) as error:
        Settings(OPENAI_API_KEY=value)
    assert "first" not in str(error.value)


@pytest.mark.parametrize("value", ["", "[]"])
def test_empty_process_collection_overrides_managed_keys(value):
    snapshot = compose_settings_snapshot(
        {"OPENAI_API_KEY": '["managed"]'}, {"OPENAI_API_KEY": value}
    )
    assert snapshot.settings.openai_api_keys == ()
    assert snapshot.sources["openai_api_keys"] is ConfigSource.PROCESS


@pytest.mark.parametrize(
    "providers", [("zai", "zai_api"), ("opencode_zen", "opencode_go")]
)
def test_shared_providers_use_one_ordered_owner(providers):
    first, second = (PROVIDER_CATALOG[provider] for provider in providers)
    assert first.credential_env == second.credential_env
    settings = Settings.model_validate({first.credential_env: '["primary", "extra"]'})
    assert build_provider_config(first, settings).api_key == "primary"
    assert build_provider_config(second, settings).api_key == "primary"


@pytest.mark.parametrize(
    "credential", [{"api_key": " first "}, {"api_keys": [" first ", "second"]}]
)
def test_custom_input_normalizes_and_uses_first(credential):
    settings = Settings(
        custom_providers=[
            {
                "provider_id": "custom_" + "1" * 32,
                "display_name": "Example",
                "base_url": "https://custom.example/v1",
                **credential,
            }
        ]
    )
    definition = settings.custom_providers[0]
    assert definition.api_keys[0] == SecretStr("first")
    assert build_custom_provider_config(definition, settings).api_key == "first"
    assert "api_key" not in definition.private_values()
    assert definition.private_values()["api_keys"][0] == "first"


def test_schema_one_upgrade_preserves_custom_order_and_unknown_values():
    store = ManagedConfigStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    custom = [
        {
            "provider_id": "custom_" + "1" * 32,
            "display_name": "Example",
            "base_url": "https://custom.example",
            "api_key": "custom-secret",
        }
    ]
    store.path.write_text(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_API_KEY=old-secret\nUNKNOWN_PRIVATE=preserve-me\n"
        + "FCC_CUSTOM_PROVIDERS="
        + quote_env_value(json.dumps(custom))
        + "\n",
        encoding="utf-8",
    )
    store.initialize({})
    values = dotenv_values_from_file(store.path)
    assert values["FCC_CONFIG_SCHEMA"] == "2"
    assert json.loads(values["OPENAI_API_KEY"]) == ["old-secret"]
    assert json.loads(values["FCC_CUSTOM_PROVIDERS"])[0]["api_keys"] == [
        "custom-secret"
    ]
    assert values["UNKNOWN_PRIVATE"] == "preserve-me"
    before = store.path.read_bytes(), store.path.stat().st_mtime_ns
    store.initialize({})
    assert (store.path.read_bytes(), store.path.stat().st_mtime_ns) == before


def _old_store(values, schema="1"):
    store = ManagedConfigStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"FCC_CONFIG_SCHEMA={schema}"] if schema else []
    lines.extend(f"{key}={quote_env_value(value)}" for key, value in values.items())
    store.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return store


@pytest.mark.parametrize("schema", ["1", None])
@pytest.mark.parametrize(
    "stored", ['["saved", ""]', '["saved", "saved"]', '["saved", 3]', '["saved"']
)
@pytest.mark.parametrize(
    "override,expected",
    [
        ("", ()),
        ("[]", ()),
        (
            '["process-first","process-second"]',
            (SecretStr("process-first"), SecretStr("process-second")),
        ),
    ],
)
def test_upgrade_preserves_shadowed_invalid_credential_list(
    schema, stored, override, expected
):
    store = _old_store(
        {
            "OPENAI_API_KEY": stored,
            "GROQ_API_KEY": "managed-good",
            "UNKNOWN_PRIVATE": "preserved",
        },
        schema,
    )
    process = {"OPENAI_API_KEY": override}
    store.initialize(process)
    snapshot = store.read(process)
    assert snapshot.sources["openai_api_keys"] is ConfigSource.PROCESS
    assert snapshot.settings.openai_api_keys == expected
    assert snapshot.managed["OPENAI_API_KEY"] == stored
    assert json.loads(snapshot.managed["GROQ_API_KEY"]) == ["managed-good"]
    assert snapshot.managed["UNKNOWN_PRIVATE"] == "preserved"
    assert snapshot.managed["FCC_CONFIG_SCHEMA"] == "2"
    before = store.path.read_bytes(), store.path.stat().st_mtime_ns
    store.initialize(process)
    assert (store.path.read_bytes(), store.path.stat().st_mtime_ns) == before
    with pytest.raises(ValidationError):
        store.read({})


@pytest.mark.parametrize(
    "saved,converted",
    [
        ([{}], [{}]),
        (
            [{"api_key": "managed-secret", "unknown": "preserved"}],
            [{"api_keys": ["managed-secret"], "unknown": "preserved"}],
        ),
    ],
)
def test_upgrade_converts_custom_credentials_before_effective_definition_validation(
    saved, converted
):
    store = _old_store({"FCC_CUSTOM_PROVIDERS": json.dumps(saved)})
    process = {"FCC_CUSTOM_PROVIDERS": "[]"}
    store.initialize(process)
    snapshot = store.read(process)
    assert snapshot.settings.custom_providers == ()
    assert snapshot.sources["custom_providers"] is ConfigSource.PROCESS
    assert json.loads(snapshot.managed["FCC_CUSTOM_PROVIDERS"]) == converted
    assert snapshot.managed["FCC_CONFIG_SCHEMA"] == "2"
    with pytest.raises(ValidationError):
        store.read({})


def test_upgrade_preserves_custom_noncredential_members():
    store = _old_store(
        {
            "FCC_CUSTOM_PROVIDERS": json.dumps(
                [
                    {
                        "provider_id": "custom_" + "1" * 32,
                        "display_name": "  Name  ",
                        "base_url": "https://custom.example/v1/",
                        "model_ids": [" raw ", " raw "],
                        "api_key": " first ",
                    }
                ]
            )
        }
    )
    store.initialize({})
    snapshot = store.read({})
    assert json.loads(snapshot.managed["FCC_CUSTOM_PROVIDERS"]) == [
        {
            "provider_id": "custom_" + "1" * 32,
            "display_name": "  Name  ",
            "base_url": "https://custom.example/v1/",
            "model_ids": [" raw ", " raw "],
            "api_keys": ["first"],
        }
    ]
    assert snapshot.settings.custom_providers[0].display_name == "Name"
    assert snapshot.settings.custom_providers[0].model_ids == ("raw",)


def test_upgrade_preserves_conflicting_custom_credential_forms_for_validation():
    saved = [{"api_key": "old-secret", "api_keys": ["new-secret"]}]
    store = _old_store({"FCC_CUSTOM_PROVIDERS": json.dumps(saved)})
    process = {"FCC_CUSTOM_PROVIDERS": "[]"}
    store.initialize(process)
    snapshot = store.read(process)
    assert json.loads(snapshot.managed["FCC_CUSTOM_PROVIDERS"]) == saved
    assert snapshot.settings.custom_providers == ()
    with pytest.raises(ValidationError):
        store.read({})


@pytest.mark.parametrize(
    "stored",
    [
        "{",
        '{"invalid":"root"}',
        '[{"api_key":3}]',
        '[{"api_keys":["duplicate","duplicate"]}]',
    ],
)
def test_upgrade_preserves_unconvertible_custom_owner_when_shadowed(stored):
    store = _old_store({"FCC_CUSTOM_PROVIDERS": stored})
    process = {"FCC_CUSTOM_PROVIDERS": "[]"}
    store.initialize(process)
    snapshot = store.read(process)
    assert snapshot.managed["FCC_CUSTOM_PROVIDERS"] == stored
    assert snapshot.settings.custom_providers == ()
    with pytest.raises(ValidationError):
        store.read({})


@pytest.mark.parametrize(
    "key,value",
    [
        ("OPENAI_API_KEY", '["", "saved"]'),
        ("FCC_CUSTOM_PROVIDERS", "{"),
        ("FCC_CUSTOM_PROVIDERS", '[{"api_keys":[3]}]'),
    ],
)
def test_unshadowed_credential_conversion_failure_preserves_original_file(key, value):
    store = _old_store({key: value, "GROQ_API_KEY": "managed-good"})
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.initialize({})
    assert store.path.read_bytes() == before
    assert list(store.path.parent.glob("*.tmp")) == []


def test_upgrade_converts_valid_stored_credential_even_when_shadowed():
    store = _old_store({"OPENAI_API_KEY": " managed-first "})
    process = {"OPENAI_API_KEY": "[]"}
    store.initialize(process)
    snapshot = store.read(process)
    assert json.loads(snapshot.managed["OPENAI_API_KEY"]) == ["managed-first"]
    assert snapshot.settings.openai_api_keys == ()
