"""Anthropic settings share the standard masked configuration boundary."""

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.harnesses.claude import build_claude_proxy_env
from free_claude_code.providers.runtime.config import (
    build_provider_config,
    has_provider_configuration,
)


def test_save_remove_workspace_optional_and_local_token_independent():
    store = ManagedConfigStore()
    store.initialize({})
    values = dict(store.read({}).managed) | {
        "ANTHROPIC_API_KEY": "upstream-key",
        "ANTHROPIC_WORKSPACE_ID": "wrkspc_test",
        "ANTHROPIC_PROXY": "http://localhost:8888",
        "ANTHROPIC_AUTH_TOKEN": "proxy-token",
    }
    store.commit(values)
    settings = store.read({}).settings
    descriptor = PROVIDER_CATALOG["anthropic"]
    config = build_provider_config(descriptor, settings)
    assert config.api_key == "upstream-key"
    assert config.proxy == "http://localhost:8888"
    assert settings.proxy_auth_token == "proxy-token"
    assert has_provider_configuration(
        descriptor, settings.model_copy(update={"anthropic_workspace_id": None})
    )
    assert FIELD_BY_KEY["ANTHROPIC_API_KEY"].secret
    assert FIELD_BY_KEY["ANTHROPIC_WORKSPACE_ID"].provider_ids == ("anthropic",)
    values.pop("ANTHROPIC_API_KEY")
    store.commit(values)
    assert not has_provider_configuration(descriptor, store.read({}).settings)


def test_claude_launch_does_not_inherit_anthropic_account_credentials():
    env = build_claude_proxy_env(
        proxy_root_url="http://localhost:8082",
        auth_token="proxy-token",
        base_env={
            "ANTHROPIC_API_KEY": "upstream-key",
            "ANTHROPIC_WORKSPACE_ID": "wrkspc_test",
            "ANTHROPIC_AUTH_TOKEN": "stale-token",
            "PATH": "retained",
        },
    )
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_WORKSPACE_ID" not in env
    assert env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert env["PATH"] == "retained"
