import os
from pathlib import Path

import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.messaging.platforms.factory import create_messaging_components
from free_claude_code.providers.runtime import build_provider_config
from smoke.lib.child_process import (
    cmd_fcc_server,
    cmd_python_c,
    run_captured_text,
)
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live]


def _isolated_config_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    for field in Settings.model_fields.values():
        alias = field.validation_alias
        if isinstance(alias, str):
            env.pop(alias, None)
    env.pop("FCC_ENV_FILE", None)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return env


def _write_managed_config(home: Path, text: str) -> None:
    path = home / ".fcc" / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"FCC_CONFIG_SCHEMA=1\n{text}", encoding="utf-8")


@pytest.mark.smoke_target("config")
def test_env_precedence_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    home = tmp_path / "home"
    _write_managed_config(
        home,
        'MODEL="open_router/test/model"\nANTHROPIC_AUTH_TOKEN="dotenv-token"\n',
    )
    env = _isolated_config_env(home)
    env["MODEL"] = "nvidia_nim/process-model"
    env["ANTHROPIC_AUTH_TOKEN"] = "process-token"
    script = (
        "from free_claude_code.config.loader import get_settings; "
        "s=get_settings(); "
        "print(s.model); print(s.proxy_auth_token)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == ["nvidia_nim/process-model", "dotenv-token"]


@pytest.mark.smoke_target("config")
def test_removed_env_migration_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    home = tmp_path / "home"
    env_file = tmp_path / "legacy-explicit.env"
    original = 'MODEL="deepseek/imported-once"\n'
    env_file.write_text(original, encoding="utf-8")
    env = _isolated_config_env(home)
    env["FCC_ENV_FILE"] = str(env_file)
    script = (
        "from free_claude_code.config.loader import get_settings; "
        "print(get_settings().model)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "deepseek/imported-once"
    assert env_file.read_text(encoding="utf-8") == original
    managed = home / ".fcc" / ".env"
    first = managed.read_bytes()

    env_file.write_text('MODEL="groq/changed-later"\n', encoding="utf-8")
    second = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "deepseek/imported-once"
    assert managed.read_bytes() == first


@pytest.mark.smoke_target("config")
def test_route_reasoning_config_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    home = tmp_path / "home"
    _write_managed_config(
        home,
        'REASONING_POLICY="off"\n'
        'REASONING_FABLE="high"\n'
        'REASONING_OPUS="client"\n'
        'REASONING_SONNET="inherit"\n'
        'REASONING_HAIKU="off"\n',
    )
    env = _isolated_config_env(home)
    script = (
        "from free_claude_code.application.routing import ModelRouter; "
        "from free_claude_code.config.loader import get_settings; "
        "s=get_settings(); "
        "r=ModelRouter(s); "
        "print(r.resolve('claude-fable-5').reasoning_preference.value); "
        "print(r.resolve('claude-opus-4-20250514').reasoning_preference.value); "
        "print(r.resolve('claude-sonnet-4-20250514').reasoning_preference.value); "
        "print(r.resolve('claude-haiku-4-20250514').reasoning_preference.value); "
        "print(r.resolve('unknown-model').reasoning_preference.value)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "high",
        "client",
        "off",
        "off",
        "off",
    ]


@pytest.mark.smoke_target("config")
def test_proxy_timeout_config_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    home = tmp_path / "home"
    _write_managed_config(
        home,
        'MODEL="open_router/test/model"\n'
        'OPENROUTER_API_KEY="key"\n'
        'OPENROUTER_PROXY="socks5://127.0.0.1:9999"\n'
        'HTTP_READ_TIMEOUT="321"\n'
        'HTTP_CONNECT_TIMEOUT="7"\n'
        'HTTP_WRITE_TIMEOUT="8"\n',
    )
    env = _isolated_config_env(home)
    script = (
        "from free_claude_code.config.loader import get_settings; "
        "from free_claude_code.config.provider_catalog import PROVIDER_CATALOG; "
        "from free_claude_code.providers.runtime import build_provider_config; "
        "s=get_settings(); c=build_provider_config(PROVIDER_CATALOG['open_router'], s); "
        "print(c.proxy); print(c.http_read_timeout); "
        "print(c.http_connect_timeout); print(c.http_write_timeout)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "socks5://127.0.0.1:9999",
        "321.0",
        "7.0",
        "8.0",
    ]


@pytest.mark.smoke_target("extensibility")
def test_provider_runtime_config_e2e() -> None:
    settings_kwargs: dict[str, str] = {}
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_attr is not None:
            settings_kwargs[_settings_init_key(descriptor.credential_attr)] = (
                f"{descriptor.provider_id}-key"
            )
        if descriptor.base_url_attr is not None:
            settings_kwargs[_settings_init_key(descriptor.base_url_attr)] = (
                descriptor.default_base_url
                or f"https://{descriptor.provider_id}.example.invalid/v1"
            )
        for attr in descriptor.required_settings_attrs:
            settings_kwargs.setdefault(
                _settings_init_key(attr),
                f"{descriptor.provider_id}-setting",
            )
    settings = Settings.model_validate(settings_kwargs)
    for descriptor in PROVIDER_CATALOG.values():
        config = build_provider_config(descriptor, settings)
        assert config.base_url
        if descriptor.credential_attr is not None or descriptor.static_credential:
            assert config.api_key
        else:
            assert config.api_key is None


def _settings_init_key(field_name: str) -> str:
    alias = Settings.model_fields[field_name].validation_alias
    return alias if isinstance(alias, str) else field_name


@pytest.mark.smoke_target("extensibility")
def test_platform_factory_e2e() -> None:
    assert create_messaging_components("not-a-platform") is None
    assert create_messaging_components("telegram") is None
    assert create_messaging_components("discord") is None


@pytest.mark.smoke_target("cli")
def test_entrypoint_server_e2e(smoke_config: SmokeConfig) -> None:
    with SmokeServerDriver(
        smoke_config,
        name="product-entrypoint",
        command=cmd_fcc_server(),
        env_overrides={"MESSAGING_PLATFORM": "none"},
    ).run() as server:
        assert server.process.poll() is None
