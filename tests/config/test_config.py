"""Contracts for pure Settings and canonical source composition."""

from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config import loader
from free_claude_code.config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    HTTP_CONNECT_TIMEOUT_DEFAULT,
)
from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.loader import (
    ConfigSource,
    ManagedConfigStore,
    clear_settings_cache,
    compose_settings_snapshot,
    get_settings,
)
from free_claude_code.config.model_refs import (
    configured_chat_model_refs,
    parse_model_name,
    parse_provider_type,
)
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.paths import (
    managed_env_path,
    messaging_state_dir_path,
    server_log_path,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings


@pytest.mark.parametrize("source", [ConfigSource.MANAGED, ConfigSource.PROCESS])
@pytest.mark.parametrize("tier", ["FABLE", "OPUS", "SONNET", "HAIKU"])
def test_retirement_normalizes_sources_without_resurrecting_overrides(source, tier):
    retired = {
        "MODEL": "github_models/openai/old",
        f"MODEL_{tier}": "github_models/vendor/opus",
        "MODEL_FALLBACKS": "github_models/old",
        f"REASONING_{tier}": "off",
    }
    managed = {
        "MODEL": "groq/managed",
        f"MODEL_{tier}": "groq/tier",
        "MODEL_FALLBACKS": "groq/backup",
    }
    snapshot = compose_settings_snapshot(
        retired if source is ConfigSource.MANAGED else managed,
        retired if source is ConfigSource.PROCESS else {},
    )
    assert snapshot.settings.model == DEFAULT_MODEL
    assert getattr(snapshot.settings, f"model_{tier.lower()}") is None
    assert (
        getattr(snapshot.settings, f"reasoning_{tier.lower()}")
        is ReasoningPreference.OFF
    )
    assert snapshot.settings.model_fallbacks is None
    expected = (
        ConfigSource.PROCESS if source is ConfigSource.PROCESS else ConfigSource.DEFAULT
    )
    assert snapshot.sources["model"] is expected
    assert snapshot.sources[f"model_{tier.lower()}"] is expected
    assert snapshot.sources["model_fallbacks"] is expected
    assert retired["MODEL"] == "github_models/openai/old"


def test_retirement_preserves_effective_default_order_and_process_environment(
    monkeypatch,
):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("MODEL_OPUS", "github_models/opus")
    snapshot = compose_settings_snapshot(
        {
            "MODEL": "groq/default",
            "MODEL_OPUS": "groq/managed-tier",
            "MODEL_FALLBACKS": "groq/first, github_models/old, deepseek/last",
        },
        dict(loader.os.environ),
    )
    assert snapshot.settings.model == "groq/default"
    assert snapshot.settings.model_opus is None
    assert snapshot.sources["model_opus"] is ConfigSource.PROCESS
    assert snapshot.settings.model_fallbacks == ("groq/first", "deepseek/last")
    assert loader.os.environ["MODEL_OPUS"] == "github_models/opus"


@pytest.mark.parametrize(
    "values",
    [
        {"MODEL": "github_models/"},
        {"MODEL": "other/unknown"},
        {"MODEL_FALLBACKS": "github_models/old,groq/a,groq/a"},
        {"MODEL_FALLBACKS": "github_models/old,,groq/a"},
        {"MODEL_FALLBACKS": "github_models/old,"},
        {"MODEL_FALLBACKS": ",github_models/old"},
        {"MODEL_FALLBACKS": "github_models/old, "},
        {"MODEL_FALLBACKS": "github_models/old,github_models/"},
    ],
)
def test_retirement_does_not_hide_invalid_configuration(values):
    with pytest.raises(ValidationError):
        compose_settings_snapshot(values, {})


def test_direct_settings_still_reject_retired_provider():
    with pytest.raises(ValidationError):
        Settings(MODEL="github_models/openai/old")


def test_settings_defaults_are_valid_and_nonempty() -> None:
    settings = Settings()

    assert settings.provider_rate_limit == 1
    assert settings.provider_rate_window == 2
    assert settings.provider_max_concurrency == 2
    assert settings.provider_progress_timeout == 600.0
    assert settings.http_read_timeout == 120.0
    assert settings.http_write_timeout == 10.0
    assert settings.http_connect_timeout == HTTP_CONNECT_TIMEOUT_DEFAULT
    assert settings.voice_note_enabled is True
    assert settings.whisper_device == "cpu"
    assert settings.whisper_model == "base"
    assert settings.enable_web_server_tools is True
    assert settings.proxy_auth_enabled is False
    assert settings.proxy_auth_token == "freecc"
    assert [
        name for name, value in settings if isinstance(value, str) and not value
    ] == []
    assert [
        name
        for name, field in Settings.model_fields.items()
        if field.get_default(call_default_factory=True) == ""
    ] == []


def test_every_external_setting_has_one_explicit_alias() -> None:
    aliases = [
        field.validation_alias
        for name, field in Settings.model_fields.items()
        if name != "nim"
    ]
    assert all(isinstance(alias, str) for alias in aliases)
    assert len(aliases) == len(set(aliases))
    assert Settings.model_fields["nim"].validation_alias is None


def test_direct_settings_construction_performs_no_environment_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL", "deepseek/process-model")
    monkeypatch.setenv("PROVIDER_RATE_LIMIT", "99")

    settings = Settings()

    assert settings.model.startswith("nvidia_nim/")
    assert settings.provider_rate_limit == 1


@pytest.mark.parametrize(
    ("key", "attribute", "value", "expected"),
    [
        ("MODEL", "model", "deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ("PROVIDER_RATE_LIMIT", "provider_rate_limit", "20", 20),
        (
            "PROVIDER_PROGRESS_TIMEOUT",
            "provider_progress_timeout",
            "900",
            900.0,
        ),
        ("HTTP_READ_TIMEOUT", "http_read_timeout", "600", 600.0),
        ("FCC_OPEN_BROWSER", "open_admin_browser", "false", False),
        ("REASONING_POLICY", "reasoning_policy", "off", ReasoningPreference.OFF),
        ("GROQ_API_KEY", "groq_api_key", " secret ", "secret"),
        ("OPENROUTER_PROXY", "open_router_proxy", " http://proxy ", "http://proxy"),
    ],
)
def test_process_values_are_parsed_at_the_loader_boundary(
    key: str,
    attribute: str,
    value: str,
    expected: object,
) -> None:
    snapshot = compose_settings_snapshot({}, {key: value})

    assert getattr(snapshot.settings, attribute) == expected
    assert snapshot.sources[attribute] is ConfigSource.PROCESS


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        float(1 << 64),
    ],
)
def test_provider_progress_timeout_must_be_representable(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(provider_progress_timeout=value)


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "inf", "-inf", "nan", str(1 << 64)],
)
def test_loader_rejects_invalid_provider_progress_timeout(value: str) -> None:
    with pytest.raises(ValidationError):
        compose_settings_snapshot({}, {"PROVIDER_PROGRESS_TIMEOUT": value})


@pytest.mark.parametrize(
    "key",
    [
        "GROQ_API_KEY",
        "OPENROUTER_PROXY",
        "MODEL_OPUS",
        "TELEGRAM_BOT_TOKEN",
        "ALLOWED_DIR",
    ],
)
def test_optional_blank_process_values_normalize_to_none(key: str) -> None:
    snapshot = compose_settings_snapshot({}, {key: "  "})
    attribute = next(
        name
        for name, field in Settings.model_fields.items()
        if field.validation_alias == key
    )

    assert getattr(snapshot.settings, attribute) is None


def test_blank_required_process_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="MODEL"):
        compose_settings_snapshot({}, {"MODEL": " "})


@pytest.mark.parametrize("process_token", ["", "  ", "terminal-token"])
def test_process_auth_token_cannot_override_retained_default(
    process_token: str,
) -> None:
    snapshot = compose_settings_snapshot({}, {"ANTHROPIC_AUTH_TOKEN": process_token})

    assert snapshot.settings.proxy_auth_token == "freecc"
    assert snapshot.sources["proxy_auth_token"] is ConfigSource.DEFAULT


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_auth_checkbox_does_not_change_retained_token(enabled: str) -> None:
    snapshot = compose_settings_snapshot(
        {"PROXY_AUTH_ENABLED": enabled, "ANTHROPIC_AUTH_TOKEN": "managed-token"},
        {"ANTHROPIC_AUTH_TOKEN": "terminal-token"},
    )
    assert snapshot.settings.proxy_auth_enabled is (enabled == "true")
    assert snapshot.settings.proxy_auth_token == "managed-token"


def test_process_precedence_and_managed_token_exception() -> None:
    snapshot = compose_settings_snapshot(
        {
            "MODEL": "deepseek/managed",
            "ANTHROPIC_AUTH_TOKEN": "managed-token",
        },
        {
            "MODEL": "groq/process",
            "ANTHROPIC_AUTH_TOKEN": "stale-process-token",
        },
    )

    assert snapshot.settings.model == "groq/process"
    assert snapshot.sources["model"] is ConfigSource.PROCESS
    assert snapshot.settings.proxy_auth_token == "managed-token"
    assert snapshot.sources["proxy_auth_token"] is ConfigSource.MANAGED


def test_get_settings_is_cached_and_creates_managed_schema() -> None:
    clear_settings_cache()

    first = get_settings()
    second = get_settings()

    assert first is second


def _write_managed_config(text: str) -> Path:
    path = managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_repair_invalid_managed_provider_proxies_removes_all_eligible_values() -> None:
    invalid_openai = "invalid://user:leaked-secret@proxy.example:8080"
    managed = _write_managed_config(
        "\n".join(
            (
                "FCC_CONFIG_SCHEMA=1",
                "MODEL=nvidia_nim/test-model",
                "OPENROUTER_PROXY=http://proxy.example:notaport",
                "GROQ_PROXY=https://proxy.example:8443",
                f"OPENAI_PROXY={invalid_openai}",
                "PRESERVE_UNKNOWN=present",
                "",
            )
        )
    )

    removed = ManagedConfigStore().repair_invalid_provider_proxies({})

    values = dotenv_values_from_file(managed)
    assert removed == ("OPENROUTER_PROXY", "OPENAI_PROXY")
    assert "OPENROUTER_PROXY" not in values
    assert "OPENAI_PROXY" not in values
    assert values["GROQ_PROXY"] == "https://proxy.example:8443"
    assert values["MODEL"] == "nvidia_nim/test-model"
    assert values["PRESERVE_UNKNOWN"] == "present"
    assert list(managed.parent.glob(f".{managed.name}.*.tmp")) == []


def test_repair_valid_managed_provider_proxy_leaves_file_unchanged() -> None:
    managed = _write_managed_config(
        "# Keep this exact text on a no-op.\n"
        "FCC_CONFIG_SCHEMA=1\n"
        "OPENAI_PROXY=https://proxy.example:8443\n"
    )
    baseline = managed.read_bytes()

    assert ManagedConfigStore().repair_invalid_provider_proxies({}) == ()
    assert managed.read_bytes() == baseline


def test_repair_without_managed_file_is_a_noop() -> None:
    managed = managed_env_path()

    assert ManagedConfigStore().repair_invalid_provider_proxies({}) == ()
    assert not managed.exists()


@pytest.mark.parametrize("process_value", ("", "invalid://process-proxy"))
def test_repair_preserves_process_owned_managed_proxy(
    process_value: str,
) -> None:
    invalid_openai = "invalid://managed-proxy"
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\n"
        f"OPENAI_PROXY={invalid_openai}\n"
        "OPENROUTER_PROXY=invalid://unshadowed\n"
    )
    process = {"OPENAI_PROXY": process_value, "KEEP_PROCESS": "unchanged"}
    baseline_process = dict(process)

    assert ManagedConfigStore().repair_invalid_provider_proxies(process) == (
        "OPENROUTER_PROXY",
    )

    values = dotenv_values_from_file(managed)
    assert values["OPENAI_PROXY"] == invalid_openai
    assert "OPENROUTER_PROXY" not in values
    assert process == baseline_process


def test_repair_propagates_atomic_write_failure_without_changing_source() -> None:
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n"
    )
    baseline = managed.read_bytes()

    with (
        patch.object(
            loader,
            "atomic_write_managed_config",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        ManagedConfigStore().repair_invalid_provider_proxies({})

    assert managed.read_bytes() == baseline


def test_repair_is_idempotent_and_writes_only_once() -> None:
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n"
    )

    with patch.object(
        loader,
        "atomic_write_managed_config",
        wraps=loader.atomic_write_managed_config,
    ) as writer:
        assert ManagedConfigStore().repair_invalid_provider_proxies({}) == (
            "OPENAI_PROXY",
        )
        repaired = managed.read_bytes()
        assert ManagedConfigStore().repair_invalid_provider_proxies({}) == ()

    assert writer.call_count == 1
    assert managed.read_bytes() == repaired


def test_repair_propagates_malformed_managed_config() -> None:
    managed = _write_managed_config('FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY="unterminated\n')
    baseline = managed.read_bytes()

    with pytest.raises(ValueError, match="Could not parse configuration file"):
        ManagedConfigStore().repair_invalid_provider_proxies({})

    assert managed.read_bytes() == baseline


def test_repair_propagates_config_lock_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n"
    )
    baseline = managed.read_bytes()

    class UnavailableLock:
        def __init__(self, _path: Path) -> None:
            pass

        def acquire(self, *, wait: bool, timeout: float) -> bool:
            assert wait is True
            assert timeout == 10.0
            return False

    monkeypatch.setattr(loader, "InterprocessFileLock", UnavailableLock)

    with pytest.raises(TimeoutError, match="Could not acquire managed-config lock"):
        ManagedConfigStore().repair_invalid_provider_proxies({})

    assert managed.read_bytes() == baseline


def test_optional_strings_share_one_normalization_rule() -> None:
    settings = Settings.model_validate(
        {
            "GROQ_API_KEY": "  key  ",
            "OPENROUTER_PROXY": " ",
            "MODEL_OPUS": "",
            "ALLOWED_DIR": None,
        }
    )

    assert settings.groq_api_key == "key"
    assert settings.open_router_proxy is None
    assert settings.model_opus is None
    assert settings.allowed_dir is None


@pytest.mark.parametrize("value", [None, "", "   ", (), []])
def test_model_fallbacks_empty_values_disable_fallback(value: object) -> None:
    settings = Settings.model_validate({"MODEL_FALLBACKS": value})

    assert settings.model_fallbacks is None


@pytest.mark.parametrize(
    "value",
    [
        "open_router/vendor/model-a, groq/vendor/model-b ",
        ("open_router/vendor/model-a", " groq/vendor/model-b "),
        ["open_router/vendor/model-a", "groq/vendor/model-b"],
    ],
)
def test_model_fallbacks_preserve_order_and_trim_members(value: object) -> None:
    settings = Settings.model_validate({"MODEL_FALLBACKS": value})

    assert settings.model_fallbacks == (
        "open_router/vendor/model-a",
        "groq/vendor/model-b",
    )


@pytest.mark.parametrize(
    "value",
    [
        "open_router/vendor/model-a,,groq/vendor/model-b",
        "open_router/vendor/model-a,open_router/vendor/model-a",
    ],
)
def test_model_fallbacks_reject_blank_and_duplicate_members(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"MODEL_FALLBACKS": value})


@pytest.mark.parametrize(
    "field",
    ["MODEL", "HOST", "WHISPER_MODEL", "LOG_LEVEL", "ANTHROPIC_AUTH_TOKEN"],
)
def test_required_strings_reject_blank(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: "   "})


def test_model_validation_and_routing() -> None:
    settings = Settings(
        model="deepseek/fallback",
        model_opus="open_router/anthropic/claude-opus",
    )

    router = ModelRouter(settings)
    assert router.resolve("claude-opus-4").primary.provider_model_ref == (
        "open_router/anthropic/claude-opus"
    )
    assert router.resolve("unknown").primary.provider_model_ref == "deepseek/fallback"
    with pytest.raises(ValidationError, match="Invalid provider"):
        Settings(model="unknown/model")


@pytest.mark.parametrize(
    "field",
    ["MODEL", "MODEL_FABLE", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"],
)
def test_model_settings_reject_empty_model_suffix(field: str) -> None:
    with pytest.raises(ValidationError, match="model suffix"):
        Settings.model_validate({field: "open_router/"})


def test_configured_chat_model_refs_are_unique() -> None:
    settings = Settings(
        model="deepseek/fallback",
        model_fable="open_router/anthropic/claude-fable",
        model_sonnet="deepseek/fallback",
        model_fallbacks=(
            "groq/vendor/model-a",
            "open_router/anthropic/claude-fable",
            "lmstudio/vendor/model-b",
        ),
    )

    refs = configured_chat_model_refs(settings)

    assert [ref.model_ref for ref in refs] == [
        "deepseek/fallback",
        "open_router/anthropic/claude-fable",
        "groq/vendor/model-a",
        "lmstudio/vendor/model-b",
    ]


@pytest.mark.parametrize(
    ("model_ref", "provider", "model"),
    [
        ("nvidia_nim/meta/llama", "nvidia_nim", "meta/llama"),
        ("open_router/deepseek/r1", "open_router", "deepseek/r1"),
        ("ollama_cloud/qwen3-coder:480b", "ollama_cloud", "qwen3-coder:480b"),
    ],
)
def test_model_ref_parsing(model_ref: str, provider: str, model: str) -> None:
    assert parse_provider_type(model_ref) == provider
    assert parse_model_name(model_ref) == model


def test_paths_are_owned_by_fcc_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert messaging_state_dir_path() == tmp_path / ".fcc" / "agent_workspace"
    assert server_log_path() == tmp_path / ".fcc" / "logs" / "server.log"


def test_nim_settings_keep_request_local_validation() -> None:
    settings = NimSettings.model_validate(
        {
            "max_tokens": "1024",
            "temperature": "0.5",
            "seed": "7",
            "stop": "",
        }
    )

    assert settings.max_tokens == 1024
    assert settings.temperature == 0.5
    assert settings.seed == 7
    assert settings.stop is None
    assert NimSettings().max_tokens == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert NimSettings().top_p == 0.95
    for unsupported_top_p in (0.0, 0.9, 1.0):
        with pytest.raises(ValidationError):
            NimSettings(top_p=unsupported_top_p)


def test_settings_defaults_do_not_contain_empty_enum_strings() -> None:
    for _name, value in Settings():
        if isinstance(value, Enum):
            assert value.value
