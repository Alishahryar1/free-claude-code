import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.config.admin.values import MASKED_SECRET
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.server_urls import local_admin_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.version import package_version
from tests.api.support import create_test_app, provider_manager_for_app, runtime_for_app


def _local_client(app):
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


@pytest.mark.parametrize(
    "path",
    (
        "/admin/chat",
        "/admin/chat/old-session",
        "/admin/api/chat/sessions",
        "/admin/api/chat/events",
        "/admin/api/chat/settings",
        f"/admin/assets/{package_version()}/chat_sessions.js",
        f"/admin/assets/{package_version()}/chat_sessions.css",
    ),
)
def test_retired_chat_urls_are_not_served(path):
    client = _local_client(create_test_app())
    assert client.get(path).status_code == 404


def test_admin_retains_code_without_chat_markup():
    client = _local_client(create_test_app())
    response = client.get("/admin")
    assert response.status_code == 200
    assert 'id="view-code"' in response.text
    assert 'id="view-chat"' not in response.text
    assert "chat_sessions" not in response.text
    assert client.get("/admin/code").status_code == 200


def test_admin_retirement_preview_apply_and_runtime_agree(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\nMODEL=groq/default\nMODEL_OPUS=groq/managed\n"
        "GITHUB_MODELS_TOKEN=preserve-secret\nGITHUB_MODELS_PROXY=http://user:secret@proxy.test\nPRIVATE=hidden\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_OPUS", "github_models/retired-process")
    app = create_test_app()
    client = _local_client(app)
    body = client.post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL_SONNET": "github_models/stale",
                "MODEL_OPUS": "deepseek/cannot-unlock",
                "MODEL_FALLBACKS": "groq/a,github_models/old,deepseek/b",
            }
        },
    ).json()
    assert body["applied"]
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=groq/default" in text
    assert "MODEL_SONNET" not in text
    assert "MODEL_OPUS=groq/managed" in text
    assert "MODEL_FALLBACKS=groq/a,deepseek/b" in text
    assert "preserve-secret" in text and "http://user:secret@proxy.test" in text
    assert "preserve-secret" not in body["env_preview"]
    assert "http://user:secret@proxy.test" not in body["env_preview"]
    assert "PRIVATE=********" in body["env_preview"]
    effective = provider_manager_for_app(app).current_settings()
    assert effective.model == "groq/default"
    assert effective.model_opus is None and effective.model_sonnet is None
    assert effective.model_fallbacks == ("groq/a", "deepseek/b")


@pytest.fixture(autouse=True)
def _offline_credential_checks(monkeypatch):
    """These tests exercise config persistence; probe HTTP has its own suite."""
    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials",
        AsyncMock(return_value=()),
    )


def _set_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _clear_process_config(monkeypatch) -> None:
    for key in (
        "MODEL",
        "MODEL_FALLBACKS",
        "NVIDIA_NIM_API_KEY",
        "HUGGINGFACE_API_KEY",
        "OPENROUTER_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "BEDROCK_BASE_URL",
        "BEDROCK_PROXY",
        "OLLAMA_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "PROXY_AUTH_ENABLED",
        "TELEGRAM_PROXY_URL",
        "FCC_ENV_FILE",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "GITHUB_MODELS_TOKEN",
        "SAMBANOVA_API_KEY",
        "HOST",
        "PORT",
        "FCC_OPEN_BROWSER",
        "VOICE_NOTE_ENABLED",
        "WHISPER_DEVICE",
        "LOG_FILE",
        "ZAI_BASE_URL",
        "CLAUDE_WORKSPACE",
        "CLAUDE_CLI_BIN",
    ):
        monkeypatch.delenv(key, raising=False)
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.proxy_attr is None:
            continue
        alias = Settings.model_fields[descriptor.proxy_attr].validation_alias
        if alias is not None:
            monkeypatch.delenv(str(alias), raising=False)


def _catalog_proxy_env_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.proxy_attr is None:
            continue
        alias = Settings.model_fields[descriptor.proxy_attr].validation_alias
        if alias is not None:
            keys.append(str(alias))
    return tuple(keys)


@pytest.mark.parametrize(
    "path", ["/admin", "/admin/model_config", "/admin/messaging", "/admin/integrations"]
)
def test_admin_page_is_loopback_only(monkeypatch, tmp_path, path):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()

    assert _local_client(app).get(path).status_code == 200
    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get(path).status_code == 403


def test_admin_page_uses_installed_version(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "free_claude_code.api.admin_routes.package_version",
        lambda: "9.8.7",
    )

    response = _local_client(create_test_app()).get("/admin")

    assert response.status_code == 200
    assert "<p>Server Control · v9.8.7</p>" in response.text
    assert 'href="https://github.com/Alishahryar1/free-claude-code"' in response.text
    assert 'target="_blank"' in response.text
    assert 'rel="noopener noreferrer"' in response.text
    assert 'aria-label="Open Free Claude Code on GitHub"' in response.text
    assert 'src="/admin/assets/9.8.7/app-icon.svg"' in response.text
    assert 'href="/admin/assets/9.8.7/admin.css"' in response.text
    assert 'href="/admin/assets/9.8.7/code_sessions.css"' in response.text
    assert 'src="/admin/assets/9.8.7/model_combobox.js"' in response.text
    assert 'src="/admin/assets/9.8.7/form_controls.js"' in response.text
    assert 'src="/admin/assets/9.8.7/code_sessions.js"' in response.text
    assert 'src="/admin/assets/9.8.7/admin.js"' in response.text
    assert 'href="/admin/assets/admin.css"' not in response.text
    assert 'href="/admin/assets/code_sessions.css"' not in response.text
    assert 'src="/admin/assets/code_sessions.js"' not in response.text
    assert 'src="/admin/assets/admin.js"' not in response.text


@pytest.mark.parametrize(
    ("filename", "media_type"),
    (
        ("admin.css", "text/css"),
        ("admin.js", "text/javascript"),
        ("form_controls.js", "text/javascript"),
        ("code_sessions.css", "text/css"),
        ("code_sessions.js", "text/javascript"),
        ("session_ui.js", "text/javascript"),
        ("model_combobox.js", "text/javascript"),
    ),
)
def test_admin_versioned_assets_serve_packaged_files(
    monkeypatch,
    tmp_path,
    filename,
    media_type,
):
    asset_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "free_claude_code"
        / "api"
        / "admin_static"
        / filename
    )
    _set_home(monkeypatch, tmp_path)
    response = _local_client(create_test_app()).get(
        f"/admin/assets/{package_version()}/{filename}"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.content == asset_path.read_bytes()


def test_admin_versioned_logo_reuses_packaged_app_icon(monkeypatch, tmp_path):
    asset_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "free_claude_code"
        / "assets"
        / "app-icon.svg"
    )
    _set_home(monkeypatch, tmp_path)
    response = _local_client(create_test_app()).get(
        f"/admin/assets/{package_version()}/app-icon.svg"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.content == asset_path.read_bytes()


@pytest.mark.parametrize(
    "path",
    (
        "/admin",
        f"/admin/assets/{package_version()}/app-icon.svg",
        f"/admin/assets/{package_version()}/admin.css",
        f"/admin/assets/{package_version()}/admin.js",
        f"/admin/assets/{package_version()}/code_sessions.css",
        f"/admin/assets/{package_version()}/code_sessions.js",
        f"/admin/assets/{package_version()}/model_combobox.js",
        "/admin/api/config",
    ),
)
def test_admin_responses_are_never_cached(monkeypatch, tmp_path, path):
    _set_home(monkeypatch, tmp_path)
    response = _local_client(create_test_app()).get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("path", "client_host", "expected_status"),
    (
        ("/admin", "203.0.113.10", 403),
        ("/admin/assets/admin.js", "127.0.0.1", 404),
        (
            f"/admin/assets/{package_version()}.stale/admin.js",
            "127.0.0.1",
            404,
        ),
        (
            f"/admin/assets/{package_version()}/missing.js",
            "127.0.0.1",
            404,
        ),
        (
            f"/admin/assets/{package_version()}/admin.js",
            "203.0.113.10",
            403,
        ),
    ),
)
def test_admin_http_errors_are_never_cached(
    monkeypatch,
    tmp_path,
    path,
    client_host,
    expected_status,
):
    _set_home(monkeypatch, tmp_path)
    client = TestClient(
        create_test_app(),
        base_url="http://127.0.0.1",
        client=(client_host, 50000),
    )

    response = client.get(path)

    assert response.status_code == expected_status
    assert response.headers["cache-control"] == "no-store"


def test_admin_apply_payload_validation_errors_are_never_cached(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    response = _local_client(create_test_app()).post(
        "/admin/api/config/apply",
        content="{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"


def test_admin_unexpected_errors_are_never_cached(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    client = TestClient(
        create_test_app(),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    with patch(
        "free_claude_code.runtime.configuration.ConfigurationService.admin_config",
        side_effect=RuntimeError("test error"),
    ):
        response = client.get("/admin/api/config")

    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"


def test_admin_cache_policy_does_not_match_similar_public_paths(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    response = _local_client(create_test_app()).get("/administrator")

    assert response.status_code == 404
    assert "cache-control" not in response.headers


def test_admin_api_fetches_bypass_browser_cache():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'cache: "no-store"' in script


class _FakeConnectedAccount:
    def __init__(self, provider_id: str = "openai") -> None:
        self.provider_id = provider_id
        self.connected = False
        self.revision = 0
        self.cancelled = False
        self.started_modes: list[ConnectedAccountLoginMode] = []

    def is_connected(self) -> bool:
        return self.connected

    def status(self) -> ConnectedAccountStatus:
        status = ConnectedAccountStatus(
            provider_id=self.provider_id,
            state=(
                ConnectedAccountState.CONNECTED
                if self.connected
                else ConnectedAccountState.DISCONNECTED
            ),
            connected=self.connected,
            revision=self.revision,
        )
        if self.provider_id == "github_copilot":
            return replace(
                status,
                display_identity="octocat" if self.connected else None,
                supported_login_modes=(ConnectedAccountLoginMode.DEVICE,),
                default_login_mode=ConnectedAccountLoginMode.DEVICE,
            )
        return replace(status, email="safe@example.com" if self.connected else None)

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        self.started_modes.append(mode)
        return replace(
            self.status(),
            state=ConnectedAccountState.CONNECTING,
            connected=False,
            attempt_id="login_safe",
            mode=mode,
            authorization_url=(
                "https://auth.openai.com/safe"
                if mode == ConnectedAccountLoginMode.BROWSER
                else None
            ),
            verification_url=(
                "https://github.com/login/device"
                if mode == ConnectedAccountLoginMode.DEVICE
                else None
            ),
            user_code="ABCD-1234" if mode == ConnectedAccountLoginMode.DEVICE else None,
        )

    async def cancel_login(self) -> ConnectedAccountStatus:
        self.cancelled = True
        return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        self.connected = False
        self.revision += 1
        return self.status()

    async def close(self) -> None:
        return None


def test_admin_connected_account_routes_are_safe_loopback_only_and_uncached(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    account = _FakeConnectedAccount()
    app = create_test_app(connected_accounts={"openai": account})
    client = _local_client(app)

    status_response = client.get("/admin/api/providers/openai/auth")
    login_response = client.post(
        "/admin/api/providers/openai/auth/login",
        json={"mode": "browser"},
    )
    cancel_response = client.post("/admin/api/providers/openai/auth/cancel")

    assert status_response.status_code == 200
    assert status_response.headers["cache-control"] == "no-store"
    assert status_response.json()["state"] == "disconnected"
    assert login_response.status_code == 200
    assert login_response.json() == {
        "provider_id": "openai",
        "state": "connecting",
        "connected": False,
        "revision": 0,
        "attempt_id": "login_safe",
        "mode": "browser",
        "authorization_url": "https://auth.openai.com/safe",
        "supported_login_modes": ["browser", "device"],
        "default_login_mode": "browser",
    }
    assert "token" not in login_response.text.lower()
    assert cancel_response.status_code == 200
    assert account.cancelled is True
    remote = TestClient(app, client=("203.0.113.10", 50000))
    assert remote.get("/admin/api/providers/openai/auth").status_code == 403


@pytest.mark.parametrize(
    ("provider_id", "default_mode"),
    [("openai", "browser"), ("github_copilot", "device")],
)
@pytest.mark.parametrize("payload", [{}, {"mode": None}])
def test_admin_login_uses_the_provider_default(
    monkeypatch,
    tmp_path,
    provider_id,
    default_mode,
    payload,
):
    _set_home(monkeypatch, tmp_path)
    account = _FakeConnectedAccount(provider_id)
    client = _local_client(
        create_test_app(providers={}, connected_accounts={provider_id: account})
    )

    response = client.post(
        f"/admin/api/providers/{provider_id}/auth/login", json=payload
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["mode"] == default_mode
    assert account.started_modes == [ConnectedAccountLoginMode(default_mode)]
    if default_mode == "device":
        assert response.json()["user_code"] == "ABCD-1234"
        assert response.json()["verification_url"] == "https://github.com/login/device"
        assert "authorization_url" not in response.json()


@pytest.mark.parametrize("mode", ["browser", "unrecognized"])
def test_admin_rejects_unsupported_login_modes_before_starting(
    monkeypatch, tmp_path, mode
):
    _set_home(monkeypatch, tmp_path)
    account = _FakeConnectedAccount("github_copilot")
    client = _local_client(
        create_test_app(providers={}, connected_accounts={"github_copilot": account})
    )

    response = client.post(
        "/admin/api/providers/github_copilot/auth/login", json={"mode": mode}
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert account.started_modes == []


def test_admin_connected_account_identity_modes_and_disconnect_are_independent(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    openai = _FakeConnectedAccount()
    copilot = _FakeConnectedAccount("github_copilot")
    openai.connected = copilot.connected = True
    client = _local_client(
        create_test_app(
            providers={},
            connected_accounts={"openai": openai, "github_copilot": copilot},
        )
    )

    openai_status = client.get("/admin/api/providers/openai/auth").json()
    copilot_status = client.get("/admin/api/providers/github_copilot/auth").json()
    assert openai_status["email"] == "safe@example.com"
    assert openai_status["supported_login_modes"] == ["browser", "device"]
    assert copilot_status["display_identity"] == "octocat"
    assert "email" not in copilot_status
    assert copilot_status["supported_login_modes"] == ["device"]
    assert copilot_status["default_login_mode"] == "device"
    assert "token" not in str(copilot_status).lower()

    cancelled = client.post("/admin/api/providers/github_copilot/auth/cancel")
    assert cancelled.status_code == 200
    assert copilot.cancelled is True
    assert openai.cancelled is False
    disconnected = client.delete("/admin/api/providers/github_copilot/auth")
    assert disconnected.status_code == 200
    assert disconnected.json()["connected"] is False
    assert disconnected.json()["supported_login_modes"] == ["device"]
    assert client.get("/admin/api/providers/openai/auth").json()["connected"] is True


def test_admin_openai_keeps_explicit_device_login(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    account = _FakeConnectedAccount()
    client = _local_client(create_test_app(connected_accounts={"openai": account}))

    response = client.post(
        "/admin/api/providers/openai/auth/login", json={"mode": "device"}
    )

    assert response.status_code == 200
    assert account.started_modes == [ConnectedAccountLoginMode.DEVICE]


def test_admin_rejects_auth_routes_for_non_connected_provider(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    response = _local_client(create_test_app()).get(
        "/admin/api/providers/nvidia_nim/auth"
    )

    assert response.status_code == 404


def test_admin_page_no_longer_renders_generated_env_panel(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()

    response = _local_client(app).get("/admin")

    assert response.status_code == 200
    assert "Generated Env" not in response.text
    assert "envPreview" not in response.text


def test_admin_page_no_longer_renders_global_status_header(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()

    response = _local_client(app).get("/admin")

    assert response.status_code == 200
    assert "Local Admin" not in response.text
    assert "serverStatus" not in response.text
    assert "modelBadge" not in response.text


def test_admin_static_no_longer_fetches_global_status_header():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'api("/admin/api/status")' not in script
    assert "updateHeader" not in script
    assert '"Running"' not in script
    assert "serverStatus" not in script
    assert "modelBadge" not in script


def test_admin_static_hides_managed_source_label():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'managed_env: "",' in script
    assert "hasOwnProperty.call(labels, source)" in script
    assert 'parts.push("locked")' in script
    assert "sourceEl.textContent = source" in script


def test_admin_static_places_reasoning_fields_in_model_config():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'sections: ["models", "reasoning", "web_tools"]' in script
    assert 'sections: ["models", "thinking", "web_tools"]' not in script


def test_admin_static_model_combobox_owns_dropdown_and_search_behavior():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )
    combobox_script = Path(
        "src/free_claude_code/api/admin_static/model_combobox.js"
    ).read_text(encoding="utf-8")
    styles = Path("src/free_claude_code/api/admin_static/admin.css").read_text(
        encoding="utf-8"
    )

    assert 'api("/admin/api/models" + (refresh ? "/refresh" : "")' in script
    assert 'field.type === "model" || field.type === "optional_model"' in script
    assert "new window.FccModelCombobox" in script
    assert 'input.setAttribute("role", "combobox")' in combobox_script
    assert 'this.listbox.setAttribute("role", "listbox")' in combobox_script
    assert 'this.toggle.className = "model-combobox-toggle"' in combobox_script
    assert "class FccModelCombobox" in combobox_script
    assert 'input.addEventListener("click", () => this.open())' in combobox_script
    assert "value.toLocaleLowerCase().includes(normalizedQuery)" in combobox_script
    assert 'event.key === "ArrowDown" || event.key === "ArrowUp"' in combobox_script
    assert "this.setActive(this.visibleOptions.length - 1)" in combobox_script
    assert 'event.key === "Enter"' in combobox_script
    assert 'event.key === "Escape"' in combobox_script
    assert 'document.createElement("datalist")' not in combobox_script
    assert ".model-combobox-list" in styles
    assert ".model-combobox-option.active" in styles
    assert styles.count("background-image: var(--dropdown-chevron)") == 2


def test_admin_static_model_combobox_preserves_custom_slugs_and_none_semantics():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert '? ["None", ...state.modelOptions]' in script
    assert "You can still enter a custom slug." in script
    assert 'input.dataset.fieldType === "optional_model"' in script
    assert "return null;" in script
    assert "await hydrateModelOptions();" in script
    assert "Model fields remain editable" in script
    assert "result.failed_providers || []" in script
    assert '"warn"' in script


def test_admin_config_masks_secrets_and_exposes_manifest(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    keys = {field["key"] for field in body["fields"]}
    assert "MODEL_FABLE" in keys
    assert "REASONING_FABLE" in keys
    assert "ANTHROPIC_AUTH_TOKEN" in keys
    assert "PROXY_AUTH_ENABLED" in keys
    assert "LOG_LEVEL" in keys
    assert "OPENROUTER_API_KEY" in keys
    assert "AWS_BEARER_TOKEN_BEDROCK" in keys
    assert "BEDROCK_BASE_URL" in keys
    assert "FIREWORKS_API_KEY" in keys
    assert "CLOUDFLARE_API_TOKEN" in keys
    assert "CLOUDFLARE_ACCOUNT_ID" in keys
    assert "GITHUB_MODELS_TOKEN" not in keys
    assert "GEMINI_API_KEY" in keys
    assert "GROQ_API_KEY" in keys
    assert "SAMBANOVA_API_KEY" in keys
    assert "TELEGRAM_PROXY_URL" in keys
    assert "CEREBRAS_API_KEY" in keys
    assert "OLLAMA_API_KEY" in keys
    assert "FCC_OPEN_BROWSER" in keys
    assert "ZAI_BASE_URL" not in keys
    assert "CLAUDE_WORKSPACE" not in keys
    assert "CLAUDE_CLI_BIN" not in keys
    assert "LOG_FILE" not in keys
    auth_field = next(
        field for field in body["fields"] if field["key"] == "ANTHROPIC_AUTH_TOKEN"
    )
    assert auth_field["secret"] is True
    assert auth_field["value"] == MASKED_SECRET
    assert auth_field["source"] == "default"
    assert auth_field["nullable"] is False
    telegram_proxy_field = next(
        field for field in body["fields"] if field["key"] == "TELEGRAM_PROXY_URL"
    )
    assert telegram_proxy_field["secret"] is True
    assert telegram_proxy_field["value"] is None
    assert telegram_proxy_field["configured"] is False
    assert telegram_proxy_field["nullable"] is True
    assert body["paths"] == {"managed": str(tmp_path / ".fcc" / ".env")}
    catalog_smoke_keys = {
        f"FCC_SMOKE_MODEL_{provider_id.upper()}" for provider_id in PROVIDER_CATALOG
    }
    actual_smoke_keys = {
        field["key"]
        for field in body["fields"]
        if field["key"].startswith("FCC_SMOKE_MODEL_")
    }
    assert actual_smoke_keys == catalog_smoke_keys | {
        "FCC_SMOKE_MODEL_MISTRAL_REASONING"
    }
    open_browser_field = next(
        field for field in body["fields"] if field["key"] == "FCC_OPEN_BROWSER"
    )
    assert open_browser_field["type"] == "boolean"
    assert open_browser_field["value"] == "true"
    assert open_browser_field["restart_required"] is False
    progress_timeout_field = next(
        field for field in body["fields"] if field["key"] == "PROVIDER_PROGRESS_TIMEOUT"
    )
    assert progress_timeout_field["label"] == "Provider Progress Timeout"
    assert progress_timeout_field["section"] == "runtime"
    assert progress_timeout_field["type"] == "number"
    assert progress_timeout_field["value"] == "600.0"
    assert progress_timeout_field["advanced"] is True
    assert progress_timeout_field["restart_required"] is False
    assert "non-empty protocol event" in progress_timeout_field["description"]
    assert "Independent of HTTP Read Timeout" in progress_timeout_field["description"]
    model_field_types = {
        field["key"]: field["type"]
        for field in body["fields"]
        if field["key"]
        in {
            "MODEL",
            "MODEL_FABLE",
            "MODEL_OPUS",
            "MODEL_SONNET",
            "MODEL_HAIKU",
            "MODEL_FALLBACKS",
        }
    }
    assert model_field_types == {
        "MODEL": "model",
        "MODEL_FABLE": "optional_model",
        "MODEL_OPUS": "optional_model",
        "MODEL_SONNET": "optional_model",
        "MODEL_HAIKU": "optional_model",
        "MODEL_FALLBACKS": "model_list",
    }
    fallback_field = next(
        field for field in body["fields"] if field["key"] == "MODEL_FALLBACKS"
    )
    assert fallback_field["label"] == "Fallback Models"
    assert fallback_field["value"] is None
    assert fallback_field["nullable"] is True
    assert fallback_field["restart_required"] is False
    assert "fails before output" in fallback_field["description"]
    assert "every client" in fallback_field["description"]
    assert "multiple providers" in fallback_field["description"]
    reasoning_policy = next(
        field for field in body["fields"] if field["key"] == "REASONING_POLICY"
    )
    assert reasoning_policy["section"] == "reasoning"
    assert reasoning_policy["type"] == "select"
    assert reasoning_policy["value"] == "client"
    assert reasoning_policy["options"] == [
        {"value": "off", "label": "Off"},
        {"value": "client", "label": "From client"},
        {"value": "low", "label": "Low"},
        {"value": "medium", "label": "Medium"},
        {"value": "high", "label": "High"},
        {"value": "xhigh", "label": "X-High"},
        {"value": "max", "label": "Max"},
    ]
    route_reasoning = next(
        field for field in body["fields"] if field["key"] == "REASONING_FABLE"
    )
    assert route_reasoning["options"] == [
        {"value": "inherit", "label": "Inherit"},
        *reasoning_policy["options"],
    ]
    restart_required = {
        field["key"] for field in body["fields"] if field["restart_required"] is True
    }
    assert {
        "ANTHROPIC_AUTH_TOKEN",
        "DEBUG_PLATFORM_EDITS",
        "DEBUG_SUBAGENT_STACK",
        "LOG_RAW_API_PAYLOADS",
        "LOG_API_ERROR_TRACEBACKS",
        "LOG_RAW_MESSAGING_CONTENT",
        "LOG_RAW_CLI_DIAGNOSTICS",
        "LOG_MESSAGING_ERROR_DETAILS",
    } <= restart_required


def test_admin_models_include_configured_and_cached_canonical_slugs():
    settings = Settings()
    settings.model = "nvidia_nim/configured-model"
    settings.model_opus = "open_router/anthropic/configured-opus"
    settings.open_router_api_key = "open-router-key"
    app = create_test_app(settings)
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("anthropic/configured-opus"),
            ProviderModelInfo("meta/llama-3.3"),
        },
    )

    response = _local_client(app).get("/admin/api/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            "nvidia_nim/configured-model",
            "open_router/anthropic/configured-opus",
            "open_router/meta/llama-3.3",
        ],
        "failed_providers": [],
    }


def test_admin_model_refresh_returns_the_updated_canonical_catalog():
    settings = Settings()
    settings.model = "deepseek/deepseek-chat"
    settings.deepseek_api_key = "deepseek-key"
    app = create_test_app(settings)
    runtime = app.state.services.admin

    async def refresh_models() -> ProviderModelRefreshResult:
        provider_manager_for_app(app).cache_model_infos(
            "deepseek",
            {ProviderModelInfo("deepseek-reasoner")},
        )
        return ProviderModelRefreshResult(refreshed_provider_ids=("deepseek",))

    runtime.refresh_models = AsyncMock(side_effect=refresh_models)

    response = _local_client(app).post("/admin/api/models/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"],
        "failed_providers": [],
    }
    runtime.refresh_models.assert_awaited_once_with()


def test_admin_model_refresh_reports_partial_provider_failures():
    settings = Settings()
    settings.model = "deepseek/deepseek-chat"
    app = create_test_app(settings)
    runtime = app.state.services.admin
    runtime.refresh_models = AsyncMock(
        return_value=ProviderModelRefreshResult(
            refreshed_provider_ids=("deepseek",),
            failed_provider_ids=("open_router",),
        )
    )

    response = _local_client(app).post("/admin/api/models/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["deepseek/deepseek-chat"],
        "failed_providers": ["open_router"],
    }


def test_admin_config_preserves_managed_env_source_contract(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("MODEL=open_router/managed-model\n", encoding="utf-8")
    app = create_test_app()

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    model_field = next(field for field in body["fields"] if field["key"] == "MODEL")
    assert model_field["source"] == "managed_env"
    assert model_field["locked"] is False


def test_admin_apply_persists_open_browser_for_next_launch(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"FCC_OPEN_BROWSER": False}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == []
    assert body["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }
    managed_env = tmp_path / ".fcc" / ".env"
    assert "FCC_OPEN_BROWSER=false" in managed_env.read_text(encoding="utf-8")


def test_admin_apply_hot_publishes_provider_progress_timeout(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PROVIDER_PROGRESS_TIMEOUT": 900}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == []
    assert body["restart"]["required"] is False
    assert (
        provider_manager_for_app(app).current_settings().provider_progress_timeout
        == 900.0
    )
    managed_env = tmp_path / ".fcc" / ".env"
    assert "PROVIDER_PROGRESS_TIMEOUT=900" in managed_env.read_text(encoding="utf-8")


def test_admin_apply_hot_publishes_ordered_model_fallbacks(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()
    value = "open_router/vendor/model-a,groq/vendor/model-b"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL_FALLBACKS": value}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["restart"]["required"] is False
    assert provider_manager_for_app(app).current_settings().model_fallbacks == (
        "open_router/vendor/model-a",
        "groq/vendor/model-b",
    )
    managed_env = tmp_path / ".fcc" / ".env"
    assert f"MODEL_FALLBACKS={value}" in managed_env.read_text(encoding="utf-8")

    loaded = _local_client(app).get("/admin/api/config").json()
    field = next(item for item in loaded["fields"] if item["key"] == "MODEL_FALLBACKS")
    assert field["value"] == value
    assert field["source"] == "managed_env"


def test_admin_apply_null_removes_model_fallbacks(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\nMODEL_FALLBACKS=open_router/vendor/model-a\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL_FALLBACKS": None}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert provider_manager_for_app(app).current_settings().model_fallbacks is None
    assert "MODEL_FALLBACKS=" not in env_file.read_text(encoding="utf-8")


def test_admin_rejects_invalid_provider_progress_timeout_without_writing(
    monkeypatch,
    tmp_path,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PROVIDER_PROGRESS_TIMEOUT": 0}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert any("PROVIDER_PROGRESS_TIMEOUT" in error for error in body["errors"])
    managed_env = tmp_path / ".fcc" / ".env"
    assert "PROVIDER_PROGRESS_TIMEOUT=" not in managed_env.read_text(encoding="utf-8")


def test_admin_apply_masks_telegram_proxy_credentials(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()
    proxy_url = "https://user:password@proxy.example:8443"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"TELEGRAM_PROXY_URL": proxy_url}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "TELEGRAM_PROXY_URL=********" in body["env_preview"]
    assert proxy_url not in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert f"TELEGRAM_PROXY_URL={proxy_url}" in text


def test_admin_apply_null_removes_optional_secret(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\nTELEGRAM_PROXY_URL=https://secret.invalid\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"TELEGRAM_PROXY_URL": None}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert "TELEGRAM_PROXY_URL=" not in env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("submitted", [MASKED_SECRET, "", "   "])
def test_admin_apply_masked_or_blank_secret_is_unchanged(
    monkeypatch,
    tmp_path,
    submitted,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\nOPENROUTER_API_KEY=original-secret\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"OPENROUTER_API_KEY": submitted}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert "OPENROUTER_API_KEY=original-secret" in env_file.read_text(encoding="utf-8")


def test_admin_apply_rejects_bad_model_shape(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "missing-provider-prefix"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert any("provider type" in error for error in body["errors"])


def test_admin_apply_rejects_duplicate_model_fallbacks(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()
    duplicate = "groq/vendor/model,groq/vendor/model"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL_FALLBACKS": duplicate}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert any("duplicate" in error.lower() for error in body["errors"])


@pytest.mark.parametrize("proxy_key", _catalog_proxy_env_keys())
def test_admin_apply_rejects_invalid_catalog_provider_proxy(
    monkeypatch,
    tmp_path,
    proxy_key,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    invalid_proxy = "not-a-proxy://user:leaked-secret@proxy.example:8080"

    response = _local_client(create_test_app()).post(
        "/admin/api/config/apply",
        json={"values": {proxy_key: invalid_proxy}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert body["errors"] == [
        (
            f"{proxy_key}: must be a proxy URL with a supported scheme and host "
            "(for example http://127.0.0.1:8080 or socks5://127.0.0.1:1080)"
        )
    ]
    assert invalid_proxy not in response.text
    assert "leaked-secret" not in response.text


@pytest.mark.parametrize(
    "proxy",
    (
        "http://user:password@127.0.0.1:8080",
        "https://proxy.example:8443",
        "socks5://127.0.0.1:1080",
        "socks5h://proxy.example:1080",
        "  http://127.0.0.1:8080  ",
    ),
)
def test_admin_apply_accepts_httpx_provider_proxy(monkeypatch, tmp_path, proxy):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)

    response = _local_client(create_test_app()).post(
        "/admin/api/config/apply",
        json={"values": {"OPENAI_PROXY": proxy}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert response.json()["valid"] is True


@pytest.mark.parametrize(
    "proxy",
    (
        "copied Admin page text",
        "socks://127.0.0.1:1080",
        "http://",
        "http:///path",
        "http://proxy.example:notaport",
    ),
)
def test_admin_apply_rejects_unusable_provider_proxy(monkeypatch, tmp_path, proxy):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)

    response = _local_client(create_test_app()).post(
        "/admin/api/config/apply",
        json={"values": {"OPENAI_PROXY": proxy}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert len(body["errors"]) == 1
    assert body["errors"][0].startswith("OPENAI_PROXY: must be a proxy URL")
    assert "OPENAI_PROXY=********" in body["env_preview"]


def test_admin_apply_rejects_invalid_provider_proxy_without_side_effects(
    monkeypatch,
    tmp_path,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("MODEL=open_router/test-model\n", encoding="utf-8")
    callbacks: list[str] = []

    def restart_callback() -> None:
        callbacks.append("restart")

    app = create_test_app(restart_callback=restart_callback)
    _local_client(app).get("/admin/api/config")
    baseline = env_file.read_bytes()
    invalid_proxy = "invalid://user:leaked-secret@proxy.example:8080"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"OPENAI_PROXY": invalid_proxy}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["applied"] is False
    assert body["pending_fields"] == []
    assert env_file.read_bytes() == baseline
    assert callbacks == []
    assert invalid_proxy not in response.text
    assert "leaked-secret" not in response.text


def test_admin_apply_validates_retained_proxy_and_allows_removal(
    monkeypatch,
    tmp_path,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    invalid_proxy = "invalid://proxy.example:8080"
    original = f"GROQ_PROXY={invalid_proxy}\n"
    env_file.write_text(original, encoding="utf-8")
    app = create_test_app()
    _local_client(app).get("/admin/api/config")
    baseline = env_file.read_bytes()

    rejected = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "open_router/test-model"}},
    )

    assert rejected.status_code == 200
    assert rejected.json()["applied"] is False
    assert env_file.read_bytes() == baseline

    removed = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"GROQ_PROXY": None}},
    )

    assert removed.status_code == 200
    assert removed.json()["applied"] is True
    assert "GROQ_PROXY=" not in env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("submitted", (MASKED_SECRET, "", "   "))
def test_admin_apply_preserves_valid_stored_proxy_secret(
    monkeypatch,
    tmp_path,
    submitted,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "OPENAI_PROXY=http://user:password@127.0.0.1:8080\n",
        encoding="utf-8",
    )

    response = _local_client(create_test_app()).post(
        "/admin/api/config/apply",
        json={"values": {"OPENAI_PROXY": submitted}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["valid"] is True
    assert "OPENAI_PROXY=********" in body["env_preview"]
    assert "password" not in response.text


def test_admin_apply_writes_complete_managed_env_and_masks_preview(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "open_router/test-model",
                "OPENROUTER_API_KEY": "router-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "OPENROUTER_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "OPENROUTER_API_KEY=router-secret" in text
    assert "ANTHROPIC_AUTH_TOKEN=" not in text
    assert "PROXY_AUTH_ENABLED=" not in text
    assert "HOST=" not in text
    assert "PORT=" not in text
    assert body["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }


def test_admin_apply_writes_fireworks_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "fireworks/test-model",
                "FIREWORKS_API_KEY": "fw-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "FIREWORKS_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=fireworks/test-model" in text
    assert "FIREWORKS_API_KEY=fw-secret" in text


def test_admin_apply_writes_gemini_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "gemini/models/gemini-3.1-flash-lite",
                "GEMINI_API_KEY": "gm-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "GEMINI_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=gemini/models/gemini-3.1-flash-lite" in text
    assert "GEMINI_API_KEY=gm-secret" in text


def test_admin_apply_writes_groq_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "groq/llama-3.3-70b-versatile",
                "GROQ_API_KEY": "gq-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "GROQ_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=groq/llama-3.3-70b-versatile" in text
    assert "GROQ_API_KEY=gq-secret" in text


def test_admin_apply_writes_sambanova_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "sambanova/Meta-Llama-3.3-70B-Instruct",
                "SAMBANOVA_API_KEY": "sn-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "SAMBANOVA_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=sambanova/Meta-Llama-3.3-70B-Instruct" in text
    assert "SAMBANOVA_API_KEY=sn-secret" in text


def test_admin_apply_writes_cerebras_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "cerebras/llama3.1-8b",
                "CEREBRAS_API_KEY": "cb-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "CEREBRAS_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=cerebras/llama3.1-8b" in text
    assert "CEREBRAS_API_KEY=cb-secret" in text


def test_admin_apply_writes_bedrock_region_config_and_masks_key(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "bedrock/openai.gpt-oss-120b",
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-secret",
                "BEDROCK_BASE_URL": ("https://bedrock-mantle.us-west-2.api.aws/v1"),
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "AWS_BEARER_TOKEN_BEDROCK=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=bedrock/openai.gpt-oss-120b" in text
    assert "AWS_BEARER_TOKEN_BEDROCK=bedrock-secret" in text
    assert "BEDROCK_BASE_URL=https://bedrock-mantle.us-west-2.api.aws/v1" in text


def test_admin_apply_writes_cloudflare_fields_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "cloudflare/@cf/moonshotai/kimi-k2.6",
                "CLOUDFLARE_API_TOKEN": "cf-secret",
                "CLOUDFLARE_ACCOUNT_ID": "cf-account",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "CLOUDFLARE_API_TOKEN=********" in body["env_preview"]
    assert "CLOUDFLARE_ACCOUNT_ID=cf-account" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=cloudflare/@cf/moonshotai/kimi-k2.6" in text
    assert "CLOUDFLARE_API_TOKEN=cf-secret" in text
    assert "CLOUDFLARE_ACCOUNT_ID=cf-account" in text


def test_admin_apply_writes_huggingface_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "huggingface/openai/gpt-oss-120b:fastest",
                "HUGGINGFACE_API_KEY": "hf-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == ["HUGGINGFACE_API_KEY"]
    assert "HUGGINGFACE_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=huggingface/openai/gpt-oss-120b:fastest" in text
    assert "HUGGINGFACE_API_KEY=hf-secret" in text


@pytest.mark.parametrize(
    ("device", "credential_key"),
    [
        ("nvidia_nim", "NVIDIA_NIM_API_KEY"),
        ("cpu", "HUGGINGFACE_API_KEY"),
    ],
)
def test_admin_key_change_requires_restart_for_active_voice_backend(
    monkeypatch,
    tmp_path,
    device,
    credential_key,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "VOICE_NOTE_ENABLED=true",
                f"WHISPER_DEVICE={device}",
                f"{credential_key}=old-key",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {credential_key: "new-key"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == [credential_key]
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": [credential_key],
    }


@pytest.mark.parametrize(
    ("key", "initial", "updated"),
    [
        ("ANTHROPIC_AUTH_TOKEN", "old-token", "new-token"),
        ("DEBUG_PLATFORM_EDITS", "true", "false"),
        ("DEBUG_SUBAGENT_STACK", "true", "false"),
        ("LOG_RAW_API_PAYLOADS", "true", "false"),
        ("LOG_API_ERROR_TRACEBACKS", "true", "false"),
        ("LOG_RAW_MESSAGING_CONTENT", "true", "false"),
        ("LOG_RAW_CLI_DIAGNOSTICS", "true", "false"),
        ("LOG_MESSAGING_ERROR_DETAILS", "true", "false"),
    ],
)
def test_admin_constructor_captured_setting_requires_restart(
    monkeypatch,
    tmp_path,
    key,
    initial,
    updated,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f"{key}={initial}\n", encoding="utf-8")
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {key: updated}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == [key]
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": [key],
    }


def test_admin_apply_writes_cohere_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "cohere/command-a-plus-05-2026",
                "COHERE_API_KEY": "cohere-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "COHERE_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=cohere/command-a-plus-05-2026" in text
    assert "COHERE_API_KEY=cohere-secret" in text


def test_admin_apply_replaces_retired_model_and_ignores_retired_credential(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "github_models/openai/gpt-4.1",
                "GITHUB_MODELS_TOKEN": "github-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "GITHUB_MODELS_TOKEN" not in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=" not in text
    assert "GITHUB_MODELS_TOKEN" not in text


def test_admin_apply_preserves_hidden_diagnostics_and_smoke_values(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=nvidia_nim/old-model",
                "LOG_RAW_API_PAYLOADS=true",
                "FCC_SMOKE_MODEL_ZAI=zai/smoke-model",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "open_router/test-model"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "LOG_RAW_API_PAYLOADS=true" in text
    assert "FCC_SMOKE_MODEL_ZAI=zai/smoke-model" in text


def test_admin_apply_preserves_unrecognized_managed_zai_base_url(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=zai/glm-5.2",
                "ZAI_API_KEY=zai-secret",
                "ZAI_BASE_URL=https://custom.zai.invalid/v1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "zai/glm-5.2"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "ZAI_API_KEY=zai-secret" in text
    assert "ZAI_BASE_URL=https://custom.zai.invalid/v1" in text


def test_admin_apply_preserves_unrecognized_managed_assignments(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=open_router/test-model",
                "CLAUDE_WORKSPACE=C:/custom/workspace",
                "CLAUDE_CLI_BIN=claude-custom",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "open_router/test-model"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "CLAUDE_WORKSPACE=C:/custom/workspace" in text
    assert "CLAUDE_CLI_BIN=claude-custom" in text


def test_admin_apply_restart_required_reports_automatic_restart(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    callbacks: list[str] = []

    def restart_callback() -> None:
        callbacks.append("restart")

    app = create_test_app(restart_callback=restart_callback)
    instance_id = _local_client(app).get("/admin/api/status").json()["instance_id"]

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PORT": "9090"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == ["PORT"]
    assert body["restart"] == {
        "required": True,
        "automatic": True,
        "admin_url": "http://127.0.0.1:9090/admin",
        "fields": ["PORT"],
        "instance_id": instance_id,
    }
    assert callbacks == ["restart"]


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8082", "http://localhost:9090"])
def test_admin_restart_status_can_be_read_from_another_local_address(origin):
    app = create_test_app()
    response = _local_client(app).get("/admin/api/status", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == origin
    assert response.headers["Vary"] == "Origin"
    initial = response.json()
    assert initial["status"] == "running"
    assert isinstance(initial["instance_id"], str) and initial["instance_id"]
    runtime_for_app(app).begin_shutdown()
    stopping = _local_client(app).get("/admin/api/status").json()
    assert stopping["status"] == "stopping"
    assert stopping["instance_id"] == initial["instance_id"]


def test_admin_restart_status_does_not_allow_a_remote_web_origin():
    response = _local_client(create_test_app()).get(
        "/admin/api/status", headers={"Origin": "https://example.com"}
    )
    assert response.status_code == 403
    assert "Access-Control-Allow-Origin" not in response.headers


def test_admin_apply_restart_required_reports_manual_fallback(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PORT": "9091"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == ["PORT"]
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": ["PORT"],
    }


def test_restart_signal_failure_reports_saved_config_and_manual_restart(
    monkeypatch, tmp_path, caplog
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)

    def restart_callback() -> None:
        raise RuntimeError("private restart failure detail")

    app = create_test_app(restart_callback=restart_callback)
    client = _local_client(app)
    original_port = client.get("/admin/api/status").json()["port"]
    response = client.post("/admin/api/config/apply", json={"values": {"PORT": "9090"}})

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": ["PORT"],
    }
    assert "PORT=9090" in (tmp_path / ".fcc" / ".env").read_text()
    status = client.get("/admin/api/status").json()
    assert status["port"] == original_port
    assert status["pending_fields"] == ["PORT"]
    assert "private restart failure detail" not in response.text + caplog.text


@pytest.mark.parametrize(
    "followup",
    [{"PORT": "9090"}, {"MODEL": "nvidia_nim/new"}, {"LOG_LEVEL": "WARNING"}],
)
@pytest.mark.parametrize("signal_recovers", [False, True])
def test_pending_restart_survives_later_apply(
    monkeypatch, tmp_path, followup, signal_recovers
):
    from unittest.mock import MagicMock

    from free_claude_code.config.loader import ManagedConfigStore

    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    store = ManagedConfigStore()
    store.initialize()
    settings = store.read().settings
    callback = MagicMock(
        side_effect=[
            RuntimeError("first signal failed"),
            None if signal_recovers else RuntimeError("still unavailable"),
        ]
    )
    app = create_test_app(settings, restart_callback=callback)
    client = _local_client(app)
    assert client.post(
        "/admin/api/config/apply", json={"values": {"PORT": "9090"}}
    ).json()["restart"]["required"]
    result = client.post("/admin/api/config/apply", json={"values": followup}).json()
    fields = ["PORT", "LOG_LEVEL"] if "LOG_LEVEL" in followup else ["PORT"]
    assert result["applied"] is True
    assert result["restart"]["required"] is True
    assert result["restart"]["automatic"] is signal_recovers
    assert result["restart"]["fields"] == fields
    assert callback.call_count == 2
    status = client.get("/admin/api/status").json()
    assert status["port"] == settings.port
    assert status["pending_fields"] == ([] if signal_recovers else fields)
    assert provider_manager_for_app(app).current_generation_id == 1


def test_reverting_pending_restart_restores_hot_apply(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from free_claude_code.config.loader import ManagedConfigStore

    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    store = ManagedConfigStore()
    store.initialize()
    settings = store.read().settings
    callback = MagicMock(side_effect=RuntimeError("restart failed"))
    app = create_test_app(settings, restart_callback=callback)
    client = _local_client(app)
    client.post("/admin/api/config/apply", json={"values": {"PORT": "9090"}})
    with patch.object(
        provider_manager_for_app(app), "_refresh_generation_in_background", AsyncMock()
    ):
        result = client.post(
            "/admin/api/config/apply",
            json={"values": {"PORT": str(settings.port), "MODEL": "nvidia_nim/new"}},
        ).json()
    assert result["applied"] is True
    assert result["restart"]["required"] is False
    assert callback.call_count == 1
    status = client.get("/admin/api/status").json()
    assert status["pending_fields"] == []
    assert status["port"] == settings.port
    assert status["model"] == "nvidia_nim/new"


def test_admin_process_env_values_are_locked_and_not_written(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.setenv("MODEL", "open_router/process-model")
    app = create_test_app()

    config = _local_client(app).get("/admin/api/config").json()
    model_field = next(field for field in config["fields"] if field["key"] == "MODEL")
    assert model_field["locked"] is True
    assert model_field["source"] == "process"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "deepseek/managed-model"}},
    )

    assert response.status_code == 200
    env_file = tmp_path / ".fcc" / ".env"
    assert "deepseek/managed-model" not in env_file.read_text("utf-8")


def test_admin_process_model_fallbacks_are_locked_and_not_written(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.setenv("MODEL_FALLBACKS", "open_router/process-model")
    app = create_test_app()

    config = _local_client(app).get("/admin/api/config").json()
    field = next(item for item in config["fields"] if item["key"] == "MODEL_FALLBACKS")
    assert field["locked"] is True
    assert field["source"] == "process"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL_FALLBACKS": "groq/managed-model"}},
    )

    assert response.status_code == 200
    env_file = tmp_path / ".fcc" / ".env"
    assert "MODEL_FALLBACKS=" not in env_file.read_text("utf-8")


def test_admin_never_reads_arbitrary_current_directory_env(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MODEL=deepseek/deepseek-chat\nDEEPSEEK_API_KEY=deepseek-secret\n",
        encoding="utf-8",
    )
    app = create_test_app()

    config = _local_client(app).get("/admin/api/config").json()
    model_field = next(field for field in config["fields"] if field["key"] == "MODEL")
    assert model_field["value"] == "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    assert model_field["source"] == "default"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {}},
    )

    assert response.status_code == 200
    managed_text = (tmp_path / ".fcc" / ".env").read_text("utf-8")
    assert "MODEL=deepseek/deepseek-chat" not in managed_text
    assert "DEEPSEEK_API_KEY=deepseek-secret" not in managed_text


def test_admin_migration_removes_blank_required_managed_values(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "MESSAGING_PLATFORM=\nWHISPER_DEVICE=\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"NVIDIA_NIM_API_KEY": "nim-secret"}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    managed_text = (tmp_path / ".fcc" / ".env").read_text("utf-8")
    assert "NVIDIA_NIM_API_KEY=nim-secret" in managed_text
    assert "MESSAGING_PLATFORM=" not in managed_text
    assert "WHISPER_DEVICE=" not in managed_text


def test_admin_migrates_empty_auth_token_to_explicit_disabled_state(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "ANTHROPIC_AUTH_TOKEN=\nPROVIDER_RATE_LIMIT=1\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PROVIDER_RATE_LIMIT": "2"}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    managed_lines = env_file.read_text("utf-8").splitlines()
    assert not any(line.startswith("ANTHROPIC_AUTH_TOKEN=") for line in managed_lines)
    assert "PROXY_AUTH_ENABLED=false" in managed_lines
    assert "PROVIDER_RATE_LIMIT=2" in managed_lines


@pytest.mark.parametrize(
    ("submitted", "message"),
    [
        (None, "this setting cannot be removed"),
        ("", "this setting cannot be blank"),
        ("   ", "this setting cannot be blank"),
    ],
)
def test_admin_apply_rejects_missing_required_value_without_writing(
    monkeypatch,
    tmp_path,
    submitted,
    message,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    original = "ANTHROPIC_AUTH_TOKEN=\nMESSAGING_PLATFORM=none\n"
    env_file.write_text(original, encoding="utf-8")
    app = create_test_app()
    _local_client(app).get("/admin/api/config")
    baseline = env_file.read_bytes()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MESSAGING_PLATFORM": submitted}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert f"MESSAGING_PLATFORM: {message}" in body["errors"]
    assert env_file.read_bytes() == baseline


def test_admin_apply_preserves_false_and_numeric_zero(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "FCC_OPEN_BROWSER": False,
                "HTTP_WRITE_TIMEOUT": 0,
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    managed = (tmp_path / ".fcc" / ".env").read_text(encoding="utf-8")
    assert "FCC_OPEN_BROWSER=false" in managed
    assert "HTTP_WRITE_TIMEOUT=0" in managed


def test_admin_local_provider_status_reports_reachable(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            return httpx.Response(200, json={"data": []})

    with patch("free_claude_code.api.admin_routes.httpx.AsyncClient", FakeAsyncClient):
        response = _local_client(app).get("/admin/api/providers/local-status")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert {provider["status"] for provider in providers} == {"reachable"}


def test_admin_local_provider_status_checks_all_providers_concurrently(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()
    calls = 0
    active = 0
    max_active = 0

    class SlowAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            nonlocal active, calls, max_active
            calls += 1
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json={"data": []})

    with patch("free_claude_code.api.admin_routes.httpx.AsyncClient", SlowAsyncClient):
        response = _local_client(app).get("/admin/api/providers/local-status")

    assert response.status_code == 200
    assert calls == 3
    assert max_active == 3


def test_admin_config_exposes_structured_provider_configuration_targets(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)

    response = _local_client(create_test_app()).get("/admin/api/config")

    assert response.status_code == 200
    providers = {
        provider["provider_id"]: provider
        for provider in response.json()["provider_status"]
    }
    assert providers["nvidia_nim"]["configuration_keys"] == ["NVIDIA_NIM_API_KEY"]
    assert providers["nvidia_nim"]["missing_configuration_keys"] == [
        "NVIDIA_NIM_API_KEY"
    ]
    assert "configuration" not in providers["nvidia_nim"]
    assert providers["lmstudio"]["status"] == "configured"
    assert providers["lmstudio"]["configuration_keys"] == ["LM_STUDIO_BASE_URL"]


def test_admin_local_provider_failure_does_not_return_exception_text(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    class FailingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            raise RuntimeError(
                "Provider rejected credential CREDENTIAL[unrecognized-format-987654321]"
            )

    with patch(
        "free_claude_code.api.admin_routes.httpx.AsyncClient",
        return_value=FailingAsyncClient(),
    ):
        response = _local_client(app).get("/admin/api/providers/local-status")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert {provider["status"] for provider in providers} == {"offline"}
    for provider in providers:
        assert provider["message"] == (
            "Could not connect. Verify the URL and that the local provider is running."
        )
        assert "CREDENTIAL[unrecognized-format-987654321]" not in provider["message"]
        assert "RuntimeError" not in provider["message"]
        assert "error_type" not in provider


@pytest.mark.parametrize(
    "result",
    (
        {"provider_id": "nvidia_nim", "ok": True, "models": ["model-a"]},
        {
            "provider_id": "nvidia_nim",
            "ok": False,
            "message": "NVIDIA_NIM_API_KEY is not set.",
        },
    ),
)
def test_admin_provider_test_route_preserves_runtime_result(
    monkeypatch, tmp_path, result
):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()
    runtime = runtime_for_app(app)

    with patch.object(runtime, "test_provider", new=AsyncMock(return_value=result)):
        response = _local_client(app).post(
            "/admin/api/providers/nvidia_nim/test",
            json={},
        )

    assert response.status_code == 200
    assert response.json() == result


def test_admin_launch_url_uses_loopback_for_wildcard_host():
    settings = Settings.model_construct(host="0.0.0.0", port=8082)

    assert local_admin_url(settings) == "http://127.0.0.1:8082/admin"
