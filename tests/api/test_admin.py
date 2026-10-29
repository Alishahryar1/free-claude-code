from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api.admin_config import MASKED_SECRET
from api.admin_urls import local_admin_url
from api.app import create_app
from config.settings import Settings


def _local_client(app):
    return TestClient(app, client=("127.0.0.1", 50000))


def _set_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _clear_process_config(monkeypatch) -> None:
    for key in (
        "MODEL",
        "NVIDIA_NIM_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "FCC_ENV_FILE",
        "HOST",
        "PORT",
        "LOG_FILE",
        "ZAI_BASE_URL",
        "CLAUDE_WORKSPACE",
        "CLAUDE_CLI_BIN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_admin_page_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    assert _local_client(app).get("/admin").status_code == 200
    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin").status_code == 403


def test_admin_page_no_longer_renders_generated_env_panel(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin")

    assert response.status_code == 200
    assert "Generated Env" not in response.text
    assert "envPreview" not in response.text


def test_admin_page_no_longer_renders_global_status_header(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin")

    assert response.status_code == 200
    assert "Local Admin" not in response.text
    assert "serverStatus" not in response.text
    assert "modelBadge" not in response.text


def test_admin_static_no_longer_fetches_global_status_header():
    script = Path("api/admin_static/admin.js").read_text(encoding="utf-8")

    assert 'api("/admin/api/status")' not in script
    assert "updateHeader" not in script
    assert '"Running"' not in script
    assert "serverStatus" not in script
    assert "modelBadge" not in script


def test_admin_static_hides_managed_source_label():
    script = Path("api/admin_static/admin.js").read_text(encoding="utf-8")

    assert 'managed_env: "",' in script
    assert "hasOwnProperty.call(labels, source)" in script
    assert 'parts.push("locked")' in script
    assert "sourceEl.textContent = source" in script


def test_admin_config_masks_secrets_and_exposes_manifest(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    keys = {field["key"] for field in body["fields"]}
    assert "ANTHROPIC_AUTH_TOKEN" in keys
    assert "OPENROUTER_API_KEY" in keys
    assert "FIREWORKS_API_KEY" in keys
    assert "GEMINI_API_KEY" in keys
    assert "GROQ_API_KEY" in keys
    assert "CEREBRAS_API_KEY" in keys
    assert "ZAI_BASE_URL" not in keys
    assert "CLAUDE_WORKSPACE" not in keys
    assert "CLAUDE_CLI_BIN" not in keys
    assert "LOG_FILE" not in keys
    auth_field = next(
        field for field in body["fields"] if field["key"] == "ANTHROPIC_AUTH_TOKEN"
    )
    assert auth_field["secret"] is True
    assert auth_field["value"] == MASKED_SECRET
    assert auth_field["source"] == "template"


def test_admin_config_preserves_managed_env_source_contract(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("MODEL=open_router/managed-model\n", encoding="utf-8")
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    model_field = next(field for field in body["fields"] if field["key"] == "MODEL")
    assert model_field["source"] == "managed_env"
    assert model_field["locked"] is False


def test_admin_validate_rejects_bad_model_shape(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).post(
        "/admin/api/config/validate",
        json={"values": {"MODEL": "missing-provider-prefix"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert any("provider type" in error for error in body["errors"])


def test_admin_apply_writes_complete_managed_env_and_masks_preview(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

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
    assert "ANTHROPIC_AUTH_TOKEN=" in text
    assert body["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }


def test_admin_apply_writes_fireworks_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

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
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "gemini/gemini-2.5-flash",
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
    assert "MODEL=gemini/gemini-2.5-flash" in text
    assert "GEMINI_API_KEY=gm-secret" in text


def test_admin_apply_writes_groq_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

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


def test_admin_apply_writes_cerebras_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

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
    app = create_app(lifespan_enabled=False)

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


def test_admin_apply_omits_stale_zai_base_url(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=zai/glm-5.1",
                "ZAI_API_KEY=zai-secret",
                "ZAI_BASE_URL=https://custom.zai.invalid/v1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "zai/glm-5.1"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "ZAI_API_KEY=zai-secret" in text
    assert "ZAI_BASE_URL" not in text


def test_admin_apply_omits_stale_fixed_claude_runtime_settings(monkeypatch, tmp_path):
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
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "open_router/test-model"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "CLAUDE_WORKSPACE" not in text
    assert "CLAUDE_CLI_BIN" not in text


def test_admin_apply_restart_required_reports_automatic_restart(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)
    callbacks: list[str] = []

    async def restart_callback() -> None:
        callbacks.append("restart")

    app.state.admin_restart_callback = restart_callback

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
    }
    assert callbacks == ["restart"]


def test_admin_apply_restart_required_reports_manual_fallback(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

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


def test_admin_process_env_values_are_locked_and_not_written(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.setenv("MODEL", "open_router/process-model")
    app = create_app(lifespan_enabled=False)

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


def test_admin_first_apply_migrates_repo_env(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MODEL=deepseek/deepseek-chat\nDEEPSEEK_API_KEY=deepseek-secret\n",
        encoding="utf-8",
    )
    app = create_app(lifespan_enabled=False)

    config = _local_client(app).get("/admin/api/config").json()
    model_field = next(field for field in config["fields"] if field["key"] == "MODEL")
    assert model_field["value"] == "deepseek/deepseek-chat"
    assert model_field["source"] == "repo_env"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {}},
    )

    assert response.status_code == 200
    managed_text = (tmp_path / ".fcc" / ".env").read_text("utf-8")
    assert "MODEL=deepseek/deepseek-chat" in managed_text
    assert "DEEPSEEK_API_KEY=deepseek-secret" in managed_text


def test_admin_local_provider_status_reports_reachable(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_app(lifespan_enabled=False)

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            return httpx.Response(200, json={"data": []})

    with patch("api.admin_routes.httpx.AsyncClient", FakeAsyncClient):
        response = _local_client(app).get("/admin/api/providers/local-status")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert {provider["status"] for provider in providers} == {"reachable"}


def test_admin_launch_url_uses_loopback_for_wildcard_host():
    settings = Settings.model_construct(host="0.0.0.0", port=8082)

    assert local_admin_url(settings) == "http://127.0.0.1:8082/admin"


def test_admin_metrics_endpoint_returns_empty_store(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "requests" in body
    assert "summary" in body
    assert isinstance(body["requests"], list)
    assert body["summary"]["total"] == 0
    assert body["summary"]["avg_latency_ms"] == 0.0
    assert body["summary"]["total_input_tokens"] == 0
    assert body["summary"]["total_output_tokens"] == 0
    assert body["summary"]["total_cost_usd"] == 0.0
    assert body["summary"]["per_provider_cost_usd"] == {}


def test_admin_trajectory_endpoint_returns_disabled_summary(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.delenv("TRAJECTORY_LOG_ENABLED", raising=False)
    from api import trajectory

    trajectory.reload_config()
    trajectory.clear()
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/trajectory")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["total"] == 0
    assert body["per_skill"] == {}


def test_admin_trajectory_endpoint_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin/api/trajectory").status_code == 403


def test_admin_skillopt_endpoint_returns_disabled_snapshot(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.delenv("SKILLOPT_ENABLED", raising=False)
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    from api import skillopt

    skillopt.invalidate_cache()
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/skillopt")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["loaded"] is False


def test_admin_skillopt_endpoint_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin/api/skillopt").status_code == 403


def test_admin_auto_forge_endpoint_returns_state(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.delenv("AUTO_FORGE_ENABLED", raising=False)
    from api import auto_forge

    auto_forge.reset_for_tests()
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/auto-forge")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["running"] is False
    assert body["last_run_status"] == "never"


def test_admin_auto_forge_endpoint_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin/api/auto-forge").status_code == 403


def test_admin_forge_now_runs_and_invalidates_cache(monkeypatch, tmp_path):
    """POST /admin/api/skillopt/forge runs EvoForge immediately."""
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))

    # Seed a small trajectory corpus so EvoForge has something to score.
    import json as _json

    (tmp_path / "trajectories.jsonl").write_text(
        "\n".join(
            _json.dumps(
                {
                    "provider_id": "openai",
                    "model": "gpt-4o",
                    "skill": "edit",
                    "latency_ms": 500,
                    "cost_usd": 0.001,
                    "status": "ok",
                    "input_tokens": 100,
                    "output_tokens": 100,
                }
            )
            for _ in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    from api import auto_forge, skillopt

    auto_forge.reset_for_tests()
    skillopt.invalidate_cache()

    app = create_app(lifespan_enabled=False)
    response = _local_client(app).post("/admin/api/skillopt/forge")

    assert response.status_code == 200
    body = response.json()
    assert body["last_run_status"] == "ok"
    assert body["last_policy_skills"] == 1
    assert (tmp_path / "skillopt_policy.json").is_file()


def test_admin_forge_now_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert (
        remote_client.post("/admin/api/skillopt/forge").status_code == 403
    )


def test_admin_metrics_endpoint_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin/api/metrics").status_code == 403


def test_admin_health_history_endpoint_returns_empty(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    from api import health_history

    health_history.clear()
    app = create_app(lifespan_enabled=False)

    response = _local_client(app).get("/admin/api/health-history")

    assert response.status_code == 200
    body = response.json()
    assert body == {"providers": {}}


def test_admin_health_history_endpoint_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)

    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin/api/health-history").status_code == 403


def test_health_history_records_and_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    # Re-import to pick up the new cache dir
    import importlib

    from api import health_history

    importlib.reload(health_history)
    health_history.clear()

    health_history.record(provider_id="openai", status="ok", latency_ms=125.4)
    health_history.record(
        provider_id="openai", status="error", latency_ms=2100.0, error_type="Timeout"
    )

    snapshot = health_history.snapshot()
    assert "openai" in snapshot
    assert len(snapshot["openai"]) == 2
    assert snapshot["openai"][0]["status"] == "ok"
    assert snapshot["openai"][1]["error_type"] == "Timeout"

    # Persisted file should exist
    persisted = tmp_path / "health_history.json"
    assert persisted.exists()

    # Reload picks up persisted entries
    importlib.reload(health_history)
    snapshot2 = health_history.snapshot()
    assert len(snapshot2["openai"]) == 2


def test_health_history_caps_per_provider_buffer(monkeypatch, tmp_path):
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    import importlib

    from api import health_history

    importlib.reload(health_history)
    health_history.clear()

    for i in range(80):
        health_history.record(provider_id="openai", status="ok", latency_ms=float(i))

    snapshot = health_history.snapshot()
    # Bounded to _MAX_ENTRIES_PER_PROVIDER (50)
    assert len(snapshot["openai"]) == 50
    # Oldest dropped — first entry is i=30
    assert snapshot["openai"][0]["latency_ms"] == 30.0


# ── Provider credential validation ─────────────────────────────────────────


from providers.base import BaseProvider as _BaseProvider  # noqa: E402
from providers.base import ProviderConfig as _ProviderConfig  # noqa: E402

_STUB_CONFIG = _ProviderConfig(api_key="stub", base_url="http://stub.invalid")


class _StubValidatingProvider(_BaseProvider):
    """Minimal BaseProvider subclass returning a pre-canned validate result."""

    def __init__(self, result: dict) -> None:
        super().__init__(_STUB_CONFIG)
        self._stub_result = result
        self.calls: list[dict] = []

    async def cleanup(self) -> None:
        return None

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset({"stub-model"})

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ):
        if False:
            yield ""

    async def validate_credentials(self, preferred_model: str | None = None) -> dict:
        self.calls.append({"preferred_model": preferred_model})
        return self._stub_result


def _inject_stub_provider(app, provider_id, result) -> tuple:
    """Insert a stub provider directly into the registry to bypass real I/O.

    Returns ``(registry, stub)`` so callers can inspect call records.
    """
    from providers.registry import ProviderRegistry

    stub = _StubValidatingProvider(result)
    registry = ProviderRegistry(providers={provider_id: stub})
    app.state.provider_registry = registry
    return registry, stub


def test_validate_endpoint_full_pass(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    from api import health_history

    health_history.clear()
    app = create_app(lifespan_enabled=False)
    _inject_stub_provider(
        app,
        "openai",
        {
            "auth_ok": True,
            "completion_ok": True,
            "models_count": 12,
            "test_model": "stub-model",
        },
    )

    response = _local_client(app).post("/admin/api/providers/openai/validate")

    assert response.status_code == 200
    body = response.json()
    assert body["provider_id"] == "openai"
    assert body["auth_ok"] is True
    assert body["completion_ok"] is True
    assert body["test_model"] == "stub-model"
    # health history records the outcome
    history = health_history.snapshot().get("openai", [])
    assert len(history) == 1
    assert history[0]["status"] == "ok"


def test_validate_endpoint_auth_ok_but_completion_fails(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    from api import health_history

    health_history.clear()
    app = create_app(lifespan_enabled=False)
    _inject_stub_provider(
        app,
        "openai",
        {
            "auth_ok": True,
            "completion_ok": False,
            "models_count": 12,
            "test_model": "stub-model",
            "error_type": "BadRequestError",
        },
    )

    response = _local_client(app).post("/admin/api/providers/openai/validate")

    assert response.status_code == 200
    body = response.json()
    assert body["auth_ok"] is True
    assert body["completion_ok"] is False
    assert body["error_type"] == "BadRequestError"
    # Partial pass (completion failed) → health history records as error
    history = health_history.snapshot().get("openai", [])
    assert history[0]["status"] == "error"


def test_validate_endpoint_auth_failure(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    from api import health_history

    health_history.clear()
    app = create_app(lifespan_enabled=False)
    _inject_stub_provider(
        app,
        "openai",
        {
            "auth_ok": False,
            "completion_ok": False,
            "models_count": 0,
            "error_type": "AuthenticationError",
        },
    )

    response = _local_client(app).post("/admin/api/providers/openai/validate")

    assert response.status_code == 200
    body = response.json()
    assert body["auth_ok"] is False
    assert body["error_type"] == "AuthenticationError"


def test_validate_endpoint_exception_returns_auth_false(monkeypatch, tmp_path):
    """Exception raised by validate_credentials must surface as auth_ok=False."""
    _set_home(monkeypatch, tmp_path)
    from api import health_history

    health_history.clear()
    app = create_app(lifespan_enabled=False)

    class _RaisingProvider(_StubValidatingProvider):
        def __init__(self):
            super().__init__(result={})

        async def validate_credentials(self, preferred_model=None):
            raise ConnectionError("network unreachable")

    from providers.registry import ProviderRegistry

    registry = ProviderRegistry(providers={"openai": _RaisingProvider()})
    app.state.provider_registry = registry

    response = _local_client(app).post("/admin/api/providers/openai/validate")

    assert response.status_code == 200
    body = response.json()
    assert body["provider_id"] == "openai"
    assert body["auth_ok"] is False
    assert body["completion_ok"] is False
    assert body["error_type"] == "ConnectionError"
    # Recorded as error in health history
    history = health_history.snapshot().get("openai", [])
    assert len(history) == 1
    assert history[0]["status"] == "error"
    assert history[0]["error_type"] == "ConnectionError"


def test_validate_endpoint_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_app(lifespan_enabled=False)
    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.post("/admin/api/providers/openai/validate").status_code == 403


def test_validate_endpoint_forwards_configured_model(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setenv("MODEL", "openai/gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Bust the settings lru_cache so the new MODEL env var is picked up
    from config.settings import get_settings

    get_settings.cache_clear()
    try:
        app = create_app(lifespan_enabled=False)
        stub_result = {
            "auth_ok": True,
            "completion_ok": True,
            "models_count": 5,
            "test_model": "gpt-4o",
        }
        _registry, stub = _inject_stub_provider(app, "openai", stub_result)

        response = _local_client(app).post("/admin/api/providers/openai/validate")

        assert response.status_code == 200
        assert stub.calls == [{"preferred_model": "gpt-4o"}]
    finally:
        get_settings.cache_clear()


def test_base_provider_validate_credentials_default_impl():
    """Default implementation should report auth_ok based on list_model_ids() success."""
    import asyncio

    class _DummyProvider:
        async def list_model_ids(self):
            return frozenset({"a", "b", "c"})

        # Bound to BaseProvider's default impl
        from providers.base import BaseProvider

        validate_credentials = BaseProvider.validate_credentials

    result = asyncio.run(_DummyProvider().validate_credentials())
    assert result["auth_ok"] is True
    assert result["models_count"] == 3
    assert result["completion_ok"] is None


def test_base_provider_validate_credentials_default_impl_auth_failure():
    import asyncio

    class _FailingProvider:
        async def list_model_ids(self):
            raise PermissionError("invalid key")

        from providers.base import BaseProvider

        validate_credentials = BaseProvider.validate_credentials

    result = asyncio.run(_FailingProvider().validate_credentials())
    assert result["auth_ok"] is False
    assert result["models_count"] == 0
    assert result["error_type"] == "PermissionError"
