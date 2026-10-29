from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.app import create_app


def _client():
    return TestClient(create_app(lifespan_enabled=False))


# ── llms.txt ─────────────────────────────────────────────────────────


def test_llms_txt_is_public_and_markdown():
    r = _client().get("/llms.txt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert body.startswith("# Free Claude Code")
    assert "/v1/messages" in body
    assert "/.well-known/" in body


def test_llms_full_txt_is_public_and_expanded():
    r = _client().get("/llms-full.txt")
    assert r.status_code == 200
    body = r.text
    assert "Full reference" in body or "full reference" in body
    assert "EvoMetaClaw" in body
    assert "SkillOpt" in body


# ── /.well-known/ manifests ──────────────────────────────────────────


def test_ai_plugin_manifest_shape():
    r = _client().get("/.well-known/ai-plugin.json")
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "v1"
    assert body["name_for_model"] == "free_claude_code"
    assert body["api"]["type"] == "openapi"
    assert body["api"]["url"].endswith("/openapi.json")


def test_agents_manifest_lists_capabilities():
    r = _client().get("/.well-known/agents.json")
    assert r.status_code == 200
    body = r.json()
    names = {cap["name"] for cap in body["capabilities"]}
    assert "anthropic.messages" in names
    assert "anthropic.count_tokens" in names
    assert "anthropic.models" in names
    # ecosystem field surfaces our provider list for agent discovery
    assert "openai" in body["ecosystem"]["upstream_providers"]
    assert "claude-code" in body["ecosystem"]["clients"]


def test_mcp_manifest_points_to_openapi_and_docs():
    r = _client().get("/.well-known/mcp.json")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "openapi-service"
    assert body["openapi"].endswith("/openapi.json")
    assert body["docs"]["llms_txt"].endswith("/llms.txt")


# ── robots.txt ───────────────────────────────────────────────────────


def test_robots_txt_welcomes_ai_crawlers_and_blocks_admin():
    r = _client().get("/robots.txt")
    assert r.status_code == 200
    body = r.text
    assert "Disallow: /admin" in body
    assert "GPTBot" in body
    assert "ClaudeBot" in body


# ── OpenAPI reflects the renamed title ───────────────────────────────


def test_openapi_has_agent_friendly_title():
    r = _client().get("/openapi.json")
    assert r.status_code == 200
    spec = json.loads(r.text)
    assert "Free Claude Code" in spec["info"]["title"]
    assert spec["info"]["version"] == "2.4.0"


# ── public — no auth needed, no proxy env to set ─────────────────────


def test_discovery_endpoints_are_unauthenticated(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "shh")
    client = TestClient(create_app(lifespan_enabled=False))
    # auth-required routes return 401 without a token
    assert client.get("/v1/models").status_code in (401, 403)
    # discovery routes still return 200
    assert client.get("/llms.txt").status_code == 200
    assert client.get("/.well-known/agents.json").status_code == 200
