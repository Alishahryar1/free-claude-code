"""Agent-discovery routes.

These endpoints are public (no auth required) and expose a stable,
machine-readable description of what this proxy *is* so AI agents and
crawlers can discover and use it without a human in the loop:

    GET /llms.txt                      — compact Markdown summary
    GET /llms-full.txt                 — expanded Markdown reference
    GET /.well-known/ai-plugin.json    — ChatGPT-style plugin manifest
    GET /.well-known/agents.json       — generic agent manifest
    GET /.well-known/mcp.json          — MCP discovery pointer
    GET /robots.txt                    — agent-crawler-friendly rules

The surface is intentionally small: everything else about the proxy
remains behind the Anthropic auth token. These routes only describe
*what* is available and *how to call it*, never the admin UI, env
values, or trajectories.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter(include_in_schema=False)

_SERVICE_NAME = "Free Claude Code — Anthropic-compatible model gateway"
_SERVICE_DESCRIPTION = (
    "Drop-in Anthropic Messages API proxy that routes Claude Code CLI, "
    "VS Code, JetBrains ACP, and bot traffic to 18 provider backends "
    "(OpenAI, DeepSeek, Groq, Cerebras, Gemini, Mistral, NVIDIA NIM, "
    "OpenRouter, Fireworks, Kimi, Wafer, Z.ai, OpenCode Zen / Go, and "
    "local LM Studio / llama.cpp / Ollama)."
)
_REPO_URL = "https://github.com/dnzengou/free-claude-code"
_LIVE_URL = "https://free-claude-code-main-ebon.vercel.app"


def _base_url(request: Request) -> str:
    """Public base URL for this instance, honoring the proxy's host header."""
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get(
        "host"
    )
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    if forwarded_host:
        return f"{scheme}://{forwarded_host}"
    return str(request.base_url).rstrip("/")


@router.get("/llms.txt")
async def llms_txt(request: Request) -> PlainTextResponse:
    """Compact LLM-friendly service summary.

    Follows the emerging llms.txt convention
    (https://llmstxt.org): the first thing an AI agent reads to decide
    whether and how to use this service.
    """
    base = _base_url(request)
    body = f"""# Free Claude Code

> {_SERVICE_DESCRIPTION}

- Live: {_LIVE_URL}
- Repo: {_REPO_URL}
- API shape: Anthropic Messages API ({base}/v1/messages)
- Auth: `Authorization: Bearer <ANTHROPIC_AUTH_TOKEN>` (or `x-api-key`)

## Public endpoints

- POST {base}/v1/messages — streaming + non-streaming completions
- POST {base}/v1/messages/count_tokens — token count
- GET  {base}/v1/models — gateway model discovery
- GET  {base}/health, {base}/health/ready — probes

## Discovery

- {base}/.well-known/ai-plugin.json — ChatGPT-style plugin manifest
- {base}/.well-known/agents.json — generic agent manifest
- {base}/.well-known/mcp.json — MCP pointer
- {base}/llms-full.txt — expanded reference

## Usage from an agent

Call `POST /v1/messages` with the standard Anthropic Messages request
body (`model`, `max_tokens`, `messages`, optional `tools`, `system`).
The gateway rewrites the upstream provider; the request/response shape
on the wire is unchanged from the Anthropic SDK's expectation.
"""
    return PlainTextResponse(body, media_type="text/plain; charset=utf-8")


@router.get("/llms-full.txt")
async def llms_full_txt(request: Request) -> PlainTextResponse:
    """Expanded reference — the full capability description for an agent."""
    base = _base_url(request)
    body = f"""# Free Claude Code — full reference

{_SERVICE_DESCRIPTION}

## Why this exists

Claude Code (and compatible clients: VS Code, JetBrains ACP, Discord /
Telegram bot wrappers) speaks the Anthropic Messages API. This proxy
accepts that same wire format and routes the upstream call to any of
18 provider backends. Clients see no difference; the operator picks
cost, latency, and provider freely.

## Authentication

All non-discovery endpoints require either:

    Authorization: Bearer <token>
    x-api-key: <token>

The token is whatever ``ANTHROPIC_AUTH_TOKEN`` is set to on the
instance. When unset, no auth is required.

## Core API

### POST {base}/v1/messages

Anthropic Messages request body:

```
{{
  "model": "claude-3-opus-20240229",
  "max_tokens": 1024,
  "messages": [{{"role": "user", "content": "hi"}}],
  "tools": [...],
  "system": "...",
  "stream": true
}}
```

Streaming response is SSE in Anthropic's event format
(`message_start`, `content_block_delta`, `message_delta`, …). Tool use
and thinking blocks pass through normalized.

Model routing:

- `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU` map Claude tiers to
  specific `provider/model` upstreams.
- Direct slugs (`provider/model` in the model field) always win.
- When SkillOpt is enabled, routing further narrows per inferred skill
  (`edit`, `plan`, `probe`, `question`, `chat`).

### POST {base}/v1/messages/count_tokens

Returns `{{\"input_tokens\": N}}` for a given request body (no upstream
call made).

### GET {base}/v1/models

Lists Anthropic Claude model ids plus the configured gateway models
when the client opts into Gateway model discovery
(`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`).

## EvoMetaClaw × EvoForge × SkillOpt

This gateway can additionally capture a trajectory JSONL
(`TRAJECTORY_LOG_ENABLED=1`), process it offline with EvoForge
(`uv run python scripts/evoforge.py`), and have SkillOpt
(`SKILLOPT_ENABLED=1`) consult the resulting policy to narrow routing
per skill. All three stages are opt-in; the gateway behaves identically
to a plain proxy with all flags off.

## Clients

- Claude Code CLI: `ANTHROPIC_BASE_URL={base}`
- VS Code claude-code extension: `claudeCode.environmentVariables`
- JetBrains ACP: edit `~/.jetbrains/acp.json` → `acp.registry.claude-acp.env`

## Repository

{_REPO_URL}
"""
    return PlainTextResponse(body, media_type="text/plain; charset=utf-8")


@router.get("/.well-known/ai-plugin.json")
async def ai_plugin_manifest(request: Request) -> JSONResponse:
    """ChatGPT-style plugin manifest.

    Compatible with the OpenAI plugin discovery format. Agents that
    follow the ``/.well-known/ai-plugin.json`` convention can read this
    to learn the service's purpose, auth model, and OpenAPI spec URL.
    """
    base = _base_url(request)
    return JSONResponse(
        {
            "schema_version": "v1",
            "name_for_human": "Free Claude Code",
            "name_for_model": "free_claude_code",
            "description_for_human": _SERVICE_DESCRIPTION,
            "description_for_model": (
                "Anthropic Messages API proxy. Agents can send Claude-style "
                "message requests to POST /v1/messages and receive streaming or "
                "non-streaming responses. The upstream model (OpenAI, DeepSeek, "
                "Groq, Cerebras, Gemini, etc.) is chosen by the operator and the "
                "proxy keeps the Anthropic wire format."
            ),
            "auth": {
                "type": "user_http",
                "authorization_type": "bearer",
            },
            "api": {
                "type": "openapi",
                "url": f"{base}/openapi.json",
            },
            "logo_url": "",
            "contact_email": "",
            "legal_info_url": _REPO_URL,
        }
    )


@router.get("/.well-known/agents.json")
async def agents_manifest(request: Request) -> JSONResponse:
    """Generic agent discovery manifest.

    Not tied to a single vendor's plugin format — a plain capability
    description an agent runtime can parse to decide whether this
    service is relevant to a user's task.
    """
    base = _base_url(request)
    return JSONResponse(
        {
            "name": _SERVICE_NAME,
            "description": _SERVICE_DESCRIPTION,
            "homepage": _LIVE_URL,
            "repository": _REPO_URL,
            "docs": [
                f"{base}/llms.txt",
                f"{base}/llms-full.txt",
                f"{base}/openapi.json",
            ],
            "capabilities": [
                {
                    "name": "anthropic.messages",
                    "method": "POST",
                    "path": "/v1/messages",
                    "description": "Anthropic Messages API — streaming + tools",
                },
                {
                    "name": "anthropic.count_tokens",
                    "method": "POST",
                    "path": "/v1/messages/count_tokens",
                },
                {
                    "name": "anthropic.models",
                    "method": "GET",
                    "path": "/v1/models",
                },
            ],
            "auth": {
                "schemes": ["bearer", "api-key"],
                "optional_env": ["ANTHROPIC_AUTH_TOKEN"],
            },
            "ecosystem": {
                "clients": [
                    "claude-code",
                    "vscode-claude-code-extension",
                    "jetbrains-acp",
                    "discord-bot",
                    "telegram-bot",
                ],
                "upstream_providers": [
                    "openai", "deepseek", "groq", "cerebras", "gemini",
                    "mistral", "mistral_codestral", "nvidia_nim",
                    "open_router", "fireworks", "kimi", "wafer", "zai",
                    "opencode", "opencode_go", "lmstudio", "llamacpp",
                    "ollama",
                ],
            },
        }
    )


@router.get("/.well-known/mcp.json")
async def mcp_manifest(request: Request) -> JSONResponse:
    """Pointer manifest for Model Context Protocol discovery.

    This proxy itself is not an MCP server, but it exposes the structured
    pointers an MCP-aware agent uses to locate OpenAPI + llms.txt docs.
    """
    base = _base_url(request)
    return JSONResponse(
        {
            "name": _SERVICE_NAME,
            "type": "openapi-service",
            "openapi": f"{base}/openapi.json",
            "docs": {
                "llms_txt": f"{base}/llms.txt",
                "llms_full_txt": f"{base}/llms-full.txt",
            },
            "contact": {"repository": _REPO_URL},
        }
    )


@router.get("/robots.txt")
async def robots_txt() -> PlainTextResponse:
    """Welcome AI crawlers; still rate-limited by platform.

    We explicitly list the public, documentation-style surfaces so
    well-behaved AI crawlers index what we want indexed and ignore the
    rest (admin UI is already loopback-only, so it never appears here).
    """
    body = (
        "User-agent: *\n"
        "Allow: /llms.txt\n"
        "Allow: /llms-full.txt\n"
        "Allow: /.well-known/\n"
        "Allow: /openapi.json\n"
        "Allow: /docs\n"
        "Allow: /redoc\n"
        "Disallow: /admin\n"
        "Disallow: /v1/\n"
        "\n"
        "# AI agent crawlers — welcome. See /llms.txt and /llms-full.txt.\n"
        "User-agent: GPTBot\n"
        "Allow: /\n"
        "User-agent: ClaudeBot\n"
        "Allow: /\n"
        "User-agent: anthropic-ai\n"
        "Allow: /\n"
        "User-agent: PerplexityBot\n"
        "Allow: /\n"
        "User-agent: Google-Extended\n"
        "Allow: /\n"
    )
    return PlainTextResponse(body, media_type="text/plain; charset=utf-8")
