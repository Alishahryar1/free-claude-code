# Free Claude Code — Project Blueprint
**Version:** 2.3.0 · **Updated:** 2026-10-28 · **Python:** 3.14.0
**Live:** https://free-claude-code-main-ebon.vercel.app · **Repo:** https://github.com/dnzengou/free-claude-code

---

## Executive Summary

Drop-in Anthropic Messages API proxy that routes Claude Code CLI traffic to 18 provider backends. Keeps Claude Code's client-side protocol stable while swapping the underlying model. Ships with a local Admin UI, Discord/Telegram bot wrapper, and optional voice-note transcription.

---

## Architecture

```
Claude Code CLI / VS Code / JetBrains ACP
        │  Anthropic Messages API
        ▼
┌─────────────────────────────────────────┐
│           FastAPI Proxy (server.py)     │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │  /admin  │  │  /v1/messages        │ │
│  │  Admin   │  │  /v1/models          │ │
│  │  UI      │  │  /v1/messages/count  │ │
│  └──────────┘  └──────────┬───────────┘ │
│                           │             │
│  ┌────────────────────────▼───────────┐ │
│  │         Model Router               │ │
│  │  Opus → MODEL_OPUS                 │ │
│  │  Sonnet → MODEL_SONNET             │ │
│  │  Haiku → MODEL_HAIKU               │ │
│  │  * → MODEL (fallback)              │ │
│  └────────────────────────┬───────────┘ │
└───────────────────────────┼─────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 AnthropicMessages    OpenAICompat         Local
 Transport            Transport            Providers
 (DeepSeek, Kimi,    (NIM, OpenRouter,    (LM Studio,
  Wafer, Fireworks,   Gemini, Groq,        llama.cpp,
  Z.ai)              Cerebras, Mistral,    Ollama)
                      OpenCode, Codestral)
```

---

## Module Map

| Package | Responsibility |
|---|---|
| `server.py` | ASGI entry point — `create_asgi_app()` |
| `api/` | FastAPI routes, admin UI, services, model routing, request optimizations |
| `core/anthropic/` | Shared Anthropic protocol: SSE, content, conversion, thinking, tokens, tools |
| `core/` | Rate limiting, trace/logging utilities |
| `providers/` | Provider transports (18 backends), registry, base classes, error mapping |
| `config/` | Settings (`pydantic-settings`), provider catalog, logging config, paths |
| `messaging/` | Discord/Telegram platforms, session trees, voice transcription |
| `cli/` | `fcc-server`, `fcc-claude`, `fcc-init` entry points; process registry |
| `tests/` | 1443 unit, contract, and integration tests |

---

## Provider Registry

| ID | Transport | Auth |
|---|---|---|
| `nvidia_nim` | OpenAI compat | `NVIDIA_NIM_API_KEY` |
| `open_router` | OpenAI compat | `OPENROUTER_API_KEY` |
| `gemini` | OpenAI compat | `GEMINI_API_KEY` |
| `mistral` | OpenAI compat | `MISTRAL_API_KEY` |
| `mistral_codestral` | OpenAI compat | `CODESTRAL_API_KEY` |
| `opencode` | OpenAI compat | `OPENCODE_API_KEY` |
| `opencode_go` | OpenAI compat | `OPENCODE_API_KEY` |
| `groq` | OpenAI compat | `GROQ_API_KEY` |
| `cerebras` | OpenAI compat | `CEREBRAS_API_KEY` |
| `openai` | OpenAI compat | `OPENAI_API_KEY` |
| `deepseek` | Anthropic native | `DEEPSEEK_API_KEY` |
| `kimi` | Anthropic native | `KIMI_API_KEY` |
| `wafer` | Anthropic native | `WAFER_API_KEY` |
| `fireworks` | Anthropic native | `FIREWORKS_API_KEY` |
| `zai` | Anthropic native | `ZAI_API_KEY` |
| `lmstudio` | OpenAI compat (local) | `LM_STUDIO_BASE_URL` |
| `llamacpp` | Anthropic native (local) | `LLAMACPP_BASE_URL` |
| `ollama` | OpenAI compat (local) | `OLLAMA_BASE_URL` |

---

## Admin UI

Local-only (loopback guard), served at `/admin`.

| File | Purpose |
|---|---|
| `api/admin_static/index.html` | Shell: sidebar, views, toast region, action bar |
| `api/admin_static/fcc.css` | Design system, animations, toast, skeleton, responsive, metrics |
| `api/admin_static/admin.js` | Toast, auto-refresh, keyboard shortcuts, copy, dirty guard |

**Features shipped:**
- Toast notification system (ok / warn / error / info, auto-dismiss 4.2 s)
- Provider status auto-refresh every 30 s with pulsing indicator
- `Ctrl+S` → validate · `Ctrl+Enter` → apply
- Copy-to-clipboard on non-secret text fields
- Skeleton loading cards on boot
- `beforeunload` unsaved-changes browser guard
- 6 CSS keyframe animations; accent glow on focused inputs
- Full mobile-responsive layout
- Dark/light theme toggle — `localStorage` persistence, sun/moon icons, full CSS variable override
- Metrics tab — summary cards, latency sparkline, per-request table with inline bars; `GET /admin/api/metrics`
- One-click API key validation — "Validate key" button per non-local provider card; `POST /admin/api/providers/{id}/validate`; 3-state result: key valid / auth OK (completion failed) / key invalid
- **EvoMetaClaw panel** — cost + skill mix + trajectory count on the Metrics tab; opt-in via `TRAJECTORY_LOG_ENABLED=1`
- **Est. cost card** — per-provider USD estimate from `pricing.py` catalog

---

## The Moat: EvoMetaClaw × EvoForge × SkillOpt

OpenClaw and every other proxy can copy the provider registry, the admin UI, and the deploy pipeline in a weekend. They cannot copy **months of real usage trajectories**, the **skill-conditioned routing policies** derived from them, or the **evaluation curriculum** that keeps those policies honest. That is this project's three-stage flywheel:

```
       online (this repo)                            offline (planned)
┌──────────────────────────────┐                ┌─────────────────────────┐
│  EvoMetaClaw  ── capture ─── │ trajectories/  │  EvoForge  ── process ──│
│  gateway + trajectory logger │────────────▶───│  cluster · dedupe ·     │
│  (api/trajectory.py)         │  jsonl corpus  │  score · eval bench     │
└──────────────────────────────┘                └────────┬────────────────┘
              ▲                                          │
              │      per-skill routing policy JSON       │
              └──────────────── SkillOpt  ◀──────────────┘
                                (per-skill optimizer)
```

The gateway keeps working the same day one — SkillOpt policies are additive. When enabled they simply narrow provider selection *per skill* based on the trajectory evidence. When absent, routing falls back to `MODEL_*` env vars exactly like today.

### Stage 1 — EvoMetaClaw (shipped, v2.1.0)

Capture surface. Every completed proxy request emits one JSONL row with the metadata SkillOpt needs to train against.

| Component | File | Role |
|---|---|---|
| Trajectory logger | `api/trajectory.py` | Opt-in JSONL append with byte-cap rotation. Records `{request_id, provider, model, skill, thinking, tokens, latency, cost, status}` per request. Serverless-safe: read-only FS → auto-disable. |
| Skill inference | `api/trajectory.py::infer_skill` | Cheap heuristic classifier over messages + tools: `probe`, `edit`, `plan`, `question`, `chat`. Deterministic, adequate for SkillOpt bucketing; upgradeable to a learned classifier once the corpus is large enough. |
| Pricing catalog | `api/pricing.py` | Vendor rate cards (USD / 1M tokens). Prefixed lookup wins over bare model id. Zero-cost default for unknown models — never overstates spend. |
| Admin surface | `GET /admin/api/trajectory` | Loopback-only rollup: total, per-skill mix, per-provider mix, tokens, cost, log path. |
| UI callout | `admin_static/admin.js::renderEvoMetaClaw` | "Recording / Disabled" pill + skill chips on the Metrics tab. |

Enable in production:
```
TRAJECTORY_LOG_ENABLED=1
TRAJECTORY_LOG_MAX_BYTES=5000000    # optional, default 5 MB
FCC_CACHE_DIR=/var/lib/fcc          # optional, default ~/.fcc-cache
```

### Stage 2 — EvoForge (shipped, v2.2.0)

Offline processing pipeline that turns raw JSONL trajectories into a versioned SkillOpt policy. Deliberately separated from the online gateway so it can run on a schedule without touching request latency.

**Input:** `${FCC_CACHE_DIR}/trajectories.jsonl(.1)` — the newline-delimited stream EvoMetaClaw produced.
**Output:** `${FCC_CACHE_DIR}/skillopt_policy.json` — the artefact SkillOpt reads at runtime.
**Module:** `api/evoforge.py`. **CLI:** `uv run python scripts/evoforge.py [--dry-run] [--min-samples N] [--top-k K]`.

Stages:
1. **Ingest** (`load_rows`) — read + validate rows across `.jsonl` and `.jsonl.1`, drop malformed / partial writes.
2. **Aggregate** (`aggregate`) — bucket by `(skill, provider, model)`; require ≥ `min_samples` observations (default 5).
3. **Score** — utility per candidate: `-λ_cost · avg_cost - λ_latency · p95_latency + λ_success · ok_rate`. Defaults chosen so a $0.001 request at 500 ms with 95 % success scores near zero.
4. **Build policy** (`build_policy`) — argmax per skill → `primary`, next `top_k - 1` → `fallbacks`. Deterministic tie-break: highest score, then provider id, then model.
5. **Publish** (`write_policy`) — atomic write via `.tmp` + `rename`.

Ownership boundary: the gateway never writes into the forge; the forge never reads live traffic. Only shared surfaces are the JSONL corpus (gateway → forge, write-only) and the policy JSON (forge → gateway, read-only).

### Stage 3 — SkillOpt (shipped, v2.2.0)

Runtime consumer of the EvoForge policy. Reads `skillopt_policy.json` on demand with an mtime-checked cache, and, when enabled, narrows `ModelRouter` per inferred skill.

**Module:** `api/skillopt.py`. **Endpoint:** `GET /admin/api/skillopt` (loopback-only).

**Runtime contract:**
- Direct-slug requests (`provider/model` in the model name) still go where the caller asked — SkillOpt never touches them.
- Otherwise, `services.create_message` infers the skill from the raw request and passes it into `ModelRouter.resolve(..., skill=...)`. The router calls `skillopt.lookup(skill)`; when a policy exists **and** its `primary` points at a supported provider, that ref wins over `MODEL_*` tier overrides.
- If no policy applies (skill missing, file absent, disabled, unsupported provider), routing falls through to today's `MODEL_*` behavior unchanged.

**Kill switch:** `SKILLOPT_ENABLED` — default off. Fresh installs behave exactly like v2.1.0 until an operator flips the flag.

**Policy shape (produced by EvoForge, illustrative):**
```json
{
  "version": 1,
  "generated_ts": 1720000000,
  "params": { "min_samples": 5, "top_k": 3, "lambda_cost": 1000.0, "lambda_latency_ms": 0.001, "lambda_success": 1.0 },
  "policies": {
    "edit":     { "primary": "deepseek/deepseek-chat", "fallbacks": ["openai/gpt-4o"] },
    "plan":     { "primary": "openai/gpt-4o",          "fallbacks": ["gemini/gemini-2.5-flash"] },
    "probe":    { "primary": "groq/llama-3.3-70b-versatile", "fallbacks": [] },
    "question": { "primary": "openai/gpt-4o-mini",     "fallbacks": [] },
    "chat":     { "primary": "openai/gpt-4o-mini",     "fallbacks": [] }
  }
}
```

**Fallback traversal:** when SkillOpt is enabled and the primary provider fails `preflight_stream` with `AuthenticationError` / `ServiceUnavailableError` / `RateLimitError`, the service layer walks `policies[skill].fallbacks[]` in order. Each candidate rebinds the request via `ModelRouter.rebind_messages_request`; the first successful preflight commits. If every candidate fails, the primary's original error is re-raised so callers see the same exception they would have seen without SkillOpt enabled.

### Why this stack, not "just fine-tune"

Full-parameter fine-tuning is expensive, tied to one base model, and freezes the policy at training time. This stack:

1. **Doesn't fine-tune the model** — it optimises *routing*. Cheap, provider-agnostic, hot-swappable.
2. **Improves monotonically** — every request adds one trajectory; the policy only gets richer.
3. **Leaves the model choice open** — when a better base model ships next week, SkillOpt re-scores against it in the next EvoForge run instead of paying to re-train.
4. **Is legible** — the policy JSON is human-readable; an operator can override a bucket by hand without touching gateway code.

Fine-tuning is not ruled out — it's a *future consumer* of the same trajectory corpus. EvoMetaClaw's records include the fields (`skill`, `thinking`, cost, tokens) needed for both.

---

## Agent-detectable discovery ("agent SEO")

Every FCC instance (local, Vercel, sidecar, central) exposes a public, non-auth discovery surface so AI agents and crawlers can learn what the service does without a human intermediary. The admin UI (`/admin`) stays loopback-only; nothing added here leaks it.

| Endpoint | Purpose |
|---|---|
| `GET /llms.txt` | Compact service summary (follows [llmstxt.org](https://llmstxt.org/)) |
| `GET /llms-full.txt` | Expanded reference including EvoMetaClaw / EvoForge / SkillOpt |
| `GET /.well-known/ai-plugin.json` | ChatGPT-style plugin manifest with `description_for_model` + OpenAPI URL |
| `GET /.well-known/agents.json` | Generic agent manifest listing capabilities + ecosystem (clients, upstream providers) |
| `GET /.well-known/mcp.json` | Pointer manifest for MCP-aware agents (OpenAPI + docs URLs) |
| `GET /robots.txt` | Explicit `Allow: /` for `GPTBot`, `ClaudeBot`, `anthropic-ai`, `PerplexityBot`, `Google-Extended`; `Disallow: /admin` |
| `GET /openapi.json`, `/docs`, `/redoc` | FastAPI's own OpenAPI surface — titled "Free Claude Code — Anthropic-compatible model gateway" |

**Why it matters:** OpenClaw-style forks will copy the proxy's wire. They won't automatically expose their fork at well-known paths — a model inside Claude, ChatGPT, Perplexity or Gemini looking for an Anthropic-compatible model gateway can discover *this* deploy directly. Every ecosystem instance (see `ECOSYSTEM.md`) inherits this surface by shipping the same binary.

---

## Deployment

| Target | Method | URL |
|---|---|---|
| **Vercel (production)** | `vercel --prod` / push to `main` | https://free-claude-code-main-ebon.vercel.app |
| **Local** | `uv run uvicorn server:app --host 0.0.0.0 --port 8082` | http://localhost:8082 |
| **Docker** | `docker build -t fcc . && docker run -p 8082:8082 fcc` | http://localhost:8082 |

**Required env vars on Vercel dashboard:**
```
MODEL=<provider>/<model>          # e.g. nvidia_nim/nvidia/nemotron-3-super-120b-a12b
ANTHROPIC_AUTH_TOKEN=<token>      # proxy auth key (default: freecc)
NVIDIA_NIM_API_KEY=<key>          # or whichever provider key(s) you use
```
Admin UI (`/admin`) is loopback-only — not accessible on Vercel by design.

---

## CI / Quality Gates

All enforced in `.github/workflows/tests.yml` on push/PR to `main`/`master`:

| Job | Command | Status |
|---|---|---|
| Ban type ignore suppressions | `grep -rE '# type: ignore\|# ty: ignore'` | ✅ |
| ruff-format | `uv run ruff format --check` | ✅ |
| ruff-check | `uv run ruff check` | ✅ |
| ty | `uv run ty check` | ✅ |
| pytest | `uv run pytest -v --tb=short` | ✅ 1443 baseline + 27 new (skillopt/evoforge/pricing/trajectory/admin) |

---

## Roadmap

### Completed ✅
- [x] FastAPI proxy with Anthropic-compatible routes
- [x] 17 provider backends (OpenAI compat + Anthropic native + local)
- [x] Per-model-tier routing (Opus / Sonnet / Haiku / fallback)
- [x] Streaming, tool use, thinking block support
- [x] Request optimization handlers (probe mocking, title skip, etc.)
- [x] Admin UI — config editor, provider status, validate/apply
- [x] Discord + Telegram bot wrapper with session trees
- [x] Voice note transcription (local Whisper + NVIDIA NIM Riva)
- [x] `/v1/models` Gateway model discovery
- [x] Rate limiting + concurrency control per provider
- [x] **Production Admin UI overhaul** — Toast, auto-refresh, kbd shortcuts, copy, skeleton
- [x] **Real-time settings search/filter** — topbar input, key+label match, auto-hides empty sections
- [x] **Vercel deployment** — `Dockerfile` (python:3.14-slim, non-root, uv), `vercel.json`, stderr log fallback for read-only Lambda FS
- [x] **OpenAI provider** — `providers/openai/`, `openai/gpt-4o` slug, `OPENAI_API_KEY`, wired across all 9 touch-points
- [x] **`GET /health/ready`** — authenticated readiness check: provider, model, tier overrides, auth status; useful for Vercel env var verification
- [x] **Dark/light theme toggle** — `localStorage` persistence, sun/moon SVG icons, `data-theme` on `<html>`, full light-mode design token overrides in CSS
- [x] **Per-request metrics panel** — `api/metrics.py` in-memory store, `GET /admin/api/metrics`, Admin UI "Metrics" tab: summary cards (total/avg/p95/tokens), latency sparkline, request table with inline bars
- [x] **Provider health history** — `api/health_history.py` per-provider bounded deque (50 entries), `GET /admin/api/health-history`, JSON persistence to `~/.fcc-cache/health_history.json` (serverless-safe fallback), inline sparkline on each provider card
- [x] **One-click API key validation** — `POST /admin/api/providers/{id}/validate`; `validate_credentials()` on BaseProvider; OpenAI-compat override adds 1-token completion check; "Validate key" button per non-local provider card
- [x] **EvoMetaClaw trajectory logger** — `api/trajectory.py`, opt-in JSONL append with byte-cap rotation, skill inference (`probe`/`edit`/`plan`/`question`/`chat`), in-memory rollup, `GET /admin/api/trajectory`, admin-UI panel with cost + skill mix
- [x] **Pricing catalog** — `api/pricing.py`, USD/1M-token rate cards for 15 slugs; drives Est. cost card on Metrics tab and enriches `/admin/api/metrics` summary
- [x] **EvoForge offline pipeline** — `api/evoforge.py` + `scripts/evoforge.py` CLI; ingest → aggregate → score (`-λ_cost·cost - λ_lat·p95 + λ_success·ok`) → build policy → atomic write; deterministic tie-break; kills nothing at runtime
- [x] **SkillOpt runtime routing** — `api/skillopt.py`, mtime-cached policy loader; `ModelRouter.resolve(..., skill=...)` consults it after direct-slug routing and before `MODEL_*` tier overrides; `SKILLOPT_ENABLED=0` kill switch, default off
- [x] **SkillOpt fallback traversal** — `_preflight_with_fallbacks` walks `policies[skill].fallbacks[]` on retriable preflight errors; closes the single-provider SPOF; falls back to primary's error contract when every candidate fails
- [x] **Agent-detectable discovery** — `/llms.txt`, `/llms-full.txt`, `/.well-known/{ai-plugin,agents,mcp}.json`, `/robots.txt`; OpenAPI info block renamed to describe the service
- [x] **Ecosystem integration guide (`ECOSYSTEM.md`)** — four deployment patterns + cross-service EvoMetaClaw flywheel

### Planned 🔲
- [ ] **In-stream fallback** — current fallback traversal catches only `preflight_stream` failures; mid-stream disconnect still bubbles up. Needs a resumable SSE wrapper that buffers the first N bytes before committing to a candidate.
- [ ] **EvoForge held-out evaluation** — replay a held-out corpus slice against a candidate policy before publishing, so the new policy has to *beat* the incumbent on the same requests before it can win.
- [ ] **Periodic auto-forge** — background task (opt-in) that triggers EvoForge every N hours when `TRAJECTORY_LOG_ENABLED=1`.
- [ ] Learned skill classifier — replace heuristic `infer_skill` once the corpus exceeds ~10k rows across 3+ skills.

---

## Changelog

### v2.3.0 — 2026-10-28 (agent SEO + fallback traversal + ecosystem)
- **Agent-detectable discovery (`api/discovery_routes.py`)**: public, no-auth endpoints — `GET /llms.txt`, `/llms-full.txt`, `/.well-known/ai-plugin.json`, `/.well-known/agents.json`, `/.well-known/mcp.json`, `/robots.txt`. Base URL is forwarded-host aware. Admin UI stays loopback-only. OpenAPI info block renamed to "Free Claude Code — Anthropic-compatible model gateway" (v2.3.0) so the OpenAPI surface agents already crawl at `/openapi.json` describes the service correctly. 7 new tests cover the shape + public-vs-auth separation.
- **SkillOpt fallback traversal**: `_preflight_with_fallbacks` in `api/services.py` walks `policies[skill].fallbacks[]` on `AuthenticationError` / `ServiceUnavailableError` / `RateLimitError` during `preflight_stream`. Each candidate rebinds the request through the new `ModelRouter.rebind_messages_request`. All-candidates-failed re-raises the primary's original error, preserving the pre-SkillOpt failure contract. Unsupported-provider fallback refs are skipped and logged. 6 new tests cover: primary success (no fallback), auth-error → fallback, 503 → fallback, all-fail re-raises primary, unsupported-fallback is skipped, SkillOpt-off disables the loop entirely.
- **`skillopt.fallbacks(skill)`**: new tuple accessor so the service layer doesn't poke `.policies` internals.
- **`ECOSYSTEM.md`**: four deployment patterns (centralized, per-service sidecar, edge/Vercel, local-first) with concrete docker-compose + Vercel snippets; cross-service EvoMetaClaw flywheel walkthrough; rollout checklist.
- **Blueprint**: v2.3.0; agent-SEO section added; `SkillOpt fallback traversal` moved from Planned → Completed; roadmap updated.

### v2.2.0 — 2026-07-11 (EvoForge + SkillOpt)
- **EvoForge shipped** (`api/evoforge.py` + `scripts/evoforge.py`): reads `${FCC_CACHE_DIR}/trajectories.jsonl(.1)`, aggregates per `(skill, provider, model)`, scores `-λ_cost·avg_cost - λ_lat·p95 + λ_success·ok_rate`, publishes `skillopt_policy.json` via atomic `.tmp` + `rename`. Deterministic tie-break (score → provider → model) so re-runs on identical input produce identical policies. CLI supports `--dry-run`, `--min-samples`, `--top-k`.
- **SkillOpt runtime shipped** (`api/skillopt.py`): mtime-cached policy loader; `ModelRouter.resolve(..., skill=...)` consults it after direct-slug routing and before `MODEL_*` tier overrides. Kill switch `SKILLOPT_ENABLED=0` (default off) — fresh installs behave identically to v2.1.0 until an operator flips the flag. Unsupported-provider policies are logged and ignored (never raised into the request path).
- **Skill inference moved pre-routing**: `trajectory.infer_skill` gained an optional `input_tokens` parameter with a char/4 fallback so the router can consult SkillOpt before the (expensive) tiktoken count runs; the exact-token metering keeps its precise value downstream. Block-text extraction hardened to handle both dict and Pydantic content blocks.
- **`GET /admin/api/skillopt`** (loopback-only): loaded policy snapshot for the admin UI (`{enabled, loaded, path, version, policies}`).
- **Tests**: 16 new — `test_skillopt.py` (8: disabled default, enabled lookup, missing skill, missing file, malformed JSON, mtime refresh, snapshot loaded, snapshot disabled), `test_evoforge.py` (8: load valid rows, skip bad lines, under-sampled dropped, cheaper+faster wins, versioned policy shape, atomic write, end-to-end `run()`, error rate lowers ok_rate), plus 3 model_router hooks (skill hint ignored when disabled, override when enabled, fall-through for unsupported provider) + 2 admin skillopt endpoint tests.

### v2.1.0 — 2026-07-11 (blueprint: three-pillar moat)
- **Blueprint restructured**: "EvoMetaClaw (the moat)" replaced with the three-pillar architecture — **EvoMetaClaw** (capture, shipped), **EvoForge** (offline processing, planned), **SkillOpt** (runtime optimizer, planned). Adds ASCII flywheel diagram, per-stage responsibilities table, sample `skillopt_policy.json`, runtime contract with kill-switch, and a "why this stack, not just fine-tune" rationale.
- **Roadmap sharpened**: EvoForge and SkillOpt now called out as the next two milestones; provider-fallback SPOF removal folded into SkillOpt's `fallbacks[]` (single implementation covers both).

### v2.1.0 — 2026-07-11 (EvoMetaClaw + cost)
- **EvoMetaClaw trajectory logger** (`api/trajectory.py`): opt-in JSONL append with byte-cap rotation, in-memory summary, skill inference. `TRAJECTORY_LOG_ENABLED=1` to enable; serverless-safe (read-only FS → auto-disable, same pattern as `configure_logging` / `health_history`).
- **Pricing catalog** (`api/pricing.py`): USD/1M-token rate cards for OpenAI, DeepSeek, Groq, Cerebras, Gemini, Mistral, Kimi, Fireworks. Prefixed lookup wins over bare model id. Local providers always cost 0. Zero-cost default for unknown models.
- **`GET /admin/api/trajectory`** (loopback-only): returns `{enabled, total, per_skill, per_provider, tokens_in, tokens_out, cost_usd, log_path}` for the EvoMetaClaw UI panel.
- **Metrics enrichment**: `/admin/api/metrics` summary now includes `total_cost_usd` and `per_provider_cost_usd`; Metrics tab gets a 5th summary card and an EvoMetaClaw panel with recording/disabled pill + skill chips.
- **Service integration**: `_metered_stream` now emits one trajectory row per completed request (or error), pre-tagged with the inferred skill and priced against the pricing catalog. `create_message` computes the skill tag before wrapping the stream so both the metric and the trajectory carry the same label.
- **Tests**: 11 new tests — pricing lookups & summarize (6), trajectory disable/enable/rotation/skill inference (7 sub-cases), admin `/trajectory` endpoint + loopback guard (2), metrics cost fields (1).

### v2.0.0 — 2026-06-15 (audit + ty)
- **`ty` regression fix**: removed redundant `health_history = importlib.reload(health_history)` reassignments in `tests/api/test_admin.py` (3 sites). `importlib.reload()` mutates the module in place, so the rebinding only confused ty's narrowed-import type with the generic `ModuleType` return — no `# type: ignore` added (CLAUDE.md hard rule).
- **KafCade audit (`kc E`)**: read-only structured pass. Scores — Security 9/10, Correctness 9/10, Performance 10/10, Quality 9/10. No P0/P1 findings. Surfaced provider count drift (17 → 18, openai shipped 2026-05-30) and test count drift (1429 → 1443) in Module Map — corrected.
- **CI**: 1443/1443 passing across all 4 gates (ruff format, ruff check, ty, pytest).

### v2.0.0 — 2026-06-15 (validate-key)
- **One-click provider API key validation**: `POST /admin/api/providers/{id}/validate` (loopback-only). Calls `provider.validate_credentials(preferred_model?)`. `BaseProvider` default: model-list check. `OpenAIChatTransport` override: model list **+** 1-token `max_tokens=1` chat completion — proves the key works for actual inference.
- **Admin UI "Validate key" button**: every non-local provider card gets a dedicated "Validate key" button (distinct from "Refresh models"). Returns `{auth_ok, completion_ok, models_count, test_model?, error_type?}`. UI shows "Key valid ✓" / "Auth OK" (completion failed) / "Key invalid ✗" with toast.
- **CI**: 1443/1443 passing (9 new tests).

### v2.0.0 — 2026-06-15 (health history)
- **Provider health history**: `api/health_history.py` — per-provider `defaultdict(deque(maxlen=50))` with `threading.Lock`. `record(provider_id, status, latency_ms, error_type)` writes one outcome per `test_provider` or `_check_local_provider` call.
- **JSON persistence**: writes to `${FCC_CACHE_DIR:-~/.fcc-cache}/health_history.json` on every record; restored on module import. `_PERSIST_DISABLED` flag prevents repeat retries on read-only Vercel/Lambda FS (same pattern as `configure_logging`).
- **`GET /admin/api/health-history`** (loopback-only): returns `{providers: {<id>: [{ts, status, latency_ms, error_type?}, …]}}`.
- **Admin UI inline sparkline**: every provider card gets a `.provider-spark` strip showing last 30 health-check outcomes; bars color-coded (ok/slow/error); refreshes after every Test/Refresh-models click.
- **CI**: 1435/1435 passing (6 new tests: empty store, loopback enforcement, record+persist roundtrip, per-provider buffer cap)

### v2.0.0 — 2026-06-13 (metrics)
- **Per-request metrics panel**: `api/metrics.py` bounded deque (500 entries), thread-safe `record()`/`snapshot()`. `_metered_stream()` in `services.py` wraps provider SSE, parses `message_delta` for `output_tokens`, records latency on completion/error.
- **`GET /admin/api/metrics`**: loopback-only, returns `{requests[], summary{total, avg_latency_ms, p95_latency_ms, total_input_tokens, total_output_tokens}}`
- **Admin UI Metrics tab**: summary cards, latency sparkline (newest-right, color-coded by threshold), request table with inline latency bars. Loads on nav switch.
- **Vercel path conflict fix**: renamed `admin.css` → `fcc.css`; Vercel strips extensions before conflict checking so `admin.js` + `admin.css` → both `admin` = blocked.
- **CI**: 1431/1431 passing

### v2.0.0 — 2026-06-07 (theme + build)
- **Dark/light theme toggle**: sun/moon SVG button in topbar; `localStorage` key `fcc-theme`; `data-theme="light"` on `<html>`; full CSS variable override block for light mode; persists across page loads
- **Hatchling wheel fix**: removed duplicate `api/admin_static` from `force-include` — newer hatchling strict-mode rejected the double-inclusion; `uv build` now clean; unblocked Vercel re-deployment
- **CI**: 1429/1429 passing; deployed to https://free-claude-code-main-ebon.vercel.app

### v2.0.0 — 2026-05-30 (readiness)
- **`GET /health/ready`** (auth required): returns `{status, provider, model, model_opus, model_sonnet, model_haiku, auth_required}` — verify Vercel env wiring without local admin UI

### v2.0.0 — 2026-05-30 (providers)
- **OpenAI provider** (`openai/`): `providers/openai/OpenAIProvider` extends `OpenAIChatTransport`, base `https://api.openai.com/v1`, thinking via `ReasoningReplayMode.REASONING_CONTENT` for o1/o3/o4 models
- `OPENAI_API_KEY` + `OPENAI_PROXY` wired through settings, admin config, `.env.example`, admin UI, catalog, registry, and test mocks (18th provider)
- Smoke config updated: `openai/gpt-oss-120b:free` → `open_router/openai/gpt-oss-120b:free` to avoid provider ID clash
- **DeepSeek** already supported — `MODEL=deepseek/deepseek-chat` + `DEEPSEEK_API_KEY` (no code change needed)

### v2.0.0 — 2026-05-30 (deploy)
- **Live on Vercel**: https://free-claude-code-main-ebon.vercel.app — Python 3.14, uv 0.10.11, iad1
- **Dockerfile**: multi-stage `python:3.14-slim-bookworm`, non-root `fcc` user, `${PORT:-8082}`, compatible with Vercel/Railway/Render
- **`configure_logging` serverless fix**: `PermissionError/OSError` fallback to stderr when Lambda filesystem is read-only
- **Removed `[tool.uv] required-version`** from `pyproject.toml` — blocked Vercel's uv 0.10.11; `uv.lock` provides reproducibility

### v2.0.0 — 2026-05-30
- **Real-time settings search/filter**: topbar `<input type="search">`, filters all `.field` elements live by env-var key + human label; empty section cards auto-hide; clears on view-switch

### v2.0.0 — 2026-05-29
- **Production Admin UI overhaul**: Toast system, 30 s auto-refresh with pulsing dot, `Ctrl+S` / `Ctrl+Enter` shortcuts, copy-to-clipboard, skeleton loading, unsaved-changes guard, 6 CSS keyframe animations, provider card lift-on-hover, accent glow on focused inputs, full mobile reflow
- Fixed `.gitignore` `server.*` wildcard over-matching `server.py` — added `!server.py` negation
- 1429 / 1429 tests passing

### v2.0.0 (upstream baseline)
- 17 provider backends
- Messaging platforms (Discord, Telegram), voice transcription
- Admin UI for managed `.env` config
- Gateway model discovery via `/v1/models`
- Python 3.14, uv, ruff, ty CI stack
