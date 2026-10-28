# Ecosystem Integration — Free Claude Code across productized services

This proxy is a single-binary FastAPI service. It fits every deployment shape
your portfolio already uses — the only question is where the gateway lives
relative to each product. Four patterns, picked by operational constraints
rather than preference.

```
                         ┌────────────────────────────┐
                         │   Free Claude Code (FCC)   │
                         │  Anthropic Messages API    │
                         │  + 18 provider backends    │
                         └──────────────┬─────────────┘
          ┌─────────────────────────────┼────────────────────────────┐
          ▼                             ▼                            ▼
    (A) Centralized              (B) Per-service sidecar     (C) Edge / Vercel
     one gateway, many             one container per app       serverless, per deploy
     products call in                                           preview
```

## Pattern A — Centralized gateway

**When:** >1 product under the same domain umbrella, operator wants a single
provider key per upstream and shared metrics / trajectories.

Deploy FCC once (Docker, Fly, Railway, a long-lived Vercel container, or a
VM). Point each product's Claude-compatible client at the same base URL.

```
      product-a.yourdomain.io ┐
      product-b.yourdomain.io ├──► https://fcc.yourdomain.io/v1/messages
      claude code CLI (dev)   ┘
```

**Setup (one time):**

```bash
docker run -d --name fcc \
    -p 8082:8082 \
    -e MODEL=openai/gpt-4o \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
    -e ANTHROPIC_AUTH_TOKEN=$INTERNAL_SHARED_SECRET \
    -e TRAJECTORY_LOG_ENABLED=1 \
    -e SKILLOPT_ENABLED=1 \
    -e FCC_CACHE_DIR=/var/lib/fcc \
    -v fcc-data:/var/lib/fcc \
    ghcr.io/dnzengou/free-claude-code:latest
```

**Per-product env (anywhere a Claude SDK is initialized):**

```bash
ANTHROPIC_BASE_URL=https://fcc.yourdomain.io
ANTHROPIC_AUTH_TOKEN=$INTERNAL_SHARED_SECRET
```

**Operations:** one place to rotate provider keys, one trajectory corpus
feeding one EvoForge → one SkillOpt policy that improves routing for every
product. Admin UI stays loopback-only — SSH-tunnel in when editing config.

**Trade-off:** a shared gateway is a shared blast radius. Mitigate with:

- separate `ANTHROPIC_AUTH_TOKEN` **per product** (give each product its own
  internal secret and use a reverse-proxy header rule), or
- run **two** FCC instances (e.g. `fcc-prod`, `fcc-staging`) and route traffic
  accordingly.

## Pattern B — Per-service sidecar

**When:** each product has its own deployment pipeline / cluster, and you want
provider key secrets to stay scoped to that product's namespace.

Ship FCC as a sidecar container next to the product. Clients talk to
`http://localhost:8082` — no network hop beyond the pod.

```yaml
# docker-compose.yml
services:
  product-a:
    build: .
    environment:
      ANTHROPIC_BASE_URL: http://fcc:8082
      ANTHROPIC_AUTH_TOKEN: ${FCC_TOKEN}
    depends_on: [fcc]

  fcc:
    image: ghcr.io/dnzengou/free-claude-code:latest
    environment:
      MODEL: openai/gpt-4o
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_AUTH_TOKEN: ${FCC_TOKEN}
      TRAJECTORY_LOG_ENABLED: "1"
      FCC_CACHE_DIR: /var/lib/fcc
    volumes:
      - fcc-data:/var/lib/fcc

volumes: { fcc-data: {} }
```

**Operations:** every product has its own trajectory corpus. Run EvoForge per
product (`uv run python scripts/evoforge.py`) and SkillOpt tunes each one
independently. Great when products have different workload mixes
(dashboard-heavy vs chat-heavy vs IDE-plugin).

**Trade-off:** more FCC instances to patch. Compose that with your existing
rolling-deploy pipeline (`docker compose pull && docker compose up -d`).

## Pattern C — Edge (Vercel) per-deployment

**When:** the product is already on Vercel / Netlify / Cloud Run and you want
preview URLs to carry a per-deployment FCC without extra infra.

The repo ships a `Dockerfile` + `vercel.json` already — forking it, setting
env vars, and `vercel --prod` is the whole flow. The live reference deploy is
<https://free-claude-code-main-ebon.vercel.app>.

Each product gets its own forked FCC behind
`https://fcc-${product}.vercel.app`. Preview URLs on each PR carry a matching
preview FCC.

```json
// product-a/vercel.json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://fcc-product-a.vercel.app",
    "ANTHROPIC_AUTH_TOKEN": "@fcc-token"
  }
}
```

**Operations:** zero-cost scale-to-zero. The admin UI is loopback-only, so
runtime config lives in Vercel's env dashboard; trajectory logging
auto-disables on the read-only Lambda FS without raising.

**Trade-off:** no persistent trajectory corpus (serverless FS is read-only).
Keep EvoForge off this host — run it offline against trajectories you ship
elsewhere (e.g. periodically `POST` your rows to a durable log).

## Pattern D — Local-first (developer workstation)

**When:** devs want cheap/local models during build-time or offline work.

```bash
# One terminal
fcc-server &

# Another — point Claude Code at it
fcc-claude
# → launches `claude` with ANTHROPIC_BASE_URL=http://localhost:8082
```

The admin UI at `http://localhost:8082/admin` is now the single surface for
switching providers, validating keys, swapping between NVIDIA NIM / DeepSeek /
LM Studio, inspecting metrics, and reading EvoMetaClaw rollups.

**Trade-off:** per-workstation state. Nothing to share across the team unless
you add Pattern A on top of it.

---

## Agent-detectable discovery (every pattern inherits it)

Every FCC instance exposes these public, non-auth endpoints so AI agents and
crawlers can discover what it is without a human intermediary:

| Endpoint | Purpose |
|---|---|
| `GET /llms.txt` | Compact LLM-friendly service summary (llmstxt.org convention) |
| `GET /llms-full.txt` | Expanded reference including EvoMetaClaw/SkillOpt |
| `GET /.well-known/ai-plugin.json` | ChatGPT-style plugin manifest |
| `GET /.well-known/agents.json` | Generic agent manifest (capabilities + ecosystem) |
| `GET /.well-known/mcp.json` | Pointer to OpenAPI + docs for MCP-aware agents |
| `GET /robots.txt` | Allowlists `GPTBot`, `ClaudeBot`, `anthropic-ai`, `PerplexityBot`, `Google-Extended` |
| `GET /openapi.json`, `/docs`, `/redoc` | Standard FastAPI OpenAPI surface |

Agents hitting any product in your portfolio over FCC can walk this chain
without being pre-briefed. The admin UI (`/admin`) is deliberately excluded
and remains loopback-only.

## Cross-service EvoMetaClaw flywheel

The strategic value compounds across the ecosystem, not within one product:

1. **Capture** — each product's FCC appends to its trajectory log.
2. **Combine** — a nightly job ships all per-product `trajectories.jsonl` to
   a shared durable store (S3, GCS, a Postgres append-only table).
3. **EvoForge** — one scheduled run against the merged corpus produces a
   single `skillopt_policy.json`.
4. **Distribute** — push that file to every FCC instance (write to
   `${FCC_CACHE_DIR}/skillopt_policy.json`; the mtime cache picks it up on
   the next request, no restart needed).

Pattern A collapses steps 2 and 4 into zero work. Patterns B/C need the ship
step but gain per-product isolation. Pattern D is where you *develop* the
policy before shipping.

## Rollout checklist

- [ ] Decide the pattern (A/B/C/D) per product.
- [ ] Mint one `ANTHROPIC_AUTH_TOKEN` per product (never share across prod + dev).
- [ ] Set `MODEL` + at least one provider key.
- [ ] Point client env vars (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`,
      `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`).
- [ ] Enable `TRAJECTORY_LOG_ENABLED=1` once you have writable storage.
- [ ] After ~1 week of real traffic, run `uv run python scripts/evoforge.py`.
- [ ] Set `SKILLOPT_ENABLED=1` and verify `/admin/api/skillopt` reports
      `{enabled: true, loaded: true}`.
- [ ] Confirm `/llms.txt` and `/.well-known/agents.json` return 200 and
      agent crawlers see them in logs.
