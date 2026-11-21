# CLAUDE.md — free-claude-code (Rudra's fork)

## What this is

Local clone of **Rudra Verma's fork** of [steelrumor/free-claude-code](https://github.com/steelrumor/free-claude-code) — an open-source reverse proxy routing Claude Code to 61 AI providers.

- **Fork URL**: `https://github.com/rudraverma/free-claude-code`
- **Active branch**: `feature/admin-remote-access`
- **Remote aliases**: `origin` = upstream HTTPS, `fork` = Rudra's fork (HTTPS push)
- **Deployed**: Docker container `fcc` on `cyberhawk-svr01` (10.100.20.6:8120), token `freecc`

## Active PR

**PR #2309** (`feature/admin-remote-access` → `steelrumor/free-claude-code:main`)

Adds `FCC_ADMIN_ALLOW_REMOTE` setting so the Admin UI is accessible from LAN/remote without changing the default (loopback-only). Security hardened:

| Commit | What |
|---|---|
| `6f5283a` | upstream merge base |
| `4fe9b4e` | feat: add FCC_ADMIN_ALLOW_REMOTE setting |
| `6993321` | fix: CSRF (P1) + misleading proxy auth warning (P2) |
| `4588fe8` | fix: IPv6 origin comparison bug (P3) |
| `69c821b` | fix: DNS-rebinding protection — require bare IP in Host header |
| `(next)` | test: unit tests for _host_is_ip_address + DNS-rebinding scenarios |

### Key file

`src/free_claude_code/api/admin_security.py` — all Admin security logic lives here.

## Git identity (mandatory)

```bash
git config user.email "admin@cyberhawkconsultancy.org"
git config user.name "Rudra Verma"
```

## Pushing changes

Always push to the fork remote, not upstream:

```bash
git push fork feature/admin-remote-access
```

SSH key for push: `/home/cyberhawk/.ssh/id_ed25519` on the server (deploy key scoped to this repo).
For local Windows pushes: uses HTTPS to `https://github.com/rudraverma/free-claude-code.git`.

## Running locally (dev)

```bash
cd src
pip install -e ".[dev]"
python -m free_claude_code --port 8082
```

Admin UI: `http://localhost:8082/admin`

## Deploying to server

After any code change:

```bash
# on cyberhawk-svr01 as cyberhawk user
cd /home/vishal/apps
docker compose build fcc && docker compose up -d fcc
```

## Secrets

All secrets in KeyVault vault `cyberhawk-svr01` (`http://10.100.20.3:8100`).
