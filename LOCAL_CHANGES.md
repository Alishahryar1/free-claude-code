# FCC local changes

Changes layered on top of upstream `Alishahryar1/free-claude-code` that are not
part of any release. `fcc-update` reinstalls from upstream and will silently drop
everything here, so re-apply after updating.

## Current deployment

As of 2026-10-01 the live install is built from this clone, not from upstream:

```
free-claude-code 6.5.12.dev1+gc1913aae5
upstream base:    7195cef  (v6.5.11)
direct_url.json -> file:///C:/Users/aams1/OneDrive/Documents/Default%20Project/free-claude-code
```

The version is stamped from **this** commit, not upstream's, which is the proof that
site-packages was built from this tree. It previously resolved to `0.0.0` from a
`main.zip` extracted under `%TEMP%`, which meant an unversioned, uncleanable source
tree. Fixing that is what let this patch be deployed at all.

Do not install from a downloaded `main.zip`. It has no `.git`, so `hatch-vcs` stamps
it `0.0.0` and `direct_url.json` points at wherever it was unpacked — which is what
produced the broken original install. Always install from this clone.

## Where this work lives

The work is on branch **`local-hardening`** in this clone, so `main` stays clean and
`git merge origin/main` brings in upstream cleanly. The patch files
`fcc-crash-recovery.patch` and `fcc-test-timeouts.patch` are committed as a portable
copy, so the changes survive even if the branch is lost.

`origin` is the upstream author's repository (`Alishahryar1/free-claude-code`), which
this account has no push access to, and pushing there is not appropriate. The work
lives in this clone; push it to your own fork if you want a remote copy.

The clone has a repo-local git identity (`FCC Local <fcc-local@localhost>`) so that
committing does not require touching a global `~/.gitconfig`. Change it with
`git config --local user.name` / `user.email`.

## Test status

Verified green on v6.5.11 with both patches applied:

```
tests/cli/test_server_crash_recovery.py
tests/cli/test_crash_recovery_e2e.py
tests/cli/test_config_restart_lifecycle.py
tests/cli/test_desktop.py
tests/cli/test_entrypoints.py
tests/cli/test_server_logging.py
tests/cli/test_progressive_launch.py

75 passed in 228s
```

That is a targeted subset, **not** the full suite. The whole `tests/` tree has never
been run to completion, so no claim is made about it. The e2e test binds a real socket
and needs a free instance ID on the same port, so it is the slowest and most
environment-sensitive of the set.

## Applying after an update

`fcc-update` reinstalls from upstream. That drops both patches, resets the version to
`0.0.0`, and points `direct_url.json` at the temp copy again. To get back to a
patched install:

```powershell
cd "$env:USERPROFILE\OneDrive\Documents\Default Project\free-claude-code"
git fetch origin
git merge origin/main      # local work is on branch local-hardening, so a plain
                           # `git pull` will not do this for you
.\DEPLOY.ps1
```

`DEPLOY.ps1` stops whatever holds a lock on the tool directory, reinstalls, verifies
the recovery code is actually present in site-packages, and reports whether port 8082
is free. It deliberately does **not** start the server — see below. Do not run the
`uv tool install` line by hand unless you have a reason to.

`DEPLOY.ps1` will not run as-is on a default Windows machine, because every execution
policy scope is `Undefined`, which resolves to `Restricted`. Bypass it for that one
process rather than loosening the machine policy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\DEPLOY.ps1
```

That leaves `CurrentUser` and `LocalMachine` untouched and vanishes with the shell.

If `git merge` conflicts, the patch is one hunk region in
`src/free_claude_code/cli/commands.py`. If upstream refactors `ServerSupervisor.run`,
resolve by hand: keep the upstream loop structure and re-add the pieces below.
Alternatively `git apply fcc-crash-recovery.patch fcc-test-timeouts.patch` after
resetting the three touched files to upstream.

## Deploying: two things that will bite you otherwise

1. **A stale process locks the install.** An abandoned `fcc-claude` session keeps
   `%APPDATA%\uv\tools\free-claude-code\Scripts` open and `uv tool install` fails with
   `Access is denied (os error 5)`. Match on the tool directory, not the process name,
   so orphaned `python.exe` children are caught too. Matching on the tool directory
   alone is *not* sufficient either: the `fcc-*` uv shims in `%USERPROFILE%\.local\bin`
   sit outside it, survive the sweep, and can rebind port 8082 afterwards. The script
   matches both families.
2. **The script deliberately does not start the server.** It reports whether port
   8082 is free and stops there. An earlier version launched `fcc-server` hidden,
   which held the port and made the next `fcc-server` typed by hand die with
   `[WinError 10048] Only one usage of each socket address...` while the console
   log that explained it was nowhere on screen. Start it yourself so the output is
   visible: `fcc-server` for foreground logs, or `fcc-desktop` for the tray.

   Note that only bare `fcc-server` hard-fails on a taken port. The graceful
   "a healthy server already owns this" handoff in `ServerSupervisor.run()` is
   reachable only through `cli/desktop.py`, the sole caller that passes
   `existing_server=`.
3. **Host performance dominates everything else.** On the machine this was developed
   on, `import fastapi` took 21.5s against a ~0.5s baseline — a 2014 dual-core CPU
   with Defender real-time scanning every imported file and no exclusions. Running
   `ADD-DEFENDER-EXCLUSIONS.ps1` from an elevated prompt cut the targeted test set
   from 825s to 228s. Exclusions must be applied by an administrator; the agent
   deliberately does not touch Defender configuration. The same slowness is why the
   subprocess timeouts in section 2 were needed at all.

## 1. Crash recovery for the supervisor (the "server never dies" fix)

**Defect:** `ServerSupervisor.run()` wrapped its serve loop in a `try` that only
caught `KeyboardInterrupt` and `OSError`. Every other exception — provider
construction failure, unreadable config, a bug on a startup path — propagated out
of `run()`, through `serve()`, and terminated the process. Because FCC is a
long-lived local proxy, that kills `/v1/messages` for Claude Code, Codex, VS Code
and anything else attached, with no recovery and no obvious signal.

**Fix:** catch the unexpected exception inside the loop, back off exponentially
(0.5s doubling to a 30s ceiling), log loudly, and start a fresh generation. A real
`request_stop()` still wins over a pending restart, and the pre-existing
`EADDRINUSE` / `existing_server` handoff is preserved so the desktop and terminal
launchers still recognise a healthy peer.

Failure counters reset on a clean generation boundary, because that boundary is
proof the previous server released its resources, so a later crash is a fresh
fault rather than a continuation of the same one.

**Verification:** `tests/cli/test_server_crash_recovery.py` reproduces the original
crash-and-die behaviour, and asserts a stop request is not undone by the recovery
loop. Both fail before the patch and pass after it.

## 2. Portable subprocess timeouts in the test suite

**Defect:** `tests/cli/test_server_logging.py` and `tests/cli/test_progressive_launch.py`
spawn a child interpreter that imports the entire server stack under a hardcoded
30s and 12s budget. On a slow or cold host the child is still importing when the
budget expires, so the suite reports a real pass as a `TimeoutExpired` failure.
Observed on the machine this was developed on: 37.2s for a 30s budget.

**Fix:** read the budget from `FCC_TEST_SUBPROCESS_TIMEOUT` (default 300s) with a
comment explaining that these are import-bound rather than logic-bound. No
behaviour changes on a fast machine; slow hosts stop producing false failures.

**Note:** the underlying cause is host I/O, not the tests. On the development
machine `import fastapi` took 21.5s against a ~0.5s baseline, caused by a 2014
dual-core CPU with Defender real-time scanning every imported file and no
exclusions. Excluding the project and interpreter directories from Defender is
worth far more than any timeout change.

## Partly mitigated: provider throttling

The throttled entries in `~/.fcc/logs/server.log` are NVIDIA NIM enforcing its own
free-tier quota, not a fault in FCC. Observed on this machine as:

```
POST /v1/messages?beta=true HTTP/1.1" 529
NIM_ERROR: request_id=req_... exc_type=APIError http_status=None cause_types=None
```

The admission controller retries with backoff and then surfaces the error once its
bounded budget is spent. Raising `PROVIDER_RATE_LIMIT` or `PROVIDER_MAX_CONCURRENCY`
would provoke more throttling, not less — but lengthening the retry budget itself does
not, because coordinated recovery keeps retries serial. See "Provider retry budget"
below.

The other, code-free mitigation is a `MODEL_FALLBACKS` chain in `~/.fcc/.env`, already
implemented end to end: `config/model_refs.py` parses it and `application/execution.py`
applies `candidates = (primary, *resolved.fallbacks)`, so a throttled primary falls
through to the next provider instead of ending the turn. The default `MODEL` is a NIM
model (`nvidia_nim/nvidia/nemotron-3-super-120b-a12b`), which is why NIM throttling is
the only visible symptom.

Earlier this chain was deliberately left unset because the only configured alternates
all rejected their credentials, so every fallback added a second failing call:

| Provider | Key | Result (at the time) |
| --- | --- | --- |
| `xai` | original 84-char key | valid, but `403 permission-denied` — no credits |
| `xai` | replacement 129-char key | `401 bad-credentials` |
| `qwencloud` | `QWENCLOUD_API_KEY` | `401 Invalid API-key provided` |
| `qwencloud_coding` | `QWENCLOUD_CODING_API_KEY` | `401 invalid access token or token expired` |

That is no longer true: `GROQ_API_KEY` and `MISTRAL_API_KEY` are now configured in
`~/.fcc/.env`, and the Admin Providers tab reports 11 Groq and 46 Mistral models, so a
fallback chain is viable. Use `provider/model` entries (the provider id is the first
path segment), for example:

```
MODEL_FALLBACKS=groq/llama-3.3-70b-versatile,mistral/mistral-large-latest
```

Point `MODEL` at a Groq or Mistral model instead if you would rather not route through
NIM at all, or at `ollama/<model>` for a zero-quota local path (Ollama is reachable on
this machine at `http://localhost:11434`).

Note `/models` is not by itself proof of a working credential — it can return 200 with
no key at all (observed on NVIDIA NIM and QwenCloud). Confirm a chosen fallback with a
real `/chat/completions` before relying on it.

## Provider retry budget

Free tiers throttle without warning (HTTP 429/529 plus vendor-specific codes), so one
logical request could previously exhaust its five-attempt budget and surface the error
to the harness, which ends the turn before a per-minute quota window reopens. The retry
budget and backoff are now configurable, and the server ships with a more patient
default, so a throttled request waits the provider out instead of failing:

| Env var | Default | Meaning |
| --- | --- | --- |
| `PROVIDER_MAX_ATTEMPTS` | `8` | Attempts one request may make before the failure is surfaced. |
| `PROVIDER_RETRY_BASE_DELAY` | `2.0` | Seconds before the first retry; doubles each attempt. |
| `PROVIDER_RETRY_MAX_DELAY` | `120.0` | Cap, in seconds, on one backoff. |
| `PROVIDER_RETRY_JITTER` | `1.0` | Maximum random seconds added per retry. |

The policy lives in `providers/admission_policy.py` (`ProviderAdmissionLimits`), is
validated in `config/settings.py`, and is plumbed through `ProviderAdmissionRegistry`
and `ProviderAdmissionController.reconfigure` — so changing it in the Admin takes
effect on new requests without a restart. The low-level library default stays 5
attempts; only the application default is raised. Retries stay bounded and serial
(coordinated recovery keeps a single probe in flight), so a longer budget does not add
throttle pressure.

Worst case with the defaults is about 2+4+8+16+32+64+120 ≈ 246 s of backoff, well
inside `PROVIDER_PROGRESS_TIMEOUT` (600 s). If you raise the budget, keep the summed
backoff below that timeout, or the progress watchdog cuts the request anyway.

## Not a fault: only some `fcc-*` launchers work

Every `fcc-<tool>` is a thin launcher that requires the real `<tool>` CLI to already be
installed and on `PATH`. FCC bundles none of them, so a launcher reporting
`Could not find <Tool> CLI command` means the host CLI is absent — nothing is wrong
with FCC, and the request never reaches the proxy.

On this machine only `claude` and `pi` are installed, so only `fcc-claude` and
`fcc-pi` work. `fcc-desktop` works because it is FCC's own admin UI and needs no host
CLI. `fcc-opencode` was the one exception: OpenCode was installed at
`%USERPROFILE%\.opencode\bin` but that directory was missing from `PATH`. It has been
appended to the user `PATH`.

Two stale `PATH` entries remain and are harmless, since neither directory contains a
usable binary:

- `%LOCALAPPDATA%\Programs\OpenAI\Codex\bin` — directory does not exist
- `%USERPROFILE%\.cline-fcc` — no Cline extension installed in VS Code

To confirm which harnesses are usable, check for the host CLI first. Diagnosing this
through `server.log` is misleading: a launcher that cannot find its host CLI exits
before making any request, so it leaves no trace there.
