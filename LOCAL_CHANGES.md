# FCC local changes

Changes layered on top of upstream `Alishahryar1/free-claude-code` that are not
part of any release. `fcc-update` reinstalls from upstream and will silently drop
everything here, so re-apply after updating.

## Current deployment

As of 2026-09-30 the live install is built from this clone, not from upstream:

```
free_claude_code-6.5.10.dev0+g549918e2e.d20260930
direct_url.json -> file:///C:/Users/aams1/OneDrive/Documents/Default%20Project/free-claude-code
```

It previously resolved to `0.0.0` from a `main.zip` extracted under `%TEMP%`, which
meant an unversioned, uncleanable source tree. Fixing that is what let this patch be
deployed at all.

## Applying after an update

`fcc-update` reinstalls from upstream. That drops both patches, resets the version to
`0.0.0`, and points `direct_url.json` at the temp copy again. To get back to a
patched install:

```powershell
cd "$env:USERPROFILE\OneDrive\Documents\Default Project\free-claude-code"
git pull                       # may conflict with the local edits below
.\DEPLOY.ps1                   # reinstalls from this clone, restarts, health-checks
```

`DEPLOY.ps1` stops whatever holds a lock on the tool directory, reinstalls, verifies
the recovery code is actually present in site-packages, restarts the server, and polls
`/admin/api/status` until it answers. It is the whole procedure; do not run the
`uv tool install` line by hand unless you have a reason to.

If `git pull` conflicts, the patch is one hunk region in
`src/free_claude_code/cli/commands.py`. If upstream refactors `ServerSupervisor.run`,
resolve by hand: keep the upstream loop structure and re-add the pieces below.

## Deploying: two things that will bite you otherwise

1. **A stale process locks the install.** An abandoned `fcc-claude` session keeps
   `%APPDATA%\uv\tools\free-claude-code\Scripts` open and `uv tool install` fails with
   `Access is denied (os error 5)`. Match on the tool directory, not the process name,
   so orphaned `python.exe` children are caught too.
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

## Not fixed: the 429s

The `429 Too Many Requests` entries in `~/.fcc/logs/server.log` are NVIDIA NIM
enforcing its own free-tier quota, not a fault in FCC. The admission controller
retries five times with backoff and then surfaces the error, which is the correct
behaviour. Raising `PROVIDER_RATE_LIMIT` or `PROVIDER_MAX_CONCURRENCY` would
provoke more 429s, not fewer. Configure fallback models in Admin so a throttled
provider does not fail a turn.
