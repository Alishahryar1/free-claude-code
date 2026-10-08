# Server as a launchd service (macOS)

Run the free-claude-code server as a managed background service with simple start/stop/restart.

## Install

```bash
./scripts/install-service.sh   # writes ~/Library/LaunchAgents/com.fcc.server.plist
./scripts/fccd start
```

## Control

```bash
./scripts/fccd start      # load + start
./scripts/fccd stop       # stop + unload
./scripts/fccd restart    # kickstart in place
./scripts/fccd status     # state, pid, last exit code
./scripts/fccd logs       # tail server.log
./scripts/fccd enable     # auto-start on login
./scripts/fccd disable    # remove auto-start
```

The service runs `uv run uvicorn server:app` on `0.0.0.0:8082`, restarts on crash
(`KeepAlive`/`SuccessfulExit=false`), and writes stdout+stderr to `server.log`
in the repo root.

To change host/port, edit `scripts/com.fcc.server.plist.template` and re-run
`install-service.sh`, then `fccd restart`.
