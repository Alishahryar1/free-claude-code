# Deploy the local FCC patches to the live install.

# Run this from an elevated PowerShell if you also want the Defender
# exclusions. Step 2 restarts the FCC server, which disconnects any running
# fcc-claude / fcc-codex sessions, so do it when none are mid-conversation.

$repo = "$env:USERPROFILE\OneDrive\Documents\Default Project\free-claude-code"
$uv   = "$env:USERPROFILE\.local\bin\uv.exe"

# 1. Confirm the working tree carries both patches.
Set-Location $repo
git diff --stat
if ($LASTEXITCODE -ne 0) { throw "not a git repo: $repo" }

# 2. Stop everything that can hold a lock on the install directory, or that
#    would respawn into it. An abandoned fcc-claude session keeps the tool venv's
#    Scripts\ open and uv then fails with "Access is denied (os error 5)".
#    Two families must be matched: python.exe children whose ExecutablePath sits
#    under the tool directory, and the fcc-* uv shims in ~/.local/bin, which live
#    outside it and would otherwise outlive this script and rebind port 8082.
$toolRoot = "$env:APPDATA\uv\tools\free-claude-code"
$holders = Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -like "$toolRoot*" -or
    $_.CommandLine -like "*$toolRoot*" -or
    $_.Name -like "fcc-*"
}
foreach ($p in $holders) {
    "stopping PID $($p.ProcessId) [$($p.Name)]"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3

# 3. Reinstall from this clone instead of the stale temp copy.
#    uv reads the working tree, so the patched source is what lands in site-packages.
#    UV_LINK_MODE=copy avoids a hardlink warning when the cache spans filesystems.
$env:UV_LINK_MODE = "copy"
& $uv tool install --force --reinstall --python 3.14 $repo
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }

# 4. Prove the deployed copy has the recovery code.
$installed = "$env:APPDATA\uv\tools\free-claude-code\Lib\site-packages\free_claude_code\cli\commands.py"
if (Select-String -Path $installed -Pattern "_CRASH_RESTART_BASE_DELAY_SECONDS" -Quiet) {
    "OK: crash-recovery code is live"
} else {
    "FAIL: deployed copy does not contain the patch"
}

# 5. Record where the install came from, so the next audit is not a surprise again.
Get-ChildItem -LiteralPath "$env:APPDATA\uv\tools\free-claude-code\Lib\site-packages" `
    -Filter "free_claude_code*.dist-info" -Directory |
    ForEach-Object { Get-Content -LiteralPath (Join-Path $_.FullName "direct_url.json") }

# 6. Do NOT start the server from this script.
#    Step 2 stopped it, and a hidden background instance is actively harmful:
#    it takes port 8082, so the next 'fcc-server' you type dies with
#    "[WinError 10048] Only one usage of each socket address..." and the console
#    log that would explain why is nowhere on screen. Start it yourself:
#        fcc-server      # foreground: logs to the console, Ctrl+C to stop
#        fcc-desktop     # tray; reuses a healthy peer instead of failing
#
#    Only bare fcc-server hard-fails on a taken port. The graceful handoff in
#    ServerSupervisor.run() is reachable solely through cli/desktop.py, the only
#    caller that passes existing_server=.
$stale = Get-NetTCPConnection -State Listen -LocalPort 8082 -ErrorAction SilentlyContinue
if ($stale) {
    "WARN: port 8082 is still held by PID $($stale.OwningProcess) - fcc-server will fail with WinError 10048"
} else {
    "port 8082 is free; run 'fcc-server' to bring the proxy up"
}
