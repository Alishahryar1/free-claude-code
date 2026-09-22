Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try {
    if ($args.Count -ne 0) { throw "Usage: fcc-update (stop other FCC commands first)" }
    $uvPath = (Get-Command uv -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    $toolDir = (& $uvPath tool dir | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $pythonPath = Join-Path $toolDir "free-claude-code\Scripts\python.exe"
    $launcher = Join-Path $PSScriptRoot "fcc-update.cmd"
    $json = & $pythonPath -I -m free_claude_code.cli.update --uv $uvPath --launcher $launcher
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $plan = ($json | Out-String) | ConvertFrom-Json
    Write-Host "Updating FCC $($plan.version)..."
    # The installed Python process is gone before replacement starts.
    $uvArguments = @($plan.arguments)
    & $plan.uv @uvArguments
    $updateExit = $LASTEXITCODE
    if ($updateExit -ne 0) { exit $updateExit }
    $verify = @($plan.verify)
    $verifyArguments = @($verify | Select-Object -Skip 1)
    & $verify[0] @verifyArguments
    exit $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine("fcc-update: $($_.Exception.Message)")
    exit 1
}
