param(
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PackageName = "free-claude-code"
$FccHomeDirname = ".fcc"
$FccCommands = @(
    # Include retired entry points so older installations are fully stopped and removed.
    "fcc-desktop",
    "fcc-server",
    "fcc-claude",
    "fcc-codex",
    "fcc-pi",
    "fcc-opencode",
    "fcc-cline",
    "fcc-hermes",
    "fcc-dsh",
    "fcc-grok",
    "fcc-muse",
    "fcc-aider",
    "fcc-init",
    "free-claude-code"
)
$script:UvPath = ""
$script:UvToolBin = ""

function Show-Usage {
    @"
Usage: uninstall.ps1 [options]

Removes the Free Claude Code uv tool and deletes ~/.fcc/ after removal is verified.
Does not remove uv, Claude Code, Codex, Pi, OpenCode, Cline, Hermes Agent, DeepSeek Harness, Grok Build, Muse Code, Aider, the uv-managed Python runtime, or shared PATH entries.

Options:
  -DryRun                Print commands without running them.
  -Help                  Show this help text.
"@
}

function Write-Step {
    param([string] $Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Format-Argument {
    param([string] $Value)

    if ($Value -match '^[A-Za-z0-9_./:@%+=,\[\]\\-]+$') {
        return $Value
    }
    return "'" + ($Value -replace "'", "''") + "'"
}

function Format-Command {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $parts = @($FilePath) + $Arguments
    return ($parts | ForEach-Object { Format-Argument ([string] $_) }) -join " "
}

function Get-ApplicationCommand {
    param([string] $Name)

    $commands = @(Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
    if ($commands.Count -eq 0) {
        return $null
    }
    return $commands[0]
}

function Invoke-NativeResult {
    param(
        [string] $FilePath,
        [string[]] $Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $global:LASTEXITCODE = 0
        $output = (& $FilePath @Arguments 2>&1 | Out-String).Trim()
        return [pscustomobject] @{
            ExitCode = $LASTEXITCODE
            Output = $output
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Test-MissingUvToolError {
    param([string] $Output)

    $normalized = $Output.ToLowerInvariant()
    return $normalized.Contains($PackageName) -and $normalized.Contains("is not installed")
}

function Add-PathEntry {
    param([string] $PathEntry)

    if ([string]::IsNullOrWhiteSpace($PathEntry)) {
        return
    }
    $separator = [IO.Path]::PathSeparator
    $entries = @()
    if (-not [string]::IsNullOrEmpty($env:Path)) {
        $entries = $env:Path -split [regex]::Escape([string] $separator)
    }
    if ($entries -notcontains $PathEntry) {
        $env:Path = "$PathEntry$separator$env:Path"
    }
}

function Add-KnownUvPaths {
    Add-PathEntry (Join-Path $env:USERPROFILE ".local\bin")
    Add-PathEntry (Join-Path $env:USERPROFILE ".cargo\bin")
}

function Assert-NoFccProcessesRunning {
    $running = @()
    foreach ($commandName in $FccCommands) {
        $processes = @(Get-Process -Name $commandName -ErrorAction SilentlyContinue)
        if ($processes.Count -gt 0) {
            $running += $commandName
        }
    }
    if ($running.Count -gt 0) {
        throw "Free Claude Code is still running ($($running -join ', ')). Stop those processes, then rerun uninstall."
    }
}

function Initialize-UvContext {
    Add-KnownUvPaths

    if ($DryRun) {
        Write-Host "+ uv tool dir --bin"
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv is required to remove the Free Claude Code tool. Install uv, then rerun this uninstaller; ~/.fcc was not deleted."
    }
    $script:UvPath = $uvCommand.Source

    $commandText = Format-Command -FilePath $script:UvPath -Arguments @("tool", "dir", "--bin")
    Write-Host "+ $commandText"
    $result = Invoke-NativeResult -FilePath $script:UvPath -Arguments @("tool", "dir", "--bin")
    if ($result.ExitCode -ne 0) {
        if (-not [string]::IsNullOrWhiteSpace($result.Output)) {
            [Console]::Error.WriteLine($result.Output)
        }
        throw "Could not determine the uv tool bin directory (exit code $($result.ExitCode)); ~/.fcc was not deleted."
    }
    $script:UvToolBin = $result.Output.Trim()
    if ([string]::IsNullOrWhiteSpace($script:UvToolBin)) {
        throw "uv returned an empty tool bin directory; ~/.fcc was not deleted."
    }
}

function Uninstall-FreeClaudeCode {
    Write-Host "+ uv tool uninstall $PackageName"
    if ($DryRun) {
        return
    }

    $result = Invoke-NativeResult -FilePath $script:UvPath -Arguments @(
        "tool",
        "uninstall",
        $PackageName
    )
    if ($result.ExitCode -eq 0) {
        if (-not [string]::IsNullOrWhiteSpace($result.Output)) {
            Write-Host $result.Output
        }
        return
    }
    if (Test-MissingUvToolError -Output $result.Output) {
        Write-Host "Free Claude Code uv tool is already absent; verifying its entry points."
        return
    }
    if (-not [string]::IsNullOrWhiteSpace($result.Output)) {
        [Console]::Error.WriteLine($result.Output)
    }
    throw "uv tool uninstall $PackageName failed with exit code $($result.ExitCode); ~/.fcc was not deleted."
}

function Confirm-FccCommandsRemoved {
    if ($DryRun) {
        Write-Host "+ verify all Free Claude Code entry points are absent from the uv tool bin directory"
        return
    }

    $remaining = @()
    $extensions = @("", ".exe", ".cmd", ".bat", ".ps1")
    foreach ($commandName in $FccCommands) {
        foreach ($extension in $extensions) {
            $commandPath = Join-Path $script:UvToolBin "$commandName$extension"
            if (Test-Path -LiteralPath $commandPath) {
                $remaining += $commandPath
            }
        }
    }
    if ($remaining.Count -gt 0) {
        throw "Free Claude Code entry points remain after uv uninstall: $($remaining -join ', '); ~/.fcc was not deleted."
    }
}

function Test-EquivalentPath {
    param(
        [string] $Left,
        [string] $Right
    )

    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) {
        return $false
    }
    try {
        return [string]::Equals(
            [IO.Path]::GetFullPath($Left),
            [IO.Path]::GetFullPath($Right),
            [StringComparison]::OrdinalIgnoreCase
        )
    }
    catch {
        return $false
    }
}

function Test-FccDesktopShortcutTarget {
    param([string] $TargetPath)

    foreach ($extension in @("", ".exe", ".cmd", ".bat", ".ps1")) {
        $expectedTarget = Join-Path $script:UvToolBin "fcc-desktop$extension"
        if (Test-EquivalentPath -Left $TargetPath -Right $expectedTarget) {
            return $true
        }
    }
    return $false
}

function Remove-FccDesktopShortcuts {
    $shortcutPaths = @(
        (Join-Path $env:USERPROFILE "Desktop\Free Claude Code.lnk"),
        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Free Claude Code.lnk")
    )
    $shell = New-Object -ComObject WScript.Shell
    foreach ($shortcutPath in $shortcutPaths) {
        if (-not (Test-Path -LiteralPath $shortcutPath)) {
            continue
        }
        try {
            $shortcut = $shell.CreateShortcut($shortcutPath)
            $isFccShortcut = Test-FccDesktopShortcutTarget -TargetPath $shortcut.TargetPath
        }
        catch {
            $isFccShortcut = $false
        }
        if (-not $isFccShortcut) {
            Write-Host "A shortcut not managed by Free Claude Code exists at $shortcutPath; leaving it unchanged."
            continue
        }
        Write-Host "+ Remove-Item -LiteralPath $(Format-Argument $shortcutPath) -Force"
        if (-not $DryRun) {
            Remove-Item -LiteralPath $shortcutPath -Force
        }
    }
}

function Purge-FccHome {
    $fccHome = Join-Path $env:USERPROFILE $FccHomeDirname
    if (-not (Test-Path -LiteralPath $fccHome)) {
        Write-Host "No FCC config directory at $fccHome; skipping purge."
        return
    }

    $commandText = @(
        "Remove-Item",
        "-LiteralPath",
        (Format-Argument $fccHome),
        "-Recurse",
        "-Force"
    ) -join " "
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    Remove-Item -LiteralPath $fccHome -Recurse -Force
    if (Test-Path -LiteralPath $fccHome) {
        throw "FCC config directory still exists after deletion: $fccHome"
    }
}

if ($Help) {
    Show-Usage
    return
}
if ($RemainingArgs.Count -gt 0) {
    Show-Usage
    throw "Unknown option: $($RemainingArgs -join ' ')"
}
if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
    throw "USERPROFILE is not set; cannot locate Free Claude Code data."
}

Write-Step "Checking for running Free Claude Code processes"
Assert-NoFccProcessesRunning

Write-Step "Locating the uv-managed Free Claude Code installation"
Initialize-UvContext

Write-Step "Removing the Free Claude Code uv tool"
Uninstall-FreeClaudeCode

Write-Step "Verifying Free Claude Code entry points were removed"
Confirm-FccCommandsRemoved

Write-Step "Removing Free Claude Code desktop shortcuts"
Remove-FccDesktopShortcuts

Write-Step "Purging FCC config and data from ~/.fcc"
Purge-FccHome

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No changes were made."
}
else {
    Write-Host "Free Claude Code has been removed and verified."
    Write-Host "uv, Claude Code, Codex, Pi, OpenCode, Cline, Hermes Agent, DeepSeek Harness, Grok Build, Muse Code, Aider, the uv-managed Python runtime, and shared PATH entries were left installed."
}
