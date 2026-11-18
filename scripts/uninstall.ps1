param(
    [switch] $DryRun,
    [switch] $KeepIntegrations,
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
    "fcc-agy",
    "fcc-doctor",
    "fcc-update",
    "fcc-init",
    "free-claude-code"
)
$script:UvPath = ""
$script:UvToolBin = ""

function Show-Usage {
    @"
Usage: uninstall.ps1 [options]

Removes the Free Claude Code installation and deletes ~/.fcc/ after removal is verified.
Does not remove uv, Claude Code, Codex, Pi, OpenCode, Cline, Hermes Agent, DeepSeek Harness, Grok Build, Muse Code, Aider, Antigravity CLI, the uv-managed Python runtime, or shared PATH entries.

Options:
  -DryRun                Print commands without running them.
  -KeepIntegrations      Leave editor/app integrations unchanged.
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
        $global:LASTEXITCODE = $null
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
    param([switch] $Optional)
    Add-KnownUvPaths

    if ($DryRun) {
        Write-Host "+ uv tool dir --bin"
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        if ($Optional -and ((Test-FccBundleOnUserPath) -or -not (Test-UnbundledFccCommand))) { return }
        throw "uv is required to remove the Free Claude Code tool. Install uv, then rerun this uninstaller; ~/.fcc was not deleted."
    }
    $script:UvPath = $uvCommand.Source

    $commandText = Format-Command -FilePath $script:UvPath -Arguments @("tool", "dir", "--bin")
    Write-Host "+ $commandText"
    $result = Invoke-NativeResult -FilePath $script:UvPath -Arguments @("tool", "dir", "--bin")
    if ($Optional -and ($result.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($result.Output))) {
        if (Test-FccBundleOnUserPath) { return }
    }
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
    if (-not $DryRun -and -not $script:UvToolBin) { return }
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

function Disconnect-FccIntegrations {
    if ($KeepIntegrations) {
        return
    }
    if ($DryRun) {
        Write-Host "+ fcc-doctor --disconnect-integrations"
        return
    }
    $bundleBin = Get-FccBundleBin
    $bundleDoctor = Join-Path $bundleBin 'fcc-doctor.exe'
    $doctorPath = $null
    if ($script:UvToolBin) {
        foreach ($extension in @(".exe", ".cmd", ".bat", "")) {
            $candidate = Join-Path $script:UvToolBin "fcc-doctor$extension"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $doctorPath = $candidate
                break
            }
        }
    }
    if ((-not $doctorPath -or (Test-FccBundleOnUserPath)) -and (Test-Path -LiteralPath $bundleDoctor -PathType Leaf)) {
        $doctorPath = $bundleDoctor
    }
    if (-not $doctorPath) {
        throw "FCC's integration cleanup command is unavailable. Update FCC first, or disconnect editor/app integrations manually and rerun with -KeepIntegrations. FCC data was not deleted."
    }
    Write-Host "+ $(Format-Command -FilePath $doctorPath -Arguments @('--disconnect-integrations'))"
    $result = Invoke-NativeResult -FilePath $doctorPath -Arguments @("--disconnect-integrations")
    if (-not [string]::IsNullOrWhiteSpace($result.Output)) {
        Write-Host $result.Output
    }
    if ($result.ExitCode -ne 0) {
        throw "Editor/app integration cleanup failed. Correct the reported problem and retry. For older FCC versions, update first or disconnect manually and rerun with -KeepIntegrations. FCC and its data were not deleted."
    }
}

function Confirm-FccCommandsRemoved {
    if ($DryRun) {
        Write-Host "+ verify all Free Claude Code entry points are absent from the uv tool bin directory"
        return
    }
    if (-not $script:UvToolBin) { return }

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

function Test-PathEntry {
    param([string] $Entry, [string] $Directory, [switch] $Expand)
    if ($Expand) { $Entry = [Environment]::ExpandEnvironmentVariables($Entry) }
    # Relative entries can resolve differently in the next shell.
    if ($Entry -notmatch '^(?:[a-z]:[\\/]|[\\/]{2}[^\\/])') { return $false }
    return Test-EquivalentPath -Left $Entry -Right $Directory
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
            [IO.Path]::GetFullPath($Left).TrimEnd('\', '/'),
            [IO.Path]::GetFullPath($Right).TrimEnd('\', '/'),
            [StringComparison]::OrdinalIgnoreCase
        )
    }
    catch {
        return $false
    }
}

function Test-FccDesktopShortcutTarget {
    param([string] $TargetPath)

    if (Test-EquivalentPath -Left $TargetPath -Right (Join-Path (Get-FccBundleBin) 'fcc-desktop.exe')) { return $true }
    if (-not $script:UvToolBin) { return $false }
    foreach ($extension in @("", ".exe", ".cmd", ".bat", ".ps1")) {
        $expectedTarget = Join-Path $script:UvToolBin "fcc-desktop$extension"
        if (Test-EquivalentPath -Left $TargetPath -Right $expectedTarget) {
            return $true
        }
    }
    return $false
}

function Get-FccHome {
    return Join-Path $env:USERPROFILE $FccHomeDirname
}

function Get-FccBundlePath {
    return Join-Path (Get-FccHome) 'windows-app'
}

function Get-FccBundleBin {
    return Join-Path (Get-FccBundlePath) 'bin'
}

function Test-FccBundleOnUserPath {
    $bin = Get-FccBundleBin
    foreach ($directory in ([Environment]::GetEnvironmentVariable("Path", "User") -split ';')) {
        if (Test-PathEntry -Entry $directory -Directory $bin) { return $true }
    }
    return $false
}

function Test-UnbundledFccCommand {
    $bin = Get-FccBundleBin
    $originalPath = $env:Path
    try {
        # The current terminal may predate the switch to the uv-managed commands.
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath) { $env:Path = "$originalPath;$userPath" }
        $command = Get-Command -Name $FccCommands -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-PathEntry -Entry (Split-Path -Parent $_.Source) -Directory $bin) } |
            Select-Object -First 1
        return $null -ne $command
    } finally {
        $env:Path = $originalPath
    }
}

function Remove-FccBundlePathEntry {
    param([string] $BundlePath)

    if ($DryRun) { return }
    $bin = Get-FccBundleBin
    $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey("Environment")
    try {
        $path = $key.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if ($path) {
            $kind = $key.GetValueKind("Path")
            $expand = $kind -eq 'ExpandString'
            $newPath = ($path -split ';' | Where-Object {
                -not (Test-PathEntry -Entry $_ -Directory $bin -Expand:$expand)
            }) -join ';'
            if ($newPath -cne $path) { $key.SetValue("Path", $newPath, $kind) }
        }
    } finally {
        $key.Dispose()
    }
    $launcher = Join-Path $BundlePath 'bin\fcc-server.exe'
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        $null = Invoke-NativeResult -FilePath $launcher -Arguments @('--fcc-notify-environment')
    }
    $env:Path = ($env:Path -split ';' | Where-Object {
        -not (Test-PathEntry -Entry $_ -Directory $bin)
    }) -join ';'
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
    param([switch] $SignedBundle)

    $fccHome = Get-FccHome
    $target = $fccHome
    if ($SignedBundle) {
        $target = "$fccHome.removing"
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            Write-Host "+ Move-Item -LiteralPath $(Format-Argument $fccHome) -Destination $(Format-Argument $target)"
            # Move the whole directory before deleting any files. A failed move preserves
            # the working commands, and a pending directory can be purged on retry.
            if (-not $DryRun) { [IO.Directory]::Move($fccHome, $target) }
        }
    } elseif (-not (Test-Path -LiteralPath $fccHome)) {
        Write-Host "No FCC config directory at $fccHome; skipping purge."
        return
    }

    $commandText = @(
        "Remove-Item",
        "-LiteralPath",
        (Format-Argument $target),
        "-Recurse",
        "-Force"
    ) -join " "
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    if ($SignedBundle) { Remove-FccBundlePathEntry -BundlePath (Join-Path $target 'windows-app') }
    Remove-Item -LiteralPath $target -Recurse -Force
    if (Test-Path -LiteralPath $target) {
        throw "FCC config directory still exists after deletion: $target"
    }
}

function Lock-FccInstallation {
    # Use the installer lock without holding a file inside the directory being purged.
    $path = Join-Path $env:USERPROFILE '.fcc-install.lock'
    try {
        return [IO.FileStream]::new(
            $path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None, 4096, [IO.FileOptions]::DeleteOnClose
        )
    } catch {
        throw "Could not lock the FCC installation. Wait for any other FCC installer or uninstaller to finish. $($_.Exception.Message)"
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

$installLock = $null
try {
    if (-not $DryRun) { $installLock = Lock-FccInstallation }
    Write-Step "Checking for running Free Claude Code processes"
    Assert-NoFccProcessesRunning

    $fccHome = Get-FccHome
    $resumingBundleRemoval = Test-Path -LiteralPath "$fccHome.removing" -PathType Container
    $bundlePath = Get-FccBundlePath
    $previousBundle = "$bundlePath.previous"
    $hasBundle = Test-Path -LiteralPath $bundlePath -PathType Container
    if (-not $resumingBundleRemoval) {
        if (-not $hasBundle -and (Test-Path -LiteralPath $previousBundle -PathType Container)) {
            if (-not $DryRun) { [IO.Directory]::Move($previousBundle, $bundlePath) }
            $hasBundle = $true
        }
        Write-Step "Locating the Free Claude Code installation"
        $hasFccHome = Test-Path -LiteralPath $fccHome
        Initialize-UvContext -Optional:($hasBundle -or -not $hasFccHome)
        if (-not $DryRun -and -not $hasFccHome -and -not $script:UvPath -and
            -not ($FccCommands | Where-Object { Get-ApplicationCommand $_ } | Select-Object -First 1)) {
            Write-Host "Free Claude Code is already absent."
            return
        }

        Write-Step "Disconnecting FCC editor/app integrations"
        Disconnect-FccIntegrations

        Write-Step "Removing any Free Claude Code uv tool installation"
        Uninstall-FreeClaudeCode

        Write-Step "Verifying Free Claude Code entry points were removed"
        Confirm-FccCommandsRemoved

        Write-Step "Removing Free Claude Code desktop shortcuts"
        Remove-FccDesktopShortcuts
    }

    Write-Step "Purging FCC config and data from ~/.fcc"
    Purge-FccHome -SignedBundle:($hasBundle -or $resumingBundleRemoval)
} finally {
    if ($installLock) { $installLock.Dispose() }
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No changes were made."
}
else {
    if ($resumingBundleRemoval) {
        Write-Host "Finished removing the signed Windows bundle and FCC data."
    } elseif ($hasBundle -and -not $script:UvToolBin) {
        Write-Host "The signed Windows bundle has been removed. uv could not verify any separate uv installation."
    } else {
        Write-Host "Free Claude Code has been removed and verified."
    }
    Write-Host "uv, Claude Code, Codex, Pi, OpenCode, Cline, Hermes Agent, DeepSeek Harness, Grok Build, Muse Code, Aider, the uv-managed Python runtime, and shared PATH entries were left installed."
}
