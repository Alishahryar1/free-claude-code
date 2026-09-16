param(
    [switch] $VoiceNim,
    [switch] $VoiceLocal,
    [switch] $VoiceAll,
    [string] $TorchBackend = "",
    [switch] $Rtk,
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoArchiveUrl = "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
# Windows on ARM emulates x64, whose Python package ecosystem has broader wheel support.
$PythonRequest = "cpython-3.14.0-windows-x86_64-none"
$MinUvVersion = "0.12.13"
$ClaudeInstallUrl = "https://claude.ai/install.ps1"
$CodexInstallUrl = "https://chatgpt.com/codex/install.ps1"
$PiInstallUrl = "https://pi.dev/install.ps1"
$OpenCodeReleaseBaseUrl = "https://github.com/anomalyco/opencode/releases/latest/download"
$HermesInstallUrl = "https://hermes-agent.nousresearch.com/install.ps1"
$DshVersion = "0.1.0-rc.8"
$DshPackage = "@deepseek-ai/dsh@$DshVersion"
$GrokInstallUrl = "https://x.ai/cli/install.ps1"
$MuseInstallUrl = "https://dev.meta.ai/install.ps1"
$MuseLegacyOwner = "free-claude-code-muse-installer"
$MuseLegacyOwnerSchemaVersion = 1
$RtkVersion = "0.44.2"
$RtkReleaseBaseUrl = "https://github.com/rtk-ai/rtk/releases/download/v$RtkVersion"
$RtkWindowsAssetName = "rtk-x86_64-pc-windows-msvc.zip"
$RtkWindowsAssetSha256 = "3a1e114edce9080f8a10663e9c87488363a82f14a5ca8aab2ad416817f89d47c"
$UvInstallUrl = "https://astral.sh/uv/install.ps1"
$script:InstallClaudeCode = $true
$script:InstallCodex = $true
$script:InstallPi = $true
$script:InstallOpenCode = $true
$script:InstallCline = $false
$script:InstallHermes = $true
$script:InstallDsh = $true
$script:InstallGrok = $true
$script:InstallMuse = $true
$script:InstallAider = $true
$script:PiAvailable = $false
$script:MuseAvailable = $false
$script:EnableRtk = $Rtk.IsPresent
$FccCommands = @(
    # Include retired entry points so updates reject older FCC processes before replacement.
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

function Show-Usage {
    @"
Usage: install.ps1 [options]

Installs or updates Free Claude Code and lets you choose which coding agents to install or verify.

Options:
  -VoiceNim              Install NVIDIA NIM voice transcription support.
  -VoiceLocal            Install local Whisper voice transcription support.
  -VoiceAll              Install all voice transcription backends.
  -TorchBackend VALUE    Use a uv PyTorch backend, such as cu130. Requires local voice.
  -Rtk                   Install and configure RTK for the selected coding agents.
  -DryRun                Print commands without running them.
  -Help                  Show this help text.
"@
}

function Write-Step {
    param([string] $Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Test-InteractiveInstaller {
    return (-not [Console]::IsInputRedirected) -and (-not [Console]::IsOutputRedirected)
}

function Read-YesNo {
    param(
        [string] $Prompt,
        [bool] $DefaultYes = $true
    )

    while ($true) {
        $hint = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
        $answer = ([string] (Read-Host "$Prompt $hint")).Trim().ToLowerInvariant()
        if ($answer -eq "") {
            return $DefaultYes
        }
        if ($answer -in @("y", "yes")) {
            return $true
        }
        if ($answer -in @("n", "no")) {
            return $false
        }
        Write-Host "Please answer Y or N."
    }
}

function Select-CodingAgents {
    while ($true) {
        $script:InstallClaudeCode = Read-YesNo "Install or verify Claude Code for fcc-claude?"
        $script:InstallCodex = Read-YesNo "Install or verify Codex for fcc-codex?"
        $script:InstallPi = Read-YesNo "Install or verify Pi for fcc-pi?"
        $script:InstallOpenCode = Read-YesNo "Install or verify OpenCode for fcc-opencode?"
        $script:InstallCline = Read-YesNo `
            -Prompt "Install or verify Cline CLI for fcc-cline?" `
            -DefaultYes $script:InstallCline
        $script:InstallHermes = Read-YesNo `
            -Prompt "Install or verify Hermes Agent for fcc-hermes?" `
            -DefaultYes $script:InstallHermes
        $script:InstallDsh = Read-YesNo `
            -Prompt "Install or verify DeepSeek Harness for fcc-dsh?" `
            -DefaultYes $script:InstallDsh
        $script:InstallGrok = Read-YesNo `
            -Prompt "Install or verify Grok Build for fcc-grok?" `
            -DefaultYes $script:InstallGrok
        $script:InstallMuse = Read-YesNo `
            -Prompt "Install or verify Muse Code for fcc-muse?" `
            -DefaultYes $script:InstallMuse
        $script:InstallAider = Read-YesNo `
            -Prompt "Install or verify Aider for fcc-aider?" `
            -DefaultYes $script:InstallAider

        if ($script:InstallClaudeCode -or $script:InstallCodex -or $script:InstallPi -or $script:InstallOpenCode -or $script:InstallCline -or $script:InstallHermes -or $script:InstallDsh -or $script:InstallGrok -or $script:InstallMuse -or $script:InstallAider) {
            break
        }
        Write-Host "Select at least one coding agent."
        Write-Host ""
    }

    if (-not $script:EnableRtk) {
        $script:EnableRtk = Read-YesNo `
            -Prompt "Enable RTK token optimization globally for the selected coding agents?" `
            -DefaultYes $false
    }
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

function Invoke-NativeCommand {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    $global:LASTEXITCODE = 0
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }
}

function Invoke-Utf8NativeCapture {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    $originalOutputEncoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $global:LASTEXITCODE = 0
        $output = & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        [Console]::OutputEncoding = $originalOutputEncoding
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }

    return ($output | Out-String).Trim()
}

function Get-ApplicationCommand {
    param([string] $Name)

    $commands = @(Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
    if ($commands.Count -eq 0) {
        return $null
    }

    return $commands[0]
}

function Get-PowerShellExecutable {
    param([string] $PowerShellHome = $PSHOME)

    $executableName = if ($PSVersionTable.PSEdition -eq "Core") {
        "pwsh.exe"
    }
    else {
        "powershell.exe"
    }
    $bundledExecutable = Join-Path $PowerShellHome $executableName
    if (Test-Path -LiteralPath $bundledExecutable -PathType Leaf) {
        return $bundledExecutable
    }

    $pathCommand = Get-ApplicationCommand ([IO.Path]::GetFileNameWithoutExtension($executableName))
    if ($pathCommand) {
        return $pathCommand.Source
    }

    throw "Unable to locate a PowerShell executable for the downloaded installer."
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

function Prioritize-PathEntry {
    param([string] $PathEntry)

    if ([string]::IsNullOrWhiteSpace($PathEntry)) {
        return
    }

    $separator = [IO.Path]::PathSeparator
    $env:Path = "$PathEntry$separator$env:Path"
}

function Add-KnownBinDirectories {
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        Add-PathEntry (Join-Path $env:USERPROFILE ".local\bin")
        Add-PathEntry (Join-Path $env:USERPROFILE ".opencode\bin")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Add-PathEntry (Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\bin")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Add-PathEntry (Join-Path $env:LOCALAPPDATA "Programs\OpenAI\Codex\bin")
        Add-PathEntry (Join-Path $env:LOCALAPPDATA "pi-node\current")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:APPDATA)) {
        Add-PathEntry (Join-Path $env:APPDATA "npm")
    }
    if ($env:GROK_BIN_DIR) {
        Add-PathEntry $env:GROK_BIN_DIR
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        Add-PathEntry (Join-Path $env:USERPROFILE ".grok\bin")
    }
}

function Add-NpmBinDirectories {
    if ($DryRun) {
        return
    }

    Add-KnownBinDirectories
    $npm = Get-ApplicationCommand "npm"
    if (-not $npm) {
        return
    }

    $prefix = (& $npm.Source prefix -g 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($prefix)) {
        $prefix = (& $npm.Source config get prefix 2>$null | Out-String).Trim()
    }
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($prefix)) {
        Add-PathEntry $prefix
    }
}

function Assert-NoFccProcessesRunning {
    $running = @()
    foreach ($commandName in $FccCommands) {
        $processes = @(Get-Process -Name $commandName -ErrorAction SilentlyContinue)
        foreach ($process in $processes) {
            $running += "$commandName (PID $($process.Id))"
        }
    }

    if ($running.Count -gt 0) {
        throw "Free Claude Code is still running ($($running -join ', ')). Stop those processes, then rerun the installer."
    }
}

function Invoke-DownloadedPowerShellInstaller {
    param(
        [string] $Url,
        [string] $Name,
        [switch] $NonInteractive,
        [string[]] $ScriptArguments = @()
    )

    if ($DryRun) {
        Write-Host "+ irm $Url -OutFile <temporary-script>"
        $prefix = if ($NonInteractive) { "CODEX_NON_INTERACTIVE=1 " } else { "" }
        $suffix = if ($ScriptArguments.Count -gt 0) {
            " " + (($ScriptArguments | ForEach-Object { Format-Argument $_ }) -join " ")
        }
        else {
            ""
        }
        Write-Host "+ ${prefix}powershell -NoProfile -ExecutionPolicy Bypass -File <temporary-script>$suffix"
        return
    }

    $temporaryScript = Join-Path ([IO.Path]::GetTempPath()) ("fcc-install-" + [guid]::NewGuid().ToString("N") + ".ps1")
    try {
        Write-Host "+ irm $Url -OutFile $(Format-Argument $temporaryScript)"
        Invoke-RestMethod -Uri $Url -OutFile $temporaryScript -ErrorAction Stop
        if ((-not (Test-Path -LiteralPath $temporaryScript)) -or ((Get-Item -LiteralPath $temporaryScript).Length -eq 0)) {
            throw "The downloaded $Name installer was empty."
        }

        $tokens = $null
        $parseErrors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $temporaryScript,
            [ref] $tokens,
            [ref] $parseErrors
        ) | Out-Null
        if ($parseErrors.Count -gt 0) {
            throw "The downloaded $Name installer from '$Url' is not valid PowerShell. A network proxy or filter may have replaced it with an HTML response."
        }

        $powerShellPath = Get-PowerShellExecutable

        $hadNonInteractive = Test-Path Env:CODEX_NON_INTERACTIVE
        $previousNonInteractive = $env:CODEX_NON_INTERACTIVE
        try {
            if ($NonInteractive) {
                $env:CODEX_NON_INTERACTIVE = "1"
            }
            $installerArguments = @(
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                $temporaryScript
            ) + $ScriptArguments
            Invoke-NativeCommand -FilePath $powerShellPath -Arguments $installerArguments
        }
        finally {
            if ($hadNonInteractive) {
                $env:CODEX_NON_INTERACTIVE = $previousNonInteractive
            }
            else {
                Remove-Item Env:CODEX_NON_INTERACTIVE -ErrorAction SilentlyContinue
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryScript -Force -ErrorAction SilentlyContinue
    }
}

function Confirm-Application {
    param(
        [string] $CommandName,
        [string] $DisplayName
    )

    if ($DryRun) {
        Write-Host "+ $CommandName --version"
        return
    }

    $command = Get-ApplicationCommand $CommandName
    if (-not $command) {
        throw "$DisplayName was installed, but '$CommandName' is not available on PATH."
    }
    Invoke-NativeCommand -FilePath $command.Source -Arguments @("--version")
}

function Test-PiApplication {
    param($Command)

    try {
        $helpOutput = (& $Command.Source --help 2>$null | Out-String)
    }
    catch {
        return $false
    }
    return (
        $LASTEXITCODE -eq 0 -and
        $helpOutput.Contains("--extension") -and
        $helpOutput.Contains("--models")
    )
}

function Confirm-PiApplication {
    if ($DryRun) {
        Write-Host "+ pi --help (verify --extension and --models support)"
        Write-Host "+ pi --version"
        return
    }

    $command = Get-ApplicationCommand "pi"
    if (-not $command) {
        throw "Pi was installed, but 'pi' is not available on PATH."
    }
    if (-not (Test-PiApplication $command)) {
        throw "The 'pi' command at '$($command.Source)' is not a compatible Pi Coding Agent."
    }
    Invoke-NativeCommand -FilePath $command.Source -Arguments @("--version")
}

function Install-Rtk {
    $archiveUrl = "$RtkReleaseBaseUrl/$RtkWindowsAssetName"
    if ($DryRun) {
        Write-Host "+ irm $archiveUrl -OutFile <temporary-archive>"
        Write-Host "+ verify pinned SHA-256 for $RtkWindowsAssetName"
        Write-Host "+ extract and install rtk.exe to ~/.local/bin"
        return
    }

    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("fcc-rtk-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $temporaryRoot $RtkWindowsAssetName
    $extractPath = Join-Path $temporaryRoot "extracted"
    try {
        New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

        Write-Host "+ irm $archiveUrl -OutFile $(Format-Argument $archivePath)"
        Invoke-RestMethod -Uri $archiveUrl -OutFile $archivePath -ErrorAction Stop
        if ((-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) -or ((Get-Item -LiteralPath $archivePath).Length -eq 0)) {
            throw "The RTK release archive was empty."
        }

        $sha256 = [Security.Cryptography.SHA256]::Create()
        $archiveStream = [IO.File]::OpenRead($archivePath)
        try {
            $actualHash = [BitConverter]::ToString($sha256.ComputeHash($archiveStream)).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $archiveStream.Dispose()
            $sha256.Dispose()
        }
        if ($actualHash -ne $RtkWindowsAssetSha256) {
            throw "RTK checksum verification failed for $RtkWindowsAssetName."
        }

        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath
        $extractedExecutable = Join-Path $extractPath "rtk.exe"
        if (-not (Test-Path -LiteralPath $extractedExecutable -PathType Leaf)) {
            throw "The verified RTK archive did not contain rtk.exe."
        }

        $installDirectory = Join-Path $env:USERPROFILE ".local\bin"
        New-Item -ItemType Directory -Force -Path $installDirectory | Out-Null
        Copy-Item -LiteralPath $extractedExecutable -Destination (Join-Path $installDirectory "rtk.exe") -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryRoot) {
            Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-RtkCommand {
    param([string[]] $Arguments)

    if ($DryRun) {
        Write-Host "+ RTK_TELEMETRY_DISABLED=1 $(Format-Command -FilePath 'rtk' -Arguments $Arguments)"
        return
    }

    $command = Get-ApplicationCommand "rtk"
    if (-not $command) {
        throw "RTK was installed, but 'rtk' is not available on PATH."
    }

    $hadTelemetryDisabled = Test-Path Env:RTK_TELEMETRY_DISABLED
    $previousTelemetryDisabled = $env:RTK_TELEMETRY_DISABLED
    try {
        $env:RTK_TELEMETRY_DISABLED = "1"
        Invoke-NativeCommand -FilePath $command.Source -Arguments $Arguments
    }
    finally {
        if ($hadTelemetryDisabled) {
            $env:RTK_TELEMETRY_DISABLED = $previousTelemetryDisabled
        }
        else {
            Remove-Item Env:RTK_TELEMETRY_DISABLED -ErrorAction SilentlyContinue
        }
    }
}

function Ensure-RtkClaudeConfigDirectory {
    $claudeConfigDirectory = $env:CLAUDE_CONFIG_DIR
    if ([string]::IsNullOrWhiteSpace($claudeConfigDirectory)) {
        $claudeConfigDirectory = Join-Path $env:USERPROFILE ".claude"
    }

    if ($DryRun) {
        Write-Host "+ mkdir $(Format-Argument $claudeConfigDirectory)"
        return
    }

    New-Item -ItemType Directory -Force -Path $claudeConfigDirectory | Out-Null
}

function Confirm-RtkApplication {
    if ($DryRun) {
        Invoke-RtkCommand -Arguments @("--version")
        Invoke-RtkCommand -Arguments @("gain")
        return
    }

    $command = Get-ApplicationCommand "rtk"
    if (-not $command) {
        throw "RTK was installed, but 'rtk' is not available on PATH."
    }

    try {
        Invoke-RtkCommand -Arguments @("--version")
        Invoke-RtkCommand -Arguments @("gain")
    }
    catch {
        throw "The 'rtk' command at '$($command.Source)' is not a compatible Rust Token Killer installation. Remove the conflicting command from PATH, then rerun the installer. $($_.Exception.Message)"
    }
}

function Ensure-Rtk {
    if (Get-ApplicationCommand "rtk") {
        Write-Host "RTK already found on PATH; verifying it without updating it."
    }
    else {
        Install-Rtk
        Add-KnownBinDirectories
    }

    Confirm-RtkApplication
}

function Configure-RtkForSelectedAgents {
    if (-not $script:EnableRtk) {
        return
    }

    Write-Step "Installing and configuring RTK token optimization"
    Ensure-Rtk

    if ($script:InstallClaudeCode) {
        Ensure-RtkClaudeConfigDirectory
        Invoke-RtkCommand -Arguments @("init", "--global", "--auto-patch")
    }
    if ($script:InstallCodex) {
        Invoke-RtkCommand -Arguments @("init", "--global", "--codex")
    }
    if ($script:InstallPi -and $script:PiAvailable) {
        Invoke-RtkCommand -Arguments @("init", "--global", "--agent", "pi")
    }
    if ($script:InstallOpenCode) {
        Invoke-RtkCommand -Arguments @("init", "--global", "--opencode")
    }
    if ($script:InstallCline) {
        Write-Host "Optional for each project: cd <project>; `$env:RTK_TELEMETRY_DISABLED='1'; rtk init --agent cline"
    }
}

function Ensure-ClaudeCode {
    if (Get-ApplicationCommand "claude") {
        Write-Host "Claude Code already found on PATH; verifying it."
    }
    else {
        Invoke-DownloadedPowerShellInstaller -Url $ClaudeInstallUrl -Name "Claude Code"
        Add-KnownBinDirectories
    }

    Confirm-Application -CommandName "claude" -DisplayName "Claude Code"
}

function Ensure-Codex {
    if (Get-ApplicationCommand "codex") {
        Write-Host "Codex already found on PATH; verifying it."
    }
    else {
        Invoke-DownloadedPowerShellInstaller -Url $CodexInstallUrl -Name "Codex" -NonInteractive
        Add-KnownBinDirectories
    }

    Confirm-Application -CommandName "codex" -DisplayName "Codex"
}

function Ensure-Pi {
    $script:PiAvailable = $false
    Add-NpmBinDirectories
    $existingPi = Get-ApplicationCommand "pi"
    if ($existingPi -and ($DryRun -or (Test-PiApplication $existingPi))) {
        Write-Host "Pi already found on PATH; verifying it."
    }
    else {
        if ($existingPi) {
            Write-Host "The existing 'pi' command at '$($existingPi.Source)' is not Pi Coding Agent; installing Pi."
        }
        Invoke-DownloadedPowerShellInstaller -Url $PiInstallUrl -Name "Pi"
        Add-NpmBinDirectories

        if (-not $DryRun) {
            $currentPi = Get-ApplicationCommand "pi"
            $unchangedIncompatiblePi = (
                $currentPi -and
                $existingPi -and
                $currentPi.Source -eq $existingPi.Source -and
                -not (Test-PiApplication $currentPi)
            )
            if ((-not $currentPi) -or $unchangedIncompatiblePi) {
                Write-Host "Pi was not installed; continuing without it."
                return
            }
        }
    }

    Confirm-PiApplication
    $script:PiAvailable = $true
}

function Convert-SemanticVersionOutput {
    param([string] $Output)

    if ([string]::IsNullOrWhiteSpace($Output)) {
        return ""
    }
    if ($Output -match '(?m)^\s*(?:(?:uv|opencode|cline|dsh|node)(?:\s+version)?\s+|Hermes Agent\s+v?|v)?(?<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?)(?:\s+\([^\r\n]*\))?\s*$') {
        return $Matches["version"]
    }
    return ""
}

function Test-SupportedStableVersion {
    param(
        [string] $Version,
        [string] $Minimum
    )

    $parsedVersion = Convert-SemanticVersionOutput $Version
    $parsedMinimum = Convert-SemanticVersionOutput $Minimum
    if ([string]::IsNullOrWhiteSpace($parsedVersion) -or [string]::IsNullOrWhiteSpace($parsedMinimum)) {
        throw "Unable to compare semantic versions."
    }
    if ($parsedVersion.Contains("-")) {
        return $false
    }

    $normalizedVersion = $parsedVersion -replace '\+.*$', ''
    $normalizedMinimum = $parsedMinimum -replace '\+.*$', ''
    return ([version] $normalizedVersion) -ge ([version] $normalizedMinimum)
}

function Get-OpenCodeWindowsAssetName {
    $architecture = $env:PROCESSOR_ARCHITEW6432
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        $architecture = $env:PROCESSOR_ARCHITECTURE
    }
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        $architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    }

    switch ($architecture.ToUpperInvariant()) {
        "ARM64" { return "opencode-windows-arm64.zip" }
        "AMD64" { return "opencode-windows-x64-baseline.zip" }
        "X64" { return "opencode-windows-x64-baseline.zip" }
        "X86_64" { return "opencode-windows-x64-baseline.zip" }
        default { throw "OpenCode does not provide a supported Windows release for architecture '$architecture'." }
    }
}

function Install-OpenCode {
    $assetName = Get-OpenCodeWindowsAssetName
    $archiveUrl = "$OpenCodeReleaseBaseUrl/$assetName"
    $installDirectory = Join-Path $env:USERPROFILE ".opencode\bin"
    if ($DryRun) {
        Write-Host "+ irm $archiveUrl -OutFile <temporary-archive>"
        Write-Host "+ extract and install opencode.exe to $(Format-Argument $installDirectory)"
        return
    }

    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("fcc-opencode-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $temporaryRoot $assetName
    $extractPath = Join-Path $temporaryRoot "extracted"
    $temporaryInstallPath = Join-Path $installDirectory (".opencode-" + [guid]::NewGuid().ToString("N") + ".exe")
    try {
        New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
        Write-Host "+ irm $archiveUrl -OutFile $(Format-Argument $archivePath)"
        Invoke-RestMethod -Uri $archiveUrl -OutFile $archivePath -ErrorAction Stop
        if ((-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) -or ((Get-Item -LiteralPath $archivePath).Length -eq 0)) {
            throw "The OpenCode release archive was empty."
        }

        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath
        $executables = @(Get-ChildItem -LiteralPath $extractPath -Recurse -File -Filter "opencode.exe")
        if ($executables.Count -ne 1) {
            throw "The OpenCode release archive did not contain exactly one opencode.exe."
        }

        New-Item -ItemType Directory -Force -Path $installDirectory | Out-Null
        Copy-Item -LiteralPath $executables[0].FullName -Destination $temporaryInstallPath
        if ((-not (Test-Path -LiteralPath $temporaryInstallPath -PathType Leaf)) -or ((Get-Item -LiteralPath $temporaryInstallPath).Length -eq 0)) {
            throw "The extracted OpenCode executable was empty."
        }
        Move-Item -LiteralPath $temporaryInstallPath -Destination (Join-Path $installDirectory "opencode.exe") -Force
    }
    finally {
        Remove-Item -LiteralPath $temporaryInstallPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-OpenCode {
    $command = Get-ApplicationCommand "opencode"
    if ($command) {
        Write-Host "OpenCode already found on PATH; verifying it."
    }
    else {
        Install-OpenCode
        Add-KnownBinDirectories
    }

    Confirm-Application -CommandName "opencode" -DisplayName "OpenCode"
}

function Ensure-Cline {
    Add-NpmBinDirectories

    $command = Get-ApplicationCommand "cline"
    if ($command) {
        Write-Host "Cline already found on PATH; verifying it."
    }
    else {
        $npm = Get-ApplicationCommand "npm"
        if (-not $npm) {
            throw "Cline installation requires npm. Install Node.js from https://nodejs.org/en/download, then rerun the installer."
        }
        Invoke-NativeCommand -FilePath $npm.Source -Arguments @("install", "-g", "cline")
        Add-NpmBinDirectories
    }

    Confirm-Application -CommandName "cline" -DisplayName "Cline"
}

function Confirm-HermesArchitecture {
    $architecture = $env:PROCESSOR_ARCHITEW6432
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        $architecture = $env:PROCESSOR_ARCHITECTURE
    }
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        $architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    }
    if ($architecture.ToUpperInvariant() -notin @("ARM64", "AMD64", "X64", "X86_64")) {
        throw "Hermes Agent does not provide a supported Windows release for architecture '$architecture'."
    }
}

function Install-Hermes {
    Confirm-HermesArchitecture
    Invoke-DownloadedPowerShellInstaller `
        -Url $HermesInstallUrl `
        -Name "Hermes Agent" `
        -ScriptArguments @("-NonInteractive", "-SkipSetup")
    Add-KnownBinDirectories
}

function Ensure-Hermes {
    $command = Get-ApplicationCommand "hermes"
    if ($command) {
        Write-Host "Hermes Agent already found on PATH; verifying it."
    }
    else {
        Install-Hermes
    }

    Confirm-Application -CommandName "hermes" -DisplayName "Hermes Agent"
}

function Install-Grok {
    Invoke-DownloadedPowerShellInstaller -Url $GrokInstallUrl -Name "Grok Build"
    Add-KnownBinDirectories
}

function Ensure-Grok {
    $command = Get-ApplicationCommand "grok"
    if ($command) {
        Write-Host "Grok Build already found on PATH; verifying it."
    }
    else {
        Install-Grok
    }

    Confirm-Application -CommandName "grok" -DisplayName "Grok Build"
}

function Install-Aider {
    $uvPath = "uv"
    if (-not $DryRun) {
        $uvCommand = Get-ApplicationCommand "uv"
        if (-not $uvCommand) {
            throw "Aider installation requires the verified uv command, but it is not available on PATH."
        }
        $uvPath = $uvCommand.Source
    }

    Invoke-NativeCommand -FilePath $uvPath -Arguments @(
        "tool",
        "install",
        "--force",
        "--python",
        "python3.12",
        "--with",
        "pip",
        "aider-chat@latest"
    )
}

function Add-UvToolBinDirectory {
    param([string] $UvPath)

    $toolBin = Invoke-Utf8NativeCapture -FilePath $UvPath -Arguments @("tool", "dir", "--bin")
    if ([string]::IsNullOrWhiteSpace($toolBin)) {
        throw "uv returned an empty tool bin directory."
    }

    Add-PathEntry $toolBin
    return $toolBin
}

function Ensure-Aider {
    $command = Get-ApplicationCommand "aider"
    if ((-not $command) -and (-not $DryRun)) {
        $uvCommand = Get-ApplicationCommand "uv"
        if (-not $uvCommand) {
            throw "Aider installation requires the verified uv command, but it is not available on PATH."
        }
        $null = Add-UvToolBinDirectory -UvPath $uvCommand.Source
        $command = Get-ApplicationCommand "aider"
    }

    if ($command) {
        Write-Host "Aider already found on PATH; verifying it."
    }
    else {
        Install-Aider
    }

    Confirm-Application -CommandName "aider" -DisplayName "Aider"
}

function Get-MusePathWithoutEntry {
    param(
        [AllowEmptyString()][string] $PathValue,
        [string] $Entry
    )

    return @(
        $PathValue.Split(';') | Where-Object {
            $_.Trim().TrimEnd('\', '/') -ine $Entry.TrimEnd('\', '/')
        }
    ) -join ';'
}

function Get-MuseUserPathValue {
    return [Environment]::GetEnvironmentVariable("Path", "User")
}

function Set-MuseUserPathValue {
    param([AllowEmptyString()][string] $Value)

    [Environment]::SetEnvironmentVariable("Path", $Value, "User")
}

function Remove-LegacyMuseInstallation {
    $root = Join-Path $env:LOCALAPPDATA "Programs\Muse Code"
    if (-not (Test-Path -LiteralPath $root)) {
        return $false
    }
    $bin = Join-Path $root "bin"
    $recordPath = Join-Path $root ".fcc-muse-install.json"
    $owned = $false
    try {
        $content = Get-Content -LiteralPath $recordPath -Raw
        $record = $content | ConvertFrom-Json
        $owned = (
            $content.TrimStart().StartsWith('{') -and
            $record -is [pscustomobject] -and
            $record.owner -is [string] -and
            $record.owner -ceq $MuseLegacyOwner -and
            ($record.schema_version -is [int] -or $record.schema_version -is [long]) -and
            $record.schema_version -eq $MuseLegacyOwnerSchemaVersion
        )
    }
    catch {
        # Missing or malformed ownership evidence never authorizes deletion.
    }
    if (-not $owned) {
        Write-Host "FCC could not prove ownership of '$root'; it was preserved."
        return $false
    }
    if ($DryRun) {
        Write-Host "+ remove FCC's legacy Muse Code files and PATH entry from '$root'"
        return $true
    }

    if (Test-Path -LiteralPath $bin -PathType Container) {
        Get-ChildItem -LiteralPath $bin -Force -File | Where-Object {
            $_.Name -ieq 'muse.exe' -or
            $_.Name -cmatch '^\.muse-[0-9a-f]{32}\.(?:staging|backup)\.exe$'
        } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    }
    Get-ChildItem -LiteralPath $root -Force -File | Where-Object {
        $_.Name -cmatch '^\.fcc-muse-install\.json\.[0-9a-f]{32}\.(?:tmp|backup)$'
    } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }

    $env:Path = Get-MusePathWithoutEntry -PathValue ([string] $env:Path) -Entry $bin
    if ([string]::IsNullOrWhiteSpace($env:MUSE_NO_MODIFY_PATH)) {
        $userPath = [string] (Get-MuseUserPathValue)
        $withoutLegacy = Get-MusePathWithoutEntry -PathValue $userPath -Entry $bin
        if ($userPath -cne $withoutLegacy) {
            Set-MuseUserPathValue -Value $withoutLegacy
        }
    }

    Remove-Item -LiteralPath $recordPath -Force
    foreach ($directory in @($bin, $root)) {
        if (
            (Test-Path -LiteralPath $directory -PathType Container) -and
            @(Get-ChildItem -LiteralPath $directory -Force).Count -eq 0
        ) {
            Remove-Item -LiteralPath $directory -Force
        }
    }
    Write-Host "Removed FCC's legacy Muse Code installation."
    return $true
}

function Confirm-MuseApplication {
    if ($DryRun) {
        Write-Host "+ muse --version (Muse Code 0.2.1 or newer)"
        return
    }
    $command = Get-ApplicationCommand "muse"
    if (-not $command) {
        throw "Muse Code was installed, but 'muse' is not available on PATH."
    }
    $output = Invoke-Utf8NativeCapture -FilePath $command.Source -Arguments @("--version")
    if (
        $output -cnotmatch '(?m)^\s*Muse Code\s+(?<version>\d+\.\d+\.\d+)(?:\s+\([^\r\n]+\))?\s*$' -or
        [version] $Matches['version'] -lt [version] '0.2.1'
    ) {
        throw "The 'muse' command at '$($command.Source)' must be Muse Code 0.2.1 or newer. Update or remove it, then rerun the installer."
    }
}

function Ensure-Muse {
    $script:MuseAvailable = $false
    $userPath = [string] (Get-MuseUserPathValue)
    if (-not [string]::IsNullOrWhiteSpace($userPath)) {
        $env:Path = @($env:Path, $userPath) -join [IO.Path]::PathSeparator
    }
    $removedLegacy = Remove-LegacyMuseInstallation

    if ($DryRun -and $removedLegacy) {
        Write-Host "+ after cleanup: verify any remaining Muse, otherwise run $MuseInstallUrl"
    }
    elseif (Get-ApplicationCommand "muse") {
        Write-Host "Muse Code already found on PATH; verifying it."
    }
    else {
        Invoke-DownloadedPowerShellInstaller -Url $MuseInstallUrl -Name "Muse Code"
        $installDirectory = if ([string]::IsNullOrWhiteSpace($env:MUSE_INSTALL_DIR)) {
            Join-Path $env:LOCALAPPDATA "Programs\muse"
        }
        else {
            $env:MUSE_INSTALL_DIR
        }
        Prioritize-PathEntry $installDirectory
    }

    Confirm-MuseApplication
    $script:MuseAvailable = $true
}

function Get-DshVersion {
    param([string] $DshPath)

    $output = Invoke-Utf8NativeCapture -FilePath $DshPath -Arguments @("--version")
    $version = Convert-SemanticVersionOutput $output
    if ([string]::IsNullOrWhiteSpace($version) -or (-not $version.Contains("-"))) {
        throw "DeepSeek Harness is present, but 'dsh --version' did not return its preview semantic version."
    }
    return $version
}

function Get-DshNodeVersion {
    param([string] $NodePath)

    $output = Invoke-Utf8NativeCapture -FilePath $NodePath -Arguments @("--version")
    $version = Convert-SemanticVersionOutput $output
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "DeepSeek Harness requires a readable Node.js version."
    }
    return $version
}

function Test-DshNodeVersion {
    param([string] $Version)

    try {
        $parsed = [version] (($Version -replace '^v', '') -replace '[-+].*$', '')
    }
    catch {
        return $false
    }
    return (
        (($parsed.Major -eq 22) -and ($parsed.Minor -ge 19)) -or
        ($parsed.Major -ge 24)
    )
}

function Test-DshToolchain {
    $node = Get-ApplicationCommand "node"
    $npm = Get-ApplicationCommand "npm"
    if ((-not $node) -or (-not $npm)) {
        return $false
    }
    try {
        return (Test-DshNodeVersion -Version (Get-DshNodeVersion $node.Source))
    }
    catch {
        return $false
    }
}

function Confirm-DshToolchain {
    $node = Get-ApplicationCommand "node"
    if (-not $node) {
        throw "DeepSeek Harness requires Node.js ^22.19.0 or >=24.0.0 and npm. Install Node.js, then rerun the installer."
    }
    $npm = Get-ApplicationCommand "npm"
    if (-not $npm) {
        throw "DeepSeek Harness requires npm. Install npm, then rerun the installer."
    }
    $version = Get-DshNodeVersion $node.Source
    if (-not (Test-DshNodeVersion $version)) {
        throw "DeepSeek Harness requires Node.js ^22.19.0 or >=24.0.0; found Node.js $version."
    }
    return $npm.Source
}

function Confirm-DshApplication {
    if ($DryRun) {
        Write-Host "+ dsh --version"
        return
    }

    $command = Get-ApplicationCommand "dsh"
    if (-not $command) {
        throw "DeepSeek Harness was installed, but 'dsh' is not available on PATH."
    }
    $version = Get-DshVersion $command.Source
    if ($version -ne $DshVersion) {
        throw "DeepSeek Harness $DshVersion is required; found $version after installation."
    }
    Write-Host "Verified DeepSeek Harness $version."
}

function Install-Dsh {
    $npmPath = Confirm-DshToolchain
    Invoke-NativeCommand -FilePath $npmPath -Arguments @("install", "-g", $DshPackage)
    Add-NpmBinDirectories
}

function Ensure-Dsh {
    Add-NpmBinDirectories

    if ($DryRun) {
        if (Get-ApplicationCommand "dsh") {
            Write-Host "+ dsh --version"
            Write-Host "The exact supported DeepSeek Harness preview will be preserved; another version will be replaced."
        }
        else {
            $node = Get-ApplicationCommand "node"
            $npm = Get-ApplicationCommand "npm"
            if ((-not $node) -or (-not $npm)) {
                throw "DeepSeek Harness requires Node.js ^22.19.0 or >=24.0.0 and npm. Install Node.js, then rerun the installer."
            }
            $npmPath = $npm.Source
            Write-Host "+ $(Format-Command -FilePath $npmPath -Arguments @('install', '-g', $DshPackage))"
        }
        Confirm-DshApplication
        return
    }

    [void] (Confirm-DshToolchain)
    $command = Get-ApplicationCommand "dsh"
    if ($command) {
        $version = Get-DshVersion $command.Source
        if ($version -eq $DshVersion) {
            Write-Host "DeepSeek Harness $version already matches the supported preview; leaving it unchanged."
            return
        }
        Write-Host "DeepSeek Harness $version does not match $DshVersion; replacing it with the supported preview."
    }

    Install-Dsh
    Confirm-DshApplication
}

function Ensure-SelectedCodingAgents {
    if ($script:InstallClaudeCode) {
        Write-Step "Ensuring Claude Code is installed"
        Ensure-ClaudeCode
    }

    if ($script:InstallCodex) {
        Write-Step "Ensuring Codex is installed"
        Ensure-Codex
    }

    if ($script:InstallPi) {
        Write-Step "Checking or installing Pi"
        Ensure-Pi
    }

    if ($script:InstallOpenCode) {
        Write-Step "Ensuring OpenCode is installed"
        Ensure-OpenCode
    }

    if ($script:InstallCline) {
        Write-Step "Ensuring Cline CLI is installed"
        Ensure-Cline
    }

    if ($script:InstallHermes) {
        Write-Step "Ensuring Hermes Agent is installed"
        Ensure-Hermes
    }

    if ($script:InstallDsh) {
        Write-Step "Ensuring DeepSeek Harness is installed"
        Ensure-Dsh
    }

    if ($script:InstallGrok) {
        Write-Step "Ensuring Grok Build is installed"
        Ensure-Grok
    }

    if ($script:InstallMuse) {
        Write-Step "Ensuring Muse Code is installed"
        Ensure-Muse
    }

    if ($script:InstallAider) {
        Write-Step "Ensuring Aider is installed"
        Ensure-Aider
    }

    if ((-not $script:InstallClaudeCode) -and (-not $script:InstallCodex) -and (-not $script:PiAvailable) -and (-not $script:InstallOpenCode) -and (-not $script:InstallCline) -and (-not $script:InstallHermes) -and (-not $script:InstallDsh) -and (-not $script:InstallGrok) -and (-not $script:MuseAvailable) -and (-not $script:InstallAider)) {
        throw "No selected coding agent was installed. Re-run the installer and choose at least one."
    }
}

function Get-UvVersion {
    param([string] $UvPath)

    $output = Invoke-Utf8NativeCapture -FilePath $UvPath -Arguments @("--version")
    $version = Convert-SemanticVersionOutput $output
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "uv is present, but 'uv --version' did not return a valid version."
    }

    return $version
}

function Confirm-Uv {
    if ($DryRun) {
        Write-Host "+ uv --version"
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv was installed, but it is not available on PATH."
    }

    $version = Get-UvVersion $uvCommand.Source
    if (-not (Test-SupportedStableVersion -Version $version -Minimum $MinUvVersion)) {
        throw "Stable uv $MinUvVersion or newer is required; found uv $version after installation."
    }
    Write-Host "Verified uv $version."
}

function Get-UvInstallBinDirectory {
    $forceInstallDirectory = if (-not [string]::IsNullOrWhiteSpace($env:UV_INSTALL_DIR)) {
        $env:UV_INSTALL_DIR
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:UV_UNMANAGED_INSTALL)) {
        $env:UV_UNMANAGED_INSTALL
    }
    else {
        $null
    }

    if (-not [string]::IsNullOrWhiteSpace($forceInstallDirectory)) {
        $cargoHome = if (-not [string]::IsNullOrWhiteSpace($env:CARGO_HOME)) {
            $env:CARGO_HOME
        }
        elseif (-not [string]::IsNullOrWhiteSpace($HOME)) {
            Join-Path $HOME ".cargo"
        }
        else {
            $null
        }
        if ($cargoHome -and $forceInstallDirectory.Replace("\\", "\") -eq $cargoHome) {
            return Join-Path $forceInstallDirectory "bin"
        }
        return $forceInstallDirectory
    }
    if (-not [string]::IsNullOrWhiteSpace($env:XDG_BIN_HOME)) {
        return $env:XDG_BIN_HOME
    }
    if (-not [string]::IsNullOrWhiteSpace($env:XDG_DATA_HOME)) {
        return Join-Path $env:XDG_DATA_HOME "..\bin"
    }
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        return Join-Path $env:USERPROFILE ".local\bin"
    }

    throw "Could not determine where the standalone uv installer places uv."
}

function Ensure-Uv {
    if ($DryRun) {
        if (Get-ApplicationCommand "uv") {
            Write-Host "+ uv --version"
            Write-Host "A compatible existing uv will be left unchanged; an obsolete one will be replaced by the standalone installer."
        }
        else {
            Write-Host "uv is not installed; the current standalone uv would be installed."
            Invoke-DownloadedPowerShellInstaller -Url $UvInstallUrl -Name "uv"
            Confirm-Uv
        }
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if ($uvCommand) {
        $version = Get-UvVersion $uvCommand.Source
        if (Test-SupportedStableVersion -Version $version -Minimum $MinUvVersion) {
            Write-Host "uv $version already satisfies >=$MinUvVersion; leaving it unchanged."
            return
        }
        Write-Host "uv $version does not satisfy stable >=$MinUvVersion; installing the current standalone uv."
    }
    else {
        Write-Host "uv is not installed; installing the current standalone uv."
    }

    Invoke-DownloadedPowerShellInstaller -Url $UvInstallUrl -Name "uv"
    Prioritize-PathEntry (Get-UvInstallBinDirectory)
    Confirm-Uv
}

function Get-PackageSpec {
    $includeNim = $VoiceNim
    $includeLocal = $VoiceLocal

    if ($VoiceAll) {
        $includeNim = $true
        $includeLocal = $true
    }

    if ($includeNim -and $includeLocal) {
        return "free-claude-code[voice,voice_local] @ $RepoArchiveUrl"
    }
    if ($includeNim) {
        return "free-claude-code[voice] @ $RepoArchiveUrl"
    }
    if ($includeLocal) {
        return "free-claude-code[voice_local] @ $RepoArchiveUrl"
    }
    return "free-claude-code @ $RepoArchiveUrl"
}

function Install-FreeClaudeCode {
    Assert-NoFccProcessesRunning
    $packageSpec = Get-PackageSpec
    $arguments = @(
        "tool",
        "install",
        "--force",
        "--refresh-package",
        "free-claude-code",
        "--python",
        $PythonRequest
    )
    if (-not [string]::IsNullOrWhiteSpace($TorchBackend)) {
        $arguments += @("--torch-backend", $TorchBackend)
    }
    $arguments += $packageSpec

    $uvPath = "uv"
    if (-not $DryRun) {
        $uvCommand = Get-ApplicationCommand "uv"
        if (-not $uvCommand) {
            throw "uv is not available for the Free Claude Code installation."
        }
        $uvPath = $uvCommand.Source
    }
    Invoke-NativeCommand -FilePath $uvPath -Arguments $arguments
}

function Export-FccDesktopIcon {
    param(
        [string] $DesktopCommand,
        [string] $IconPath
    )

    $arguments = @("--export-icon", $IconPath)
    $commandText = Format-Command -FilePath $DesktopCommand -Arguments $arguments
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    # PowerShell does not wait when directly invoking a Windows GUI executable.
    $process = Start-Process `
        -FilePath $DesktopCommand `
        -ArgumentList @("--export-icon", ('"' + $IconPath + '"')) `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    try {
        $exitCode = $process.ExitCode
    }
    finally {
        $process.Dispose()
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }
    if (-not (Test-Path -LiteralPath $IconPath -PathType Leaf)) {
        throw "Free Claude Code did not export its Windows app icon to '$IconPath'."
    }
}

function Configure-AndConfirmFreeClaudeCode {
    $iconPath = Join-Path $env:USERPROFILE ".fcc\app-icon.ico"
    if ($DryRun) {
        Write-Host "+ uv tool update-shell"
        Write-Host "+ uv tool dir --bin"
        Write-Host "+ verify fcc-desktop, fcc-server, fcc-claude, fcc-codex, fcc-pi, fcc-opencode, fcc-cline, fcc-hermes, fcc-dsh, fcc-grok, fcc-muse, and fcc-aider in the uv tool bin directory"
        Write-Host "+ fcc-server --version"
        Export-FccDesktopIcon `
            -DesktopCommand "<uv-tool-bin>\fcc-desktop.exe" `
            -IconPath $iconPath
        Install-FccDesktopShortcuts `
            -DesktopCommand "<uv-tool-bin>\fcc-desktop.exe" `
            -IconPath $iconPath
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv is not available for PATH configuration."
    }
    Invoke-NativeCommand -FilePath $uvCommand.Source -Arguments @("tool", "update-shell")
    $toolBin = Add-UvToolBinDirectory -UvPath $uvCommand.Source
    $toolBinPath = ([IO.Path]::GetFullPath($toolBin)).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $installedCommands = @{}
    foreach ($commandName in @("fcc-desktop", "fcc-server", "fcc-claude", "fcc-codex", "fcc-pi", "fcc-opencode", "fcc-cline", "fcc-hermes", "fcc-dsh", "fcc-grok", "fcc-muse", "fcc-aider")) {
        $command = Get-ApplicationCommand $commandName
        if (-not $command) {
            throw "Free Claude Code installation did not create '$commandName'."
        }
        $commandDirectory = ([IO.Path]::GetFullPath((Split-Path -Parent $command.Source))).TrimEnd(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        )
        if (-not $commandDirectory.Equals($toolBinPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "'$commandName' resolved outside the uv tool bin directory: $($command.Source)"
        }
        $installedCommands[$commandName] = $command.Source
    }

    Invoke-NativeCommand -FilePath $installedCommands["fcc-server"] -Arguments @("--version")
    Export-FccDesktopIcon `
        -DesktopCommand $installedCommands["fcc-desktop"] `
        -IconPath $iconPath
    Install-FccDesktopShortcuts `
        -DesktopCommand $installedCommands["fcc-desktop"] `
        -IconPath $iconPath
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

function Install-FccDesktopShortcuts {
    param(
        [string] $DesktopCommand,
        [string] $IconPath
    )

    $shortcutPaths = @(
        (Join-Path $env:USERPROFILE "Desktop\Free Claude Code.lnk"),
        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Free Claude Code.lnk")
    )
    foreach ($shortcutPath in $shortcutPaths) {
        Write-Host "+ create shortcut $(Format-Argument $shortcutPath) -> $(Format-Argument $DesktopCommand)"
    }
    if ($DryRun) {
        return
    }

    $shell = New-Object -ComObject WScript.Shell
    foreach ($shortcutPath in $shortcutPaths) {
        if (Test-Path -LiteralPath $shortcutPath) {
            try {
                $existingShortcut = $shell.CreateShortcut($shortcutPath)
                $isFccShortcut = Test-EquivalentPath -Left $existingShortcut.TargetPath -Right $DesktopCommand
            }
            catch {
                $isFccShortcut = $false
            }
            if (-not $isFccShortcut) {
                Write-Host "A shortcut not managed by Free Claude Code already exists at $shortcutPath; leaving it unchanged."
                continue
            }
        }
        $parent = Split-Path -Parent $shortcutPath
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $DesktopCommand
        $shortcut.WorkingDirectory = $env:USERPROFILE
        $shortcut.IconLocation = "$IconPath,0"
        $shortcut.Description = "Run Free Claude Code in the background"
        $shortcut.Save()
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

if ((-not [string]::IsNullOrWhiteSpace($TorchBackend)) -and (-not ($VoiceLocal -or $VoiceAll))) {
    throw "-TorchBackend requires -VoiceLocal or -VoiceAll."
}

Add-KnownBinDirectories
$script:InstallCline = [bool] ((Get-ApplicationCommand "cline") -or (Get-ApplicationCommand "npm"))
Write-Step "Checking for running Free Claude Code processes"
Assert-NoFccProcessesRunning

if (-not (Test-InteractiveInstaller)) {
    $hasDsh = [bool] (Get-ApplicationCommand "dsh")
    $hasDryRunToolchain = [bool] (
        $DryRun -and
        (Get-ApplicationCommand "node") -and
        (Get-ApplicationCommand "npm")
    )
    $script:InstallDsh = $hasDsh -or $hasDryRunToolchain -or (Test-DshToolchain)
}

if (Test-InteractiveInstaller) {
    Write-Step "Choosing coding agents"
    Select-CodingAgents
}

Write-Step "Ensuring uv $MinUvVersion or newer is installed"
Ensure-Uv

Ensure-SelectedCodingAgents
Configure-RtkForSelectedAgents

Write-Step "Installing or updating Free Claude Code"
Install-FreeClaudeCode

Write-Step "Configuring PATH and verifying Free Claude Code"
Configure-AndConfirmFreeClaudeCode

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No changes were made."
}
else {
    Write-Host "Free Claude Code is installed and verified. Open the Free Claude Code desktop shortcut to run it in the background."
    Write-Host "For terminal use, start the proxy with: fcc-server"
    if ($script:InstallClaudeCode) {
        Write-Host "Run Claude Code with: fcc-claude"
    }
    if ($script:InstallCodex) {
        Write-Host "Run Codex with: fcc-codex"
    }
    if ($script:PiAvailable) {
        Write-Host "Run Pi with: fcc-pi"
    }
    if ($script:InstallOpenCode) {
        Write-Host "Run OpenCode with: fcc-opencode"
    }
    if ($script:InstallCline) {
        Write-Host "Run Cline with: fcc-cline"
    }
    else {
        Write-Host "The fcc-cline wrapper is ready after you install Cline CLI."
    }
    if ($script:InstallHermes) {
        Write-Host "Run Hermes Agent with: fcc-hermes"
    }
    else {
        Write-Host "The fcc-hermes wrapper is ready after you install Hermes Agent."
    }
    if ($script:InstallDsh) {
        Write-Host "Run DeepSeek Harness with: fcc-dsh"
    }
    else {
        Write-Host "The fcc-dsh wrapper is ready after you install DeepSeek Harness $DshVersion."
    }
    if ($script:InstallGrok) {
        Write-Host "Run Grok Build with: fcc-grok"
    }
    else {
        Write-Host "The fcc-grok wrapper is ready after you install Grok Build."
    }
    if ($script:MuseAvailable) {
        Write-Host "Run Muse Code with: fcc-muse"
    }
    else {
        Write-Host "The fcc-muse wrapper is ready after you install Muse Code."
    }
    if ($script:InstallAider) {
        Write-Host "Run Aider with: fcc-aider"
    }
    else {
        Write-Host "The fcc-aider wrapper is ready after you install Aider."
    }
}
