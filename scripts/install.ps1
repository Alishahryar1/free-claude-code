param(
    [switch] $VoiceLocal,
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

# Windows on ARM emulates x64, whose Python package ecosystem has broader wheel support.
$PythonRequest = "cpython-3.14.7-windows-x86_64-none"
$MinUvVersion = "0.12.13"
$ClaudeInstallUrl = "https://claude.ai/install.ps1"
$CodexInstallUrl = "https://chatgpt.com/codex/install.ps1"
$PiInstallUrl = "https://pi.dev/install.ps1"
$OpenCodeReleaseBaseUrl = "https://opencode.ai/files/bin"
$HermesInstallUrl = "https://hermes-agent.nousresearch.com/install.ps1"
$DshMinimumVersion = "0.2.0-rc.2"
$DshPackage = "@deepseek-ai/dsh@latest"
$GrokInstallUrl = "https://x.ai/cli/install.ps1"
$MuseInstallUrl = "https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install-muse.ps1"
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
$script:InstallLog = ""
$script:InstallLogWarning = $false
$script:InstallSteps = [ordered] @{}
$script:CurrentInstallStep = ""
$script:InstallStepOutcome = "Completed"
$script:InstallerPath = $PSCommandPath
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
    "fcc-doctor",
    "fcc-update",
    "fcc-init",
    "free-claude-code"
)

function Show-Usage {
    @"
Usage: install.ps1 [options]

Installs or updates Free Claude Code and lets you choose which coding agents to install or verify.

Options:
  -VoiceLocal            Install local Whisper voice transcription support.
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

function Write-InstallLog {
    param([string] $Message)

    if (-not $script:InstallLog) { return }
    try {
        [IO.File]::AppendAllText($script:InstallLog, "$(Get-Date -Format o) $Message`n", [Text.UTF8Encoding]::new($false))
    }
    catch {
        $script:InstallLog = ""
        if (-not $script:InstallLogWarning) {
            $script:InstallLogWarning = $true
            Write-Warning "Could not write the installer log. Installation will continue. $($_.Exception.Message)"
        }
    }
}

function Initialize-InstallLog {
    if ($DryRun) { return }
    try {
        $directory = Join-Path $env:USERPROFILE ".fcc\logs"
        [void] [IO.Directory]::CreateDirectory($directory)
        $script:InstallLog = Join-Path $directory ("install-$(Get-Date -Format yyyyMMdd-HHmmss)-$([guid]::NewGuid().ToString('N')).log")
        Write-InstallLog "Starting Free Claude Code installer"
    }
    catch {
        $script:InstallLogWarning = $true
        Write-Warning "Could not create the installer log. Installation will continue. $($_.Exception.Message)"
    }
}

function Complete-InstallStep {
    param([string] $Outcome = $script:InstallStepOutcome)

    if (-not $script:CurrentInstallStep) { return }
    $script:InstallSteps[$script:CurrentInstallStep] = $Outcome
    Write-InstallLog "$Outcome`: $script:CurrentInstallStep"
    $script:CurrentInstallStep = ""
}

function Start-InstallStep {
    param([string] $Name)

    if ($DryRun -or $script:InstallSteps.Count -eq 0) { return }
    Complete-InstallStep
    $script:CurrentInstallStep = $Name
    $script:InstallStepOutcome = "Completed"
    $script:InstallSteps[$Name] = "Running"
    Write-InstallLog "Starting: $Name"
}

function Get-InstallerRetryCommand {
    $options = @()
    if ($VoiceLocal) { $options += "-VoiceLocal" }
    if ($TorchBackend) { $options += @("-TorchBackend", $TorchBackend) }
    if ($script:EnableRtk) { $options += "-Rtk" }
    if ($script:InstallerPath -and (Test-Path -LiteralPath $script:InstallerPath -PathType Leaf)) {
        return "& " + (Format-Command $script:InstallerPath $options)
    }
    return "& ([scriptblock]::Create((irm 'https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install.ps1'))) " + (($options | ForEach-Object { Format-Argument $_ }) -join " ")
}

function Show-InstallReport {
    param([bool] $Succeeded)

    if ($DryRun) { return }
    Write-Host "`nInstallation summary:"
    foreach ($entry in $script:InstallSteps.GetEnumerator()) {
        Write-Host "  $($entry.Value): $($entry.Key)"
    }
    if ($script:InstallLog) { Write-Host "Installer log: $script:InstallLog" }
    if (-not $Succeeded) {
        Write-Host "Completed changes have been kept. Resolve the error, then rerun the installer."
        Write-Host "Retry: $(Get-InstallerRetryCommand)"
    }
}

function Write-WindowsLaunchGuidance {
    param([string] $FilePath, [Exception] $Exception, [string] $Stderr = "")

    $policyFailure = $Stderr -match '\(os error (4551|1260)\)'
    for ($cause = $Exception; $null -ne $cause; $cause = $cause.InnerException) {
        if ($cause -is [ComponentModel.Win32Exception] -and $cause.NativeErrorCode -in @(4551, 1260)) {
            $policyFailure = $true
        }
        # Start-Process on Windows PowerShell drops the native error code.
        foreach ($code in @(4551, 1260)) {
            if ($cause.Message.Contains([ComponentModel.Win32Exception]::new($code).Message)) {
                $policyFailure = $true
            }
        }
    }
    if ($policyFailure) {
        $guidance = @"
Windows application policy blocked '$FilePath' or a child it tried to start.
Inspect recent policy events to identify the blocked file, then ask your administrator to allow the required application:
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id=3077,3089; StartTime=(Get-Date).AddMinutes(-15)} | Format-List TimeCreated,Id,Message
"@
        Write-Host $guidance
        Write-InstallLog $guidance
    }
}

function Test-InteractiveInstaller {
    return (-not [Console]::IsInputRedirected) -and (-not [Console]::IsOutputRedirected)
}

function Read-YesNo {
    param(
        [string] $Prompt,
        [bool] $DefaultYes = $false
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

function Find-InstalledCodingAgent {
    param([string] $CommandName)

    $originalPath = $env:Path
    try {
        if ($CommandName -eq "opencode" -and $script:OriginalOpenCode) {
            return $script:OriginalOpenCode
        }
        if ($CommandName -in @("pi", "cline", "dsh")) {
            try {
                Add-NpmBinDirectories
            }
            catch {
                # An optional lookup must not prevent choosing other harnesses.
            }
        }
        $command = Get-ApplicationCommand $CommandName
        if (-not $command) {
            if ($CommandName -eq "aider") {
                if ($env:UV_TOOL_BIN_DIR) {
                    Add-PathEntry $env:UV_TOOL_BIN_DIR
                }
                elseif ($env:XDG_BIN_HOME) {
                    Add-PathEntry $env:XDG_BIN_HOME
                }
                elseif ($env:XDG_DATA_HOME) {
                    Add-PathEntry (Join-Path $env:XDG_DATA_HOME "..\bin")
                }
                elseif ($env:USERPROFILE) {
                    Add-PathEntry (Join-Path $env:USERPROFILE ".local\bin")
                }
                $command = Get-ApplicationCommand $CommandName
            }
            elseif ($CommandName -eq "muse") {
                $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
                if (-not [string]::IsNullOrWhiteSpace($userPath)) {
                    $env:Path = "$originalPath$([IO.Path]::PathSeparator)$userPath"
                    $command = Get-ApplicationCommand $CommandName
                }
            }
        }
        if ($CommandName -eq "pi" -and $command -and (-not $DryRun)) {
            if (-not (Test-PiApplication $command)) {
                return $null
            }
        }
        return $command
    }
    finally {
        # Environment variables are process-wide, even inside a function.
        $env:Path = $originalPath
    }
}

function Read-CodingAgentSelection {
    param(
        [string] $CommandName,
        [string] $DisplayName,
        [string] $FccCommand
    )

    if (Find-InstalledCodingAgent $CommandName) {
        Write-Host "$DisplayName already installed; will verify."
        return $true
    }
    return Read-YesNo -Prompt "Install $DisplayName for ${FccCommand}?"
}

function Select-CodingAgents {
    while ($true) {
        $script:InstallClaudeCode = Read-CodingAgentSelection claude "Claude Code" fcc-claude
        $script:InstallCodex = Read-CodingAgentSelection codex Codex fcc-codex
        $script:InstallPi = Read-CodingAgentSelection pi Pi fcc-pi
        $script:InstallOpenCode = Read-CodingAgentSelection opencode OpenCode fcc-opencode
        $script:InstallCline = Read-CodingAgentSelection cline "Cline CLI" fcc-cline
        $script:InstallHermes = Read-CodingAgentSelection hermes "Hermes Agent" fcc-hermes
        $script:InstallDsh = Read-CodingAgentSelection dsh "DeepSeek Harness" fcc-dsh
        $script:InstallGrok = Read-CodingAgentSelection grok "Grok Build" fcc-grok
        $script:InstallMuse = Read-CodingAgentSelection muse "Muse Code" fcc-muse
        $script:InstallAider = Read-CodingAgentSelection aider Aider fcc-aider

        if ($script:InstallClaudeCode -or $script:InstallCodex -or $script:InstallPi -or $script:InstallOpenCode -or $script:InstallCline -or $script:InstallHermes -or $script:InstallDsh -or $script:InstallGrok -or $script:InstallMuse -or $script:InstallAider) {
            break
        }
        Write-Host "Select at least one coding agent."
        Write-Host ""
    }

    if (-not $script:EnableRtk) {
        if (Get-ApplicationCommand "rtk") {
            Write-Host "RTK already installed; will verify."
            $script:EnableRtk = $true
        }
        else {
            $script:EnableRtk = Read-YesNo `
                -Prompt "Enable RTK token optimization globally for the selected coding agents?" `
                -DefaultYes $false
        }
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

    $timer = [Diagnostics.Stopwatch]::StartNew()
    Write-InstallLog "+ $commandText"
    try {
        $global:LASTEXITCODE = 0
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
        Write-InstallLog "Exit code $exitCode after $($timer.Elapsed.TotalSeconds.ToString('F1'))s: $commandText"
        if ($exitCode -ne 0) {
            throw "Command failed with exit code ${exitCode}: $commandText"
        }
    }
    catch {
        Write-InstallLog "Command failed: $commandText`n$($_.Exception.Message)"
        Write-WindowsLaunchGuidance -FilePath $FilePath -Exception $_.Exception
        throw
    }
}

function Format-NativeArgument {
    param([string] $Value)

    if ($Value -match '^[^\s"&|<>^()]+$') { return $Value }
    # Windows argv rules: escape quotes and double backslashes before quotes/end.
    return '"' + (($Value -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
}

function Invoke-Utf8NativeCapture {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @(),
        [int] $TimeoutSeconds = 10
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    if ($DryRun) { return "" }
    Write-InstallLog "+ $commandText"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("fcc-probe-" + [guid]::NewGuid().ToString("N"))
    $stdoutPath = Join-Path $temporaryRoot "stdout.txt"
    $stderrPath = Join-Path $temporaryRoot "stderr.txt"
    $stdinPath = Join-Path $temporaryRoot "stdin.txt"
    $process = $null
    $stderr = ""
    $originalOutputEncoding = [Console]::OutputEncoding
    try {
        [void] [IO.Directory]::CreateDirectory($temporaryRoot)
        [IO.File]::WriteAllText($stdinPath, "")
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $startOptions = @{
            FilePath = $FilePath
            WindowStyle = "Hidden"
            PassThru = $true
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
            RedirectStandardInput = $stdinPath
        }
        if ($Arguments.Count -gt 0) {
            $startOptions.ArgumentList = ($Arguments | ForEach-Object { Format-NativeArgument $_ }) -join ' '
        }
        $process = Start-Process @startOptions
        # Windows PowerShell needs the handle retained to report the exit code.
        [void] $process.Handle
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            throw "Command timed out after ${TimeoutSeconds}s: $commandText"
        }
        $output = [IO.File]::ReadAllText($stdoutPath, [Text.Encoding]::UTF8).Trim()
        $stderr = [IO.File]::ReadAllText($stderrPath, [Text.Encoding]::UTF8).Trim()
        if ($process.ExitCode -ne 0) {
            throw "Command failed with exit code $($process.ExitCode): $commandText`n$output`n$stderr"
        }
        return $output
    }
    catch {
        Write-InstallLog "Probe failed: $commandText`n$($_.Exception.Message)"
        Write-WindowsLaunchGuidance -FilePath $FilePath -Exception $_.Exception -Stderr $stderr
        throw
    }
    finally {
        try {
            if ($null -ne $process) {
                if (-not $process.HasExited) {
                    & "$env:SYSTEMROOT\System32\taskkill.exe" /PID $process.Id /T /F *> $null
                    [void] $process.WaitForExit(5000)
                }
                if ($process.HasExited) {
                    Write-InstallLog "Exit code $($process.ExitCode) after $($timer.Elapsed.TotalSeconds.ToString('F1'))s: $commandText"
                }
            }
        }
        catch { Write-InstallLog "Probe cleanup failed: $($_.Exception.Message)" }
        finally {
            if ($null -ne $process) { $process.Dispose() }
            [Console]::OutputEncoding = $originalOutputEncoding
        }
        foreach ($stream in @($stdoutPath, $stderrPath)) {
            if ([IO.File]::Exists($stream)) {
                try {
                    Write-InstallLog "$([IO.Path]::GetFileName($stream)): $([IO.File]::ReadAllText($stream, [Text.Encoding]::UTF8))"
                }
                catch { Write-InstallLog "Could not read probe output: $($_.Exception.Message)" }
            }
        }
        # Regular files let us finish even if a descendant inherited stdout/stderr.
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
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
        Add-PathEntry (Join-Path $env:LOCALAPPDATA "Programs\Muse Code\bin")
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

    $prefix = ""
    try { $prefix = Invoke-Utf8NativeCapture $npm.Source @("prefix", "-g") } catch { }
    if ([string]::IsNullOrWhiteSpace($prefix)) {
        try { $prefix = Invoke-Utf8NativeCapture $npm.Source @("config", "get", "prefix") } catch { }
    }
    if (-not [string]::IsNullOrWhiteSpace($prefix)) {
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

function Invoke-InstallerDownload {
    param([string] $Uri, [string] $OutFile = "", [int] $TimeoutSeconds = 30)

    $options = @{ Uri = $Uri; TimeoutSec = $TimeoutSeconds; ErrorAction = "Stop" }
    if ($OutFile) { $options.OutFile = $OutFile }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        Write-InstallLog "Download attempt $($attempt + 1): $Uri (deadline ${TimeoutSeconds}s)"
        try {
            if ($OutFile -and (Test-Path -LiteralPath $OutFile)) {
                Remove-Item -LiteralPath $OutFile -Force
            }
            $result = Invoke-RestMethod @options
            Write-InstallLog "Download completed after $($timer.Elapsed.TotalSeconds.ToString('F1'))s: $Uri"
            return $result
        }
        catch {
            Write-InstallLog "Download failed after $($timer.Elapsed.TotalSeconds.ToString('F1'))s: $Uri`n$($_.Exception.Message)"
            $transient = $false
            $response = $null
            for ($cause = $_.Exception; $null -ne $cause; $cause = $cause.InnerException) {
                if ($cause.PSObject.Properties["Response"] -and $cause.Response) { $response = $cause.Response }
                if ($cause -is [Net.WebException] -and $cause.Status -in @(
                    "Timeout", "ConnectFailure", "ConnectionClosed", "ReceiveFailure", "SendFailure", "KeepAliveFailure", "NameResolutionFailure"
                )) { $transient = $true }
                if ($cause -is [Threading.Tasks.TaskCanceledException]) { $transient = $true }
                if ($cause -is [Net.Sockets.SocketException] -and $cause.SocketErrorCode -in @(
                    "TimedOut", "ConnectionReset", "ConnectionRefused", "NetworkDown", "NetworkUnreachable", "HostUnreachable", "TryAgain"
                )) { $transient = $true }
            }
            $delay = $attempt + 1
            if ($response) {
                $transient = [int] $response.StatusCode -in @(408, 429, 500, 502, 503, 504)
                $retryAfter = if ($response.Headers -is [Net.WebHeaderCollection]) {
                    $response.Headers["Retry-After"]
                }
                else { [string] $response.Headers.RetryAfter }
                $seconds = 0
                $date = [DateTimeOffset]::MinValue
                if ([int]::TryParse($retryAfter, [ref] $seconds)) {
                    $delay = [Math]::Max($delay, [Math]::Min(30, $seconds))
                }
                elseif ([DateTimeOffset]::TryParse($retryAfter, [ref] $date)) {
                    $delay = [Math]::Max($delay, [Math]::Min(30, [Math]::Ceiling(($date - [DateTimeOffset]::UtcNow).TotalSeconds)))
                }
            }
            if (-not $transient -or $attempt -eq 2) { throw }
            Write-Host "Temporary download failure. Retrying in $delay second(s): $Uri"
            Start-Sleep -Seconds $delay
        }
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
        Invoke-InstallerDownload -Uri $Url -OutFile $temporaryScript
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
    Write-Host (Invoke-Utf8NativeCapture -FilePath $command.Source -Arguments @("--version"))
}

function Test-PiApplication {
    param($Command)

    try {
        $helpOutput = Invoke-Utf8NativeCapture -FilePath $Command.Source -Arguments @("--help")
    }
    catch {
        return $false
    }
    return (
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
    Write-Host (Invoke-Utf8NativeCapture -FilePath $command.Source -Arguments @("--version"))
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
        Invoke-InstallerDownload -Uri $archiveUrl -OutFile $archivePath -TimeoutSeconds 600
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
        if ($Arguments[0] -in @("--version", "gain")) {
            Write-Host (Invoke-Utf8NativeCapture -FilePath $command.Source -Arguments $Arguments)
        }
        else {
            Invoke-NativeCommand -FilePath $command.Source -Arguments $Arguments
        }
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
    if ($script:InstallCline) {
        Write-Host "Optional for each project: cd <project>; `$env:RTK_TELEMETRY_DISABLED='1'; rtk init --agent cline"
    }
}

function Ensure-ClaudeCode {
    if (Get-ApplicationCommand "claude") {
        $script:InstallStepOutcome = "Reused"
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
        $script:InstallStepOutcome = "Reused"
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
        $script:InstallStepOutcome = "Reused"
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
                $script:InstallStepOutcome = "Skipped"
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

function Read-OpenCodeVersionOutput {
    param([string] $OpenCodePath)

    return Invoke-Utf8NativeCapture -FilePath $OpenCodePath -Arguments @("--version")
}

function Get-OpenCodeVersion {
    param([string] $OpenCodePath)

    $output = Read-OpenCodeVersionOutput $OpenCodePath
    if ($output -match '(?m)^\s*(?:opencode(?:\s+version)?\s+)?v?(?<version>\d+\.\d+\.\d+(?:\+[0-9A-Za-z.-]+)?)\s*$') {
        return $Matches["version"]
    }
    return ""
}

function Get-OpenCodeRtkPlugin {
    $pluginPath = Join-Path $env:USERPROFILE ".config\opencode\plugins\rtk.ts"
    try {
        $plugin = Get-Item -LiteralPath $pluginPath -Force -ErrorAction Stop
    }
    catch [System.Management.Automation.ItemNotFoundException] {
        return $null
    }
    if ($plugin.PSIsContainer -or ($plugin.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Disable or migrate the RTK plugin at '$pluginPath' manually, then rerun the installer."
    }
    # These include the backup's parent; a junction must not redirect either move.
    foreach ($relativePath in @(".config", ".config\opencode", ".config\opencode\plugins")) {
        $parent = Get-Item -LiteralPath (Join-Path $env:USERPROFILE $relativePath) -Force -ErrorAction Stop
        if ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "The RTK plugin directory is linked: '$($parent.FullName)'. Disable or migrate it manually, then rerun the installer."
        }
    }
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $hash = [BitConverter]::ToString($sha256.ComputeHash([IO.File]::ReadAllBytes($pluginPath))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
    if ($hash -ne "6530c131946c84892f9522abd68d4e513e1e658d8ddbad1f59388c86ebbcb6bb") {
        throw "The RTK plugin at '$pluginPath' was modified or is unrecognized. Disable or migrate it manually, then rerun the installer."
    }
    return $pluginPath
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

function Assert-NoOpenCodeProcessesRunning {
    if (@(Get-Process -Name opencode,opencode2 -ErrorAction SilentlyContinue).Count -gt 0) {
        throw "Close OpenCode before replacing its executable or RTK plugin, then rerun the installer."
    }
}

function Install-OpenCode {
    $assetName = Get-OpenCodeWindowsAssetName
    $installDirectory = Join-Path $env:USERPROFILE ".opencode\bin"
    if ($DryRun) {
        Write-Host "+ resolve latest stable OpenCode 2 and download $assetName"
        Write-Host "+ extract and install opencode.exe to $(Format-Argument $installDirectory)"
        return
    }

    $release = Invoke-InstallerDownload -Uri "https://opencode.ai/update/api/latest/cli/npm"
    if ($release.version -isnot [string] -or $release.version -notmatch '^2\.\d+\.\d+$') {
        throw "The OpenCode release channel did not return a stable OpenCode 2 version."
    }
    $archiveUrl = "$OpenCodeReleaseBaseUrl/$($release.version)/$assetName"

    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("fcc-opencode-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $temporaryRoot $assetName
    $extractPath = Join-Path $temporaryRoot "extracted"
    $temporaryInstallPath = Join-Path $installDirectory (".opencode-" + [guid]::NewGuid().ToString("N") + ".exe")
    try {
        New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
        Write-Host "+ irm $archiveUrl -OutFile $(Format-Argument $archivePath)"
        Invoke-InstallerDownload -Uri $archiveUrl -OutFile $archivePath -TimeoutSeconds 600
        if ((-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) -or ((Get-Item -LiteralPath $archivePath).Length -eq 0)) {
            throw "The OpenCode release archive was empty."
        }

        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath
        $executables = @(Get-ChildItem -LiteralPath $extractPath -Recurse -File -Filter "opencode.exe")
        if ($executables.Count -ne 1) {
            throw "The OpenCode release archive did not contain exactly one opencode.exe."
        }
        $version = Get-OpenCodeVersion $executables[0].FullName
        if (($version -replace '\+.*$', '') -ne $release.version) {
            throw "The OpenCode archive did not contain the selected stable version $($release.version)."
        }

        New-Item -ItemType Directory -Force -Path $installDirectory | Out-Null
        Copy-Item -LiteralPath $executables[0].FullName -Destination $temporaryInstallPath
        if ((-not (Test-Path -LiteralPath $temporaryInstallPath -PathType Leaf)) -or ((Get-Item -LiteralPath $temporaryInstallPath).Length -eq 0)) {
            throw "The extracted OpenCode executable was empty."
        }
        Assert-NoOpenCodeProcessesRunning
        Move-Item -LiteralPath $temporaryInstallPath -Destination (Join-Path $installDirectory "opencode.exe") -Force
    }
    finally {
        Remove-Item -LiteralPath $temporaryInstallPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-OpenCodeInstallState {
    $command = if ($script:OriginalOpenCode) { $script:OriginalOpenCode } else { Get-ApplicationCommand "opencode" }
    $nativePath = Join-Path $env:USERPROFILE ".opencode\bin\opencode.exe"
    $install = $true
    if ($command) {
        $version = Get-OpenCodeVersion $command.Source
        if ($version -match '^2\.') {
            $install = $false
        }
        elseif ($version -match '^1\.') {
            if (-not (Test-EquivalentPath $command.Source $nativePath)) {
                throw "OpenCode 1 at '$($command.Source)' requires manual migration. Remove it with its package manager (npm: npm uninstall -g opencode-ai), then rerun this installer. See https://opencode.ai/v2/docs/migrate-v1/"
            }
        }
        else {
            throw "OpenCode at '$($command.Source)' is not a recognized stable v1 or v2. Correct that installation, then rerun the installer. See https://opencode.ai/v2/docs/migrate-v1/"
        }
    }
    $pluginPath = Get-OpenCodeRtkPlugin
    if ($install -or $pluginPath) {
        Assert-NoOpenCodeProcessesRunning
    }
    if ($install) {
        foreach ($target in @((Join-Path $env:USERPROFILE ".opencode"), (Split-Path -Parent $nativePath), $nativePath)) {
            $item = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
            if ($null -ne $item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "OpenCode installation path is linked: '$target'. Migrate it manually."
            }
        }
    }
    return [pscustomobject] @{ Install = $install; PluginPath = $pluginPath }
}

function Ensure-OpenCode {
    $command = if ($script:OriginalOpenCode) { $script:OriginalOpenCode } else { Get-ApplicationCommand "opencode" }
    $nativePath = Join-Path $env:USERPROFILE ".opencode\bin\opencode.exe"
    if ($DryRun) {
        Write-Host "+ opencode --version"
        Write-Host "Install stable OpenCode 2 if absent, or migrate v1 at '$nativePath'; external v1 requires manual upgrade."
        Write-Host "Check and back up the recognized old OpenCode RTK plugin if present."
        if (-not $command) { Install-OpenCode }
        return
    }

    $state = Get-OpenCodeInstallState
    $pluginPath = $state.PluginPath
    if ($state.Install) {
        Install-OpenCode
        Add-KnownBinDirectories
        if ((Get-OpenCodeVersion $nativePath) -notmatch '^2\.') {
            throw "The OpenCode installer did not install stable OpenCode 2 at '$nativePath'."
        }
    }
    else {
        if (-not $pluginPath) { $script:InstallStepOutcome = "Reused" }
    }
    # Check the command selected after the installer's PATH additions.
    $command = Get-ApplicationCommand "opencode"
    if (-not $command) { throw "OpenCode is not available on PATH after installation." }
    $version = Get-OpenCodeVersion $command.Source
    if ($version -notmatch '^2\.') {
        throw "OpenCode at '$($command.Source)' is not stable OpenCode 2. Correct PATH, then rerun the installer."
    }
    Write-Host "Verified OpenCode $version at '$($command.Source)'."
    if ($pluginPath) {
        $pluginPath = Get-OpenCodeRtkPlugin
        if (-not $pluginPath) { return }
        $backupPath = Join-Path (Split-Path -Parent (Split-Path -Parent $pluginPath)) ("rtk-v1-" + [guid]::NewGuid().ToString("N") + ".bak")
        Assert-NoOpenCodeProcessesRunning
        Move-Item -LiteralPath $pluginPath -Destination $backupPath -ErrorAction Stop
        Write-Host "OpenCode 2 RTK support is unavailable; the old plugin was saved at '$backupPath'."
    }
}

function Ensure-Cline {
    Add-NpmBinDirectories

    $command = Get-ApplicationCommand "cline"
    if ($command) {
        $script:InstallStepOutcome = "Reused"
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
        -ScriptArguments @("-NonInteractive")
    Add-KnownBinDirectories
}

function Ensure-Hermes {
    $command = Get-ApplicationCommand "hermes"
    if ($command) {
        $script:InstallStepOutcome = "Reused"
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
        $script:InstallStepOutcome = "Reused"
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
        $script:InstallStepOutcome = "Reused"
        Write-Host "Aider already found on PATH; verifying it."
    }
    else {
        Install-Aider
    }

    Confirm-Application -CommandName "aider" -DisplayName "Aider"
}

function Ensure-Muse {
    $script:MuseAvailable = $false
    Invoke-DownloadedPowerShellInstaller -Url $MuseInstallUrl -Name "Muse Code"
    Add-KnownBinDirectories
    $commandName = "muse"
    if (-not $DryRun) {
        $command = Find-InstalledCodingAgent "muse"
        if (-not $command) {
            throw "Muse Code was installed, but 'muse' is not available on PATH."
        }
        $commandName = $command.Source
    }
    Confirm-Application -CommandName $commandName -DisplayName "Muse Code"
    $script:MuseAvailable = $true
}

function Get-DshVersion {
    param([string] $DshPath)

    $output = Invoke-Utf8NativeCapture -FilePath $DshPath -Arguments @("--version")
    if ($output -notmatch '(?m)^\s*(?:dsh\s+)?v?(?<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?)\s*$') {
        throw "DeepSeek Harness is present, but 'dsh --version' did not return a semantic version."
    }
    return $Matches['version']
}

function Test-DshVersion {
    param([string] $Version)

    $versionParts = ($Version -replace '\+.*$', '') -split '-', 2
    $minimumParts = $DshMinimumVersion -split '-', 2
    if ($versionParts[0] -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$') {
        return $false
    }
    $preview = @()
    if ($versionParts.Count -gt 1) {
        $preview = @($versionParts[1] -split '\.')
        foreach ($part in $preview) {
            if ($part -notmatch '^[0-9A-Za-z-]+$' -or $part -match '^0[0-9]+$') { return $false }
        }
    }
    $release = [version] $versionParts[0]
    $minimum = [version] $minimumParts[0]
    if ($release -ne $minimum) { return $release -gt $minimum }
    if ($preview.Count -eq 0) { return $true }
    $floor = @($minimumParts[1] -split '\.')
    for ($index = 0; $index -lt [Math]::Min($preview.Count, $floor.Count); $index++) {
        $part = $preview[$index]
        $base = $floor[$index]
        if ($part -ceq $base) { continue }
        $numeric = $part -match '^[0-9]+$'
        $baseNumeric = $base -match '^[0-9]+$'
        if ($numeric -and $baseNumeric) {
            if ($part.Length -ne $base.Length) { return $part.Length -gt $base.Length }
        }
        elseif ($numeric -ne $baseNumeric) { return $baseNumeric }
        return [string]::CompareOrdinal($part, $base) -gt 0
    }
    return $preview.Count -ge $floor.Count
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
    if (-not (Test-DshVersion $version)) {
        throw "DeepSeek Harness requires >=$DshMinimumVersion; found $version after installation."
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
            Write-Host "DeepSeek Harness >=$DshMinimumVersion will be preserved; an older version will be upgraded to latest."
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
        if (Test-DshVersion $version) {
            $script:InstallStepOutcome = "Reused"
            Write-Host "DeepSeek Harness $version already satisfies >=$DshMinimumVersion; leaving it unchanged."
            return
        }
        Write-Host "DeepSeek Harness requires >=$DshMinimumVersion; upgrading $version to latest."
    }

    Install-Dsh
    Confirm-DshApplication
}

function Confirm-SelectedPrerequisites {
    if ($DryRun) { return }
    if ($script:InstallDsh) { [void] (Confirm-DshToolchain) }
    if ($script:InstallCline -and -not (Get-ApplicationCommand "cline") -and -not (Get-ApplicationCommand "npm")) {
        throw "Cline installation requires npm. Install Node.js from https://nodejs.org/en/download, then rerun the installer."
    }
    if ($script:InstallHermes -and -not (Get-ApplicationCommand "hermes")) { Confirm-HermesArchitecture }
    if ($script:InstallOpenCode) {
        $state = Get-OpenCodeInstallState
        if ($state.Install) { [void] (Get-OpenCodeWindowsAssetName) }
    }
}

function Ensure-SelectedCodingAgents {
    if ($script:InstallClaudeCode) {
        Start-InstallStep "Claude Code"
        Write-Step "Ensuring Claude Code is installed"
        Ensure-ClaudeCode
    }

    if ($script:InstallCodex) {
        Start-InstallStep "Codex"
        Write-Step "Ensuring Codex is installed"
        Ensure-Codex
    }

    if ($script:InstallPi) {
        Start-InstallStep "Pi"
        Write-Step "Checking or installing Pi"
        Ensure-Pi
    }

    if ($script:InstallOpenCode) {
        Start-InstallStep "OpenCode"
        Write-Step "Ensuring OpenCode is installed"
        Ensure-OpenCode
    }

    if ($script:InstallCline) {
        Start-InstallStep "Cline"
        Write-Step "Ensuring Cline CLI is installed"
        Ensure-Cline
    }

    if ($script:InstallHermes) {
        Start-InstallStep "Hermes"
        Write-Step "Ensuring Hermes Agent is installed"
        Ensure-Hermes
    }

    if ($script:InstallDsh) {
        Start-InstallStep "DeepSeek Harness"
        Write-Step "Ensuring DeepSeek Harness is installed"
        Ensure-Dsh
    }

    if ($script:InstallGrok) {
        Start-InstallStep "Grok"
        Write-Step "Ensuring Grok Build is installed"
        Ensure-Grok
    }

    if ($script:InstallMuse) {
        Start-InstallStep "Muse"
        Write-Step "Ensuring Muse Code is installed"
        Ensure-Muse
    }

    if ($script:InstallAider) {
        Start-InstallStep "Aider"
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
            $script:InstallStepOutcome = "Reused"
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
    if ($VoiceLocal) {
        return "free-claude-code[voice_local]"
    }
    return "free-claude-code"
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

    # The probe waits for GUI applications too, with a longer icon-export deadline.
    $null = Invoke-Utf8NativeCapture -FilePath $DesktopCommand -Arguments @("--export-icon", $IconPath) -TimeoutSeconds 30
    if ($DryRun) { return }
    if (-not (Test-Path -LiteralPath $IconPath -PathType Leaf)) {
        throw "Free Claude Code did not export its Windows app icon to '$IconPath'."
    }
}

function Configure-AndConfirmFreeClaudeCode {
    Start-InstallStep "PATH configuration"
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
    Start-InstallStep "FCC verification"
    $toolBinPath = ([IO.Path]::GetFullPath($toolBin)).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $installedCommands = @{}
    foreach ($commandName in @("fcc-desktop", "fcc-server", "fcc-claude", "fcc-codex", "fcc-pi", "fcc-opencode", "fcc-cline", "fcc-hermes", "fcc-dsh", "fcc-grok", "fcc-muse", "fcc-aider", "fcc-doctor", "fcc-update.cmd")) {
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

    Write-Host (Invoke-Utf8NativeCapture -FilePath $installedCommands["fcc-server"] -Arguments @("--version"))
    Start-InstallStep "Desktop integration"
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

if ((-not [string]::IsNullOrWhiteSpace($TorchBackend)) -and (-not $VoiceLocal)) {
    throw "-TorchBackend requires -VoiceLocal."
}

Initialize-InstallLog
$script:InstallSteps["Preflight"] = "Not attempted"
$installSucceeded = $false
try {
    Start-InstallStep "Preflight"
    # Preserve the user's winning command before adding installer search paths.
    $script:OriginalOpenCode = Get-ApplicationCommand "opencode"
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

    $script:InstallSteps["uv"] = "Not attempted"
    foreach ($choice in @(
        @("Claude Code", $script:InstallClaudeCode), @("Codex", $script:InstallCodex),
        @("Pi", $script:InstallPi), @("OpenCode", $script:InstallOpenCode),
        @("Cline", $script:InstallCline), @("Hermes", $script:InstallHermes),
        @("DeepSeek Harness", $script:InstallDsh), @("Grok", $script:InstallGrok),
        @("Muse", $script:InstallMuse), @("Aider", $script:InstallAider)
    )) {
        $script:InstallSteps[$choice[0]] = if ($choice[1]) { "Not attempted" } else { "Skipped" }
    }
    foreach ($name in @("FCC package", "PATH configuration", "FCC verification", "Desktop integration")) {
        $script:InstallSteps[$name] = "Not attempted"
    }
    $script:InstallSteps["RTK configuration"] = if ($script:EnableRtk) { "Not attempted" } else { "Skipped" }

    Confirm-SelectedPrerequisites

    Start-InstallStep "uv"
    Write-Step "Ensuring uv $MinUvVersion or newer is installed"
    Ensure-Uv

    Ensure-SelectedCodingAgents

    Start-InstallStep "FCC package"
    Write-Step "Installing or updating Free Claude Code"
    Install-FreeClaudeCode

    Write-Step "Configuring PATH and verifying Free Claude Code"
    Configure-AndConfirmFreeClaudeCode

    if ($script:EnableRtk) {
        Start-InstallStep "RTK configuration"
        Configure-RtkForSelectedAgents
    }
    Complete-InstallStep
    $installSucceeded = $true
}
catch {
    Write-InstallLog $_.Exception.Message
    throw
}
finally {
    if (-not $installSucceeded) { Complete-InstallStep "Failed" }
    Show-InstallReport -Succeeded $installSucceeded
}

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
        Write-Host "The fcc-dsh wrapper is ready after you install DeepSeek Harness >=$DshMinimumVersion."
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
