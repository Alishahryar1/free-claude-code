param(
    [switch] $VoiceLocal,
    [string] $TorchBackend = "",
    [switch] $RtkHeadroom,
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# Select once, then delegate all installation to that release's installer.
$repository = "https://github.com/Alishahryar1/free-claude-code"
$raw = "https://raw.githubusercontent.com/Alishahryar1/free-claude-code"
$release = ""
$options = @{}
if ($VoiceLocal) { $options.VoiceLocal = $true }
if ($TorchBackend) { $options.TorchBackend = $TorchBackend }
if ($RtkHeadroom) { $options.RtkHeadroom = $true }
$retryOptions = @()
if ($VoiceLocal) { $retryOptions += '-VoiceLocal' }
if ($TorchBackend) { $retryOptions += "-TorchBackend '" + $TorchBackend.Replace("'", "''") + "'" }
if ($RtkHeadroom) { $retryOptions += '-RtkHeadroom' }

try {
    if ($Help) {
        Write-Output 'Usage: bootstrap.ps1 [-VoiceLocal] [-TorchBackend VALUE] [-RtkHeadroom] [-DryRun] [-Help]'
        Write-Output 'Selects the latest completed FCC release and runs its installer.'
        return
    }
    if ($RemainingArgs.Count -gt 0) { throw "Unknown option: $($RemainingArgs -join ' ')" }
    if ($TorchBackend -and -not $VoiceLocal) { throw "-TorchBackend requires -VoiceLocal." }
    $requested = [Environment]::GetEnvironmentVariable("FCC_INSTALL_VERSION")
    $stable = '(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'
    if ($requested) {
        if ($requested -cnotmatch ( '\A' + $stable + '\z')) {
            throw "FCC_INSTALL_VERSION must be a canonical major.minor.patch release."
        }
        $release = $requested
    }
    if ($DryRun) {
        $selected = if ($release) { $release } else { '<latest completed release>' }
        Write-Output "Select FCC release: $selected"
        Write-Output "Download $raw/v$selected/scripts/install.ps1 and run with FCC_INSTALL_VERSION set $($retryOptions -join ' ')"
        return
    }
    if (-not $release) {
        $response = Invoke-WebRequest -Uri "$repository/releases/latest" -UseBasicParsing -TimeoutSec 10 -MaximumRedirection 5
        $selectedUrl = if ($response.BaseResponse.PSObject.Properties['ResponseUri']) {
            $response.BaseResponse.ResponseUri.AbsoluteUri
        } else {
            $response.BaseResponse.RequestMessage.RequestUri.AbsoluteUri
        }
        if ($selectedUrl -cnotmatch ('\A' + [regex]::Escape("$repository/releases/tag/v") + '(?<version>' + $stable + ')\z')) {
            throw "GitHub did not return a canonical FCC release URL."
        }
        $release = $Matches.version
    }
    $text = Invoke-RestMethod -Uri "$raw/v$release/scripts/install.ps1" -TimeoutSec 10 -MaximumRedirection 5
    if ($text -isnot [string] -or [string]::IsNullOrWhiteSpace($text)) {
        throw "The installer for FCC $release was empty or invalid."
    }
    $installer = [scriptblock]::Create($text)
    $hadVersion = Test-Path Env:FCC_INSTALL_VERSION
    $previousVersion = [Environment]::GetEnvironmentVariable("FCC_INSTALL_VERSION")
    try {
        $env:FCC_INSTALL_VERSION = $release
        & $installer @options
    } finally {
        if ($hadVersion) { $env:FCC_INSTALL_VERSION = $previousVersion }
        else { Remove-Item Env:FCC_INSTALL_VERSION -ErrorAction SilentlyContinue }
    }
} catch {
    if ($release) {
        Write-Host "Retry FCC ${release}: `$env:FCC_INSTALL_VERSION = '$release'; & ([scriptblock]::Create((irm '$raw/v$release/scripts/install.ps1'))) $($retryOptions -join ' ')"
    } else {
        Write-Host 'Retry the bootstrap command after correcting the error.'
    }
    throw
}
