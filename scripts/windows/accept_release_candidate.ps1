param(
    [Parameter(Mandatory)]
    [ValidatePattern('\Av(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}\z')]
    [string] $Tag,
    [switch] $Update
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Run on a disposable Windows 11 test system. This exercises the real installer.
$policy = Get-ItemPropertyValue -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState
if ($policy -ne 1) { throw 'Smart App Control must be set to On.' }

$repo = 'Alishahryar1/free-claude-code'
$version = $Tag.Substring(1)
$name = "free_claude_code-$version-windows-x64.zip"
$script:FccAcceptanceReleaseApi = "https://api.github.com/repos/$repo/releases/tags/$Tag"
$script:FccAcceptanceDownload = "https://github.com/$repo/releases/download/$Tag/$name"
$metadata = gh release view $Tag --repo $repo --json tagName,isDraft,isPrerelease,assets | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $metadata.isDraft -or $metadata.isPrerelease -or $metadata.tagName -cne $Tag) {
    throw "Expected a draft release for $Tag."
}
$asset = @($metadata.assets | Where-Object { $_.name -ceq $name -and $_.state -eq 'uploaded' })
if ($asset.Count -ne 1) { throw "The signed candidate $name is unavailable." }
$source = Invoke-RestMethod -Uri "https://raw.githubusercontent.com/$repo/$Tag/scripts/install.ps1"
$installer = [scriptblock]::Create($source)

$temp = Join-Path ([IO.Path]::GetTempPath()) ("fcc-candidate-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null
$script:FccAcceptanceArchive = Join-Path $temp $name
$savedVersion = [Environment]::GetEnvironmentVariable('FCC_INSTALL_VERSION')
$savedMarker = [Environment]::GetEnvironmentVariable('_FCC_INSTALLER_RELEASE')
try {
    gh release download $Tag --repo $repo --pattern $name --dir $temp
    if ($LASTEXITCODE -ne 0) { throw "Could not download $name." }
    gh attestation verify $script:FccAcceptanceArchive --repo $repo --signer-workflow "$repo/.github/workflows/post-merge.yml" `
        --source-ref refs/heads/main --deny-self-hosted-runners `
        --predicate-type "https://github.com/$repo/attestations/windows-release/v1"
    if ($LASTEXITCODE -ne 0) { throw 'Could not verify the candidate attestation.' }

    # Feed the authenticated draft asset to the unchanged installer. It still verifies
    # the digest, size, Authenticode signatures, and reported version before activation.
    $script:FccAcceptanceArchiveUsed = $false
    $script:FccAcceptanceMetadata = @{ draft = $false; prerelease = $false; immutable = $true; tag_name = $Tag; assets = $asset }
    function Invoke-RestMethod {
        [CmdletBinding()]
        param([string] $Uri, [string] $OutFile, [int] $TimeoutSec, [int] $MaximumRedirection, [hashtable] $Headers)
        if ($Uri -ceq $script:FccAcceptanceReleaseApi) {
            return $script:FccAcceptanceMetadata
        }
        Microsoft.PowerShell.Utility\Invoke-RestMethod @PSBoundParameters
    }

    function Invoke-WebRequest {
        [CmdletBinding()]
        param([string] $Uri, [string] $OutFile, [int] $TimeoutSec, [int] $MaximumRedirection, [switch] $UseBasicParsing)
        if ($Uri -ceq $script:FccAcceptanceDownload) {
            Copy-Item -LiteralPath $script:FccAcceptanceArchive -Destination $OutFile
            $script:FccAcceptanceArchiveUsed = $true
            return
        }
        Microsoft.PowerShell.Utility\Invoke-WebRequest @PSBoundParameters
    }

    $env:FCC_INSTALL_VERSION = $version
    $env:_FCC_INSTALLER_RELEASE = $version
    & $installer -Update:$Update
    if (-not $script:FccAcceptanceArchiveUsed) {
        throw 'The candidate was not installed. Restore the prior release or rerun without -Update.'
    }
    Write-Host "Candidate digest $($asset[0].digest)"
} finally {
    $env:FCC_INSTALL_VERSION = $savedVersion
    $env:_FCC_INSTALLER_RELEASE = $savedMarker
    Remove-Item -LiteralPath $script:FccAcceptanceArchive -Force -ErrorAction SilentlyContinue
    [IO.Directory]::Delete($temp)
}
