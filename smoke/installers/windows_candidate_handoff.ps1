param([switch] $SkipDownload)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ItemPropertyValue {
    param($LiteralPath, $Name)
    return 1
}

function gh {
    $global:LASTEXITCODE = 0
    if ($args[0] -eq 'release' -and $args[1] -eq 'view') {
        return '{"tagName":"v2.0.0","isDraft":true,"isPrerelease":false,"assets":[{"name":"free_claude_code-2.0.0-windows-x64.zip","state":"uploaded","size":3,"digest":"sha256:fixture"}]}'
    }
    if ($args[0] -eq 'release' -and $args[1] -eq 'download') {
        $fixture = Join-Path $args[-1] 'free_claude_code-2.0.0-windows-x64.zip'
        [IO.File]::WriteAllText($fixture, 'zip')
        return
    }
    if ($args[0] -eq 'attestation' -and $args[1] -eq 'verify') {
        if ([IO.File]::ReadAllText($args[2]) -cne 'zip') { throw 'Wrong candidate fixture' }
        return
    }
    throw 'Unexpected GitHub CLI request'
}

function Invoke-RestMethod {
    param([string] $Uri)

    if (-not $Uri.EndsWith('/v2.0.0/scripts/install.ps1')) {
        throw 'Unexpected bootstrap URL'
    }
    return ({
        param([switch] $Update)
        if (-not $Update -or $env:FCC_INSTALL_VERSION -cne '2.0.0' -or $env:_FCC_INSTALLER_RELEASE -cne '2.0.0') {
            throw 'Candidate version or update handoff was incorrect'
        }
        if ($env:FCC_TEST_SKIP_CANDIDATE) { return }
        function Install-FccBundle {
            $metadata = Invoke-RestMethod -Uri 'https://api.github.com/repos/Alishahryar1/free-claude-code/releases/tags/v2.0.0' -TimeoutSec 20
            if ($metadata.draft -or -not $metadata.immutable -or $metadata.assets[0].size -ne 3) {
                throw 'Candidate metadata did not reach the installer'
            }
            # The real installer calls the download override from a scope that also has $archive.
            $archive = [IO.Path]::GetTempFileName()
            try {
                Invoke-WebRequest -Uri 'https://github.com/Alishahryar1/free-claude-code/releases/download/v2.0.0/free_claude_code-2.0.0-windows-x64.zip' -UseBasicParsing -OutFile $archive -TimeoutSec 180
                if ([IO.File]::ReadAllText($archive) -cne 'zip') { throw 'Candidate bytes did not reach the installer' }
            } finally {
                [IO.File]::Delete($archive)
            }
        }
        Install-FccBundle
    }).ToString()
}

$env:FCC_INSTALL_VERSION = 'preserved-fcc-version'
$env:FCC_TEST_SKIP_CANDIDATE = if ($SkipDownload) { '1' } else { '' }
$candidate = Join-Path $PSScriptRoot '../../scripts/windows/accept_release_candidate.ps1'
try {
    & $candidate -Tag v2.0.0 -Update
    if ($SkipDownload) { throw 'The acceptance helper approved a skipped update' }
} catch {
    if (-not $SkipDownload -or $_.Exception.Message -notlike 'The candidate was not installed.*') { throw }
    Write-Output 'The skipped update did not produce an acceptance digest.'
}
if ($env:FCC_INSTALL_VERSION -cne 'preserved-fcc-version') { throw 'The acceptance helper left its version pin in the shell' }
