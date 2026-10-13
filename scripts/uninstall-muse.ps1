[CmdletBinding()]
param(
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:MuseOwner = "free-claude-code-muse-installer"
$script:MuseOwnerSchemaVersion = 1

function Show-MuseUninstallerUsage {
    @"
Usage: uninstall-muse.ps1 [options]

Removes only the native-Windows Muse Code executable managed by FCC.
Muse settings, credentials, sessions, logs, caches, and external installations are preserved.

Options:
  -DryRun                Print actions without changing state.
  -Help                  Show this help text.
"@
}

function Get-MuseInstallPaths {
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        throw "LOCALAPPDATA is not set; cannot locate the managed Muse Code installation."
    }

    $root = Join-Path $env:LOCALAPPDATA "Programs\Muse Code"
    $bin = Join-Path $root "bin"
    return [pscustomobject] @{
        Root = $root
        Bin = $bin
        Executable = Join-Path $bin "muse.exe"
        Record = Join-Path $root ".fcc-muse-install.json"
    }
}

function Test-MusePositiveInteger {
    param($Value)

    $integerTypes = @(
        [byte],
        [sbyte],
        [int16],
        [uint16],
        [int32],
        [uint32],
        [int64],
        [uint64]
    )
    foreach ($integerType in $integerTypes) {
        if ($Value -is $integerType) {
            return ([decimal] $Value) -gt 0
        }
    }
    return $false
}

function Read-MuseOwnershipRecord {
    param([Parameter(Mandatory = $true)][string] $RecordPath)

    if (-not (Test-Path -LiteralPath $RecordPath -PathType Leaf)) {
        return $null
    }

    try {
        $record = Get-Content -LiteralPath $RecordPath -Raw | ConvertFrom-Json
    }
    catch {
        throw "The Muse ownership record is malformed; refusing to change the managed directory."
    }

    $expectedFields = @(
        "schema_version",
        "owner",
        "release_version",
        "artifact_key",
        "sha256",
        "size"
    )
    $actualFields = @($record.PSObject.Properties.Name)
    if (
        $actualFields.Count -ne $expectedFields.Count -or
        @($expectedFields | Where-Object { $_ -notin $actualFields }).Count -gt 0 -or
        $record.schema_version -ne $script:MuseOwnerSchemaVersion -or
        $record.owner -ne $script:MuseOwner -or
        ([string] $record.release_version) -notmatch '^\d+\.\d+\.\d+-R\d+(?:\.\d+)?$' -or
        $record.artifact_key -notin @("x86_windows", "aarch64_windows") -or
        ([string] $record.sha256) -cnotmatch '^[0-9a-f]{64}$' -or
        (-not (Test-MusePositiveInteger -Value $record.size))
    ) {
        throw "The Muse ownership record is unsupported or foreign; refusing to change the managed directory."
    }

    return $record
}

function Get-MuseCanonicalPath {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string] $Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    try {
        $fullPath = [IO.Path]::GetFullPath($Path.Trim())
        $root = [IO.Path]::GetPathRoot($fullPath)
        while (
            $fullPath.Length -gt $root.Length -and
            ($fullPath.EndsWith("\") -or $fullPath.EndsWith("/"))
        ) {
            $fullPath = $fullPath.Substring(0, $fullPath.Length - 1)
        }
        return $fullPath
    }
    catch {
        return $null
    }
}

function Test-MuseEquivalentPath {
    param(
        [string] $Left,
        [string] $Right
    )

    $canonicalLeft = Get-MuseCanonicalPath -Path $Left
    $canonicalRight = Get-MuseCanonicalPath -Path $Right
    return (
        $null -ne $canonicalLeft -and
        $null -ne $canonicalRight -and
        [string]::Equals(
            $canonicalLeft,
            $canonicalRight,
            [StringComparison]::OrdinalIgnoreCase
        )
    )
}

function Remove-MusePathValue {
    param(
        [AllowEmptyString()][string] $PathValue,
        [Parameter(Mandatory = $true)][string] $ManagedBin
    )

    if ([string]::IsNullOrEmpty($PathValue)) {
        return ""
    }
    $entries = @($PathValue -split [regex]::Escape([string] [IO.Path]::PathSeparator))
    return @(
        $entries | Where-Object {
            -not (Test-MuseEquivalentPath -Left ([string] $_) -Right $ManagedBin)
        }
    ) -join [IO.Path]::PathSeparator
}

function Test-MusePathContains {
    param(
        [AllowEmptyString()][string] $PathValue,
        [Parameter(Mandatory = $true)][string] $ManagedBin
    )

    if ([string]::IsNullOrEmpty($PathValue)) {
        return $false
    }
    return @(
        $PathValue -split [regex]::Escape([string] [IO.Path]::PathSeparator) |
            Where-Object {
                Test-MuseEquivalentPath -Left ([string] $_) -Right $ManagedBin
            }
    ).Count -gt 0
}

function Get-MuseUserPathValue {
    return [Environment]::GetEnvironmentVariable("Path", "User")
}

function Set-MuseUserPathValue {
    param([AllowEmptyString()][string] $Value)

    [Environment]::SetEnvironmentVariable("Path", $Value, "User")
}

function Remove-EmptyMuseDirectory {
    param([Parameter(Mandatory = $true)][string] $Path)

    if (
        (Test-Path -LiteralPath $Path -PathType Container) -and
        @(Get-ChildItem -LiteralPath $Path -Force).Count -eq 0
    ) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Invoke-MuseUninstaller {
    $paths = Get-MuseInstallPaths
    $rootExists = Test-Path -LiteralPath $paths.Root -PathType Container
    $recordExists = Test-Path -LiteralPath $paths.Record -PathType Leaf

    if (-not $rootExists -and -not $recordExists) {
        Write-Host "No FCC-managed Muse Code installation was found."
        return
    }
    if (-not $recordExists) {
        throw "The Muse Code install root is not owned by FCC; refusing to remove '$($paths.Root)'."
    }

    [void] (Read-MuseOwnershipRecord -RecordPath $paths.Record)

    if ($DryRun) {
        Write-Host "Dry run: remove the FCC-managed executable and installer residue from '$($paths.Bin)'."
        Write-Host "Dry run: remove exact '$($paths.Bin)' entries from the user and current-process PATH."
        Write-Host "Dry run: remove the ownership record '$($paths.Record)' and empty managed directories."
        Write-Host "Dry run complete. No changes were made."
        return
    }

    if (Test-Path -LiteralPath $paths.Executable -PathType Leaf) {
        Remove-Item -LiteralPath $paths.Executable -Force
    }

    if (Test-Path -LiteralPath $paths.Bin -PathType Container) {
        $residue = @(Get-ChildItem -LiteralPath $paths.Bin -Force -File | Where-Object {
            $_.Name -cmatch '^\.muse-[0-9a-f]{32}\.(?:staging|backup)\.exe$'
        })
        foreach ($file in $residue) {
            Remove-Item -LiteralPath $file.FullName -Force
        }
    }

    $recordResidue = @(Get-ChildItem -LiteralPath $paths.Root -Force -File | Where-Object {
        $_.Name -cmatch '^\.fcc-muse-install\.json\.[0-9a-f]{32}\.(?:tmp|backup)$'
    })
    foreach ($file in $recordResidue) {
        Remove-Item -LiteralPath $file.FullName -Force
    }

    if (Test-Path -LiteralPath $paths.Executable -PathType Leaf) {
        throw "The managed Muse Code executable could not be removed."
    }

    $currentUserPath = [string] (Get-MuseUserPathValue)
    $newUserPath = Remove-MusePathValue `
        -PathValue $currentUserPath `
        -ManagedBin $paths.Bin
    if (-not [string]::Equals($currentUserPath, $newUserPath, [StringComparison]::Ordinal)) {
        Set-MuseUserPathValue -Value $newUserPath
    }
    $env:Path = Remove-MusePathValue `
        -PathValue ([string] $env:Path) `
        -ManagedBin $paths.Bin

    if (
        (Test-MusePathContains -PathValue ([string] (Get-MuseUserPathValue)) -ManagedBin $paths.Bin) -or
        (Test-MusePathContains -PathValue ([string] $env:Path) -ManagedBin $paths.Bin)
    ) {
        throw "The managed Muse Code PATH entry could not be removed."
    }

    Remove-Item -LiteralPath $paths.Record -Force
    if (Test-Path -LiteralPath $paths.Record -PathType Leaf) {
        throw "The Muse ownership record could not be removed."
    }

    Remove-EmptyMuseDirectory -Path $paths.Bin
    Remove-EmptyMuseDirectory -Path $paths.Root

    if (Test-Path -LiteralPath $paths.Root -PathType Container) {
        Write-Host "Muse Code was uninstalled; unknown files in '$($paths.Root)' were preserved."
    }
    else {
        Write-Host "The FCC-managed Muse Code installation was removed."
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    if ($Help) {
        Show-MuseUninstallerUsage
        return
    }
    if ($RemainingArgs.Count -gt 0) {
        Show-MuseUninstallerUsage
        throw "Unknown option: $($RemainingArgs -join ' ')"
    }
    Invoke-MuseUninstaller
}
