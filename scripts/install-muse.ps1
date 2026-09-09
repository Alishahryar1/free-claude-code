[CmdletBinding()]
param(
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$script:MuseStableChannelUrl = "https://api.meta.ai/muse-code/channels/muse-stable"
$script:MuseOwner = "free-claude-code-muse-installer"
$script:MuseOwnerSchemaVersion = 1
$script:MuseMinimumVersion = [version] "0.2.1"

function Show-MuseInstallerUsage {
    @"
Usage: install-muse.ps1 [options]

Installs or updates FCC's managed native-Windows Muse Code executable.
The managed command is placed in %LOCALAPPDATA%\Programs\Muse Code\bin.

Options:
  -DryRun                Print actions without downloading or changing state.
  -Help                  Show this help text.

To remove only this managed Muse installation, run scripts/uninstall-muse.ps1.
"@
}

function Get-MuseArtifactKey {
    param([Parameter(Mandatory = $true)][string] $Architecture)

    switch ($Architecture.Trim().ToUpperInvariant()) {
        "AMD64" { return "x86_windows" }
        "X64" { return "x86_windows" }
        "X86_64" { return "x86_windows" }
        "ARM64" { return "aarch64_windows" }
        default {
            throw "Muse Code does not provide a supported Windows release for architecture '$Architecture'."
        }
    }
}

function Get-MuseNativeArchitecture {
    return [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
}

function Test-MuseSafeReleaseUrl {
    param([Parameter(Mandatory = $true)][string] $Value)

    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -match '[\x00-\x1F\x7F]') {
        return $false
    }

    $uri = $null
    if (-not [Uri]::TryCreate($Value, [UriKind]::Absolute, [ref] $uri)) {
        return $false
    }

    return (
        $uri.Scheme -eq "https" -and
        $uri.DnsSafeHost -eq "lookaside.facebook.com" -and
        [string]::IsNullOrEmpty($uri.UserInfo) -and
        [string]::IsNullOrEmpty($uri.Fragment) -and
        ($uri.IsDefaultPort -or $uri.Port -eq 443)
    )
}

function Assert-MuseStableChannel {
    param([Parameter(Mandatory = $true)] $Channel)

    if ((-not (Test-MuseObjectProperty -Object $Channel -Name "channel")) -or $Channel.channel -ne "muse-stable") {
        throw "Meta's Muse stable channel response did not identify the muse-stable channel."
    }
    if ((-not (Test-MuseObjectProperty -Object $Channel -Name "state")) -or $Channel.state -ne "public") {
        throw "Meta's Muse stable channel is not public."
    }
    if (
        (-not (Test-MuseObjectProperty -Object $Channel -Name "version")) -or
        ([string] $Channel.version) -notmatch '^\d+\.\d+\.\d+-R\d+(?:\.\d+)?$'
    ) {
        throw "Meta's Muse stable channel returned an invalid release version."
    }
    if (
        (-not (Test-MuseObjectProperty -Object $Channel -Name "manifest_url")) -or
        (-not (Test-MuseSafeReleaseUrl -Value ([string] $Channel.manifest_url)))
    ) {
        throw "Meta's Muse manifest URL must use HTTPS on lookaside.facebook.com."
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

function Test-MuseObjectProperty {
    param(
        [Parameter(Mandatory = $true)] $Object,
        [Parameter(Mandatory = $true)][string] $Name
    )

    return $null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name
}

function Resolve-MuseArtifact {
    param(
        [Parameter(Mandatory = $true)][string] $Architecture,
        [Parameter(Mandatory = $true)] $Channel,
        [Parameter(Mandatory = $true)] $Manifest
    )

    $artifactKey = Get-MuseArtifactKey -Architecture $Architecture

    Assert-MuseStableChannel -Channel $Channel

    if (
        (-not (Test-MuseObjectProperty -Object $Manifest -Name "version")) -or
        ([string] $Manifest.version) -ne ([string] $Channel.version)
    ) {
        throw "Meta's Muse release manifest version does not match the stable channel."
    }
    if (
        (-not (Test-MuseObjectProperty -Object $Manifest -Name "checksum_algorithm")) -or
        ([string] $Manifest.checksum_algorithm) -cne "sha256"
    ) {
        throw "Meta's Muse release manifest must use sha256 checksums."
    }
    if (-not (Test-MuseObjectProperty -Object $Manifest -Name "artifacts")) {
        throw "Meta's Muse release manifest did not contain artifacts."
    }

    $artifactProperty = $Manifest.artifacts.PSObject.Properties[$artifactKey]
    if ($null -eq $artifactProperty) {
        throw "Meta's Muse release manifest did not contain the '$artifactKey' artifact."
    }
    $artifact = $artifactProperty.Value
    if (
        (-not (Test-MuseObjectProperty -Object $artifact -Name "url")) -or
        (-not (Test-MuseSafeReleaseUrl -Value ([string] $artifact.url)))
    ) {
        throw "Meta's Muse artifact URL must use HTTPS on lookaside.facebook.com."
    }
    if (
        (-not (Test-MuseObjectProperty -Object $artifact -Name "checksum")) -or
        ([string] $artifact.checksum) -cnotmatch '^[0-9a-f]{64}$'
    ) {
        throw "Meta's Muse artifact checksum must be a lowercase SHA-256 value."
    }
    if (
        (-not (Test-MuseObjectProperty -Object $artifact -Name "size")) -or
        (-not (Test-MusePositiveInteger -Value $artifact.size))
    ) {
        throw "Meta's Muse artifact size must be a positive integer."
    }

    return [pscustomobject] @{
        ReleaseVersion = [string] $Channel.version
        ArtifactKey = $artifactKey
        Url = [string] $artifact.url
        Sha256 = [string] $artifact.checksum
        Size = [long] $artifact.size
    }
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

function Add-MusePathValue {
    param(
        [AllowEmptyString()][string] $PathValue,
        [Parameter(Mandatory = $true)][string] $ManagedBin
    )

    $canonicalManagedBin = Get-MuseCanonicalPath -Path $ManagedBin
    if ($null -eq $canonicalManagedBin) {
        throw "The managed Muse bin directory is invalid."
    }
    if ([string]::IsNullOrEmpty($PathValue)) {
        return $canonicalManagedBin
    }

    $entries = @($PathValue -split [regex]::Escape([string] [IO.Path]::PathSeparator))
    $unrelated = @(
        $entries | Where-Object {
            -not (Test-MuseEquivalentPath -Left ([string] $_) -Right $canonicalManagedBin)
        }
    )
    return (@($canonicalManagedBin) + $unrelated) -join [IO.Path]::PathSeparator
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

function Assert-MuseArtifactIntegrity {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][long] $ExpectedSize,
        [Parameter(Mandatory = $true)][string] $ExpectedSha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "The downloaded Muse artifact is missing."
    }
    $actualSize = (Get-Item -LiteralPath $Path).Length
    if ($actualSize -ne $ExpectedSize) {
        throw "The downloaded Muse artifact size did not match the release manifest."
    }
    $stream = [IO.File]::OpenRead($Path)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $sha256.ComputeHash($stream)
        $actualSha256 = ([BitConverter]::ToString($hashBytes) -replace "-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
    if ($actualSha256 -cne $ExpectedSha256) {
        throw "The downloaded Muse artifact SHA-256 did not match the release manifest."
    }
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

function Get-MuseUserPathValue {
    return [Environment]::GetEnvironmentVariable("Path", "User")
}

function Set-MuseUserPathValue {
    param([AllowEmptyString()][string] $Value)

    [Environment]::SetEnvironmentVariable("Path", $Value, "User")
}

function Resolve-MuseExternalCommand {
    $originalPath = $env:Path
    $userPath = Get-MuseUserPathValue
    try {
        $combined = @(
            $originalPath,
            $userPath
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        $env:Path = $combined -join [IO.Path]::PathSeparator
        $commands = @(Get-Command "muse" -CommandType Application -ErrorAction SilentlyContinue)
        if ($commands.Count -eq 0) {
            return $null
        }
        return $commands[0]
    }
    finally {
        $env:Path = $originalPath
    }
}

function ConvertTo-MuseNativeArgument {
    param([AllowEmptyString()][string] $Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = [Text.StringBuilder]::new()
    [void] $builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ([int] $character -eq 92) {
            $backslashes += 1
            continue
        }
        if ([int] $character -eq 34) {
            for ($index = 0; $index -lt (($backslashes * 2) + 1); $index += 1) {
                [void] $builder.Append([char] 92)
            }
            [void] $builder.Append($character)
            $backslashes = 0
            continue
        }
        for ($index = 0; $index -lt $backslashes; $index += 1) {
            [void] $builder.Append([char] 92)
        }
        $backslashes = 0
        [void] $builder.Append($character)
    }
    for ($index = 0; $index -lt ($backslashes * 2); $index += 1) {
        [void] $builder.Append([char] 92)
    }
    [void] $builder.Append('"')
    return $builder.ToString()
}

function Invoke-MuseNativeProcess {
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [string[]] $Arguments = @(),
        [int] $TimeoutMilliseconds = 5000
    )

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = ($Arguments | ForEach-Object {
        ConvertTo-MuseNativeArgument -Argument ([string] $_)
    }) -join " "
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        try {
            if (-not $process.Start()) {
                return [pscustomobject] @{
                    Started = $false
                    TimedOut = $false
                    ExitCode = $null
                    Output = ""
                }
            }
        }
        catch {
            return [pscustomobject] @{
                Started = $false
                TimedOut = $false
                ExitCode = $null
                Output = ""
            }
        }

        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            try {
                $process.Kill()
                $process.WaitForExit()
            }
            catch {
                # The process may have exited between the timeout and the kill.
            }
            return [pscustomobject] @{
                Started = $true
                TimedOut = $true
                ExitCode = $null
                Output = ""
            }
        }

        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        return [pscustomobject] @{
            Started = $true
            TimedOut = $false
            ExitCode = $process.ExitCode
            Output = (@($stdout, $stderr) | Where-Object {
                -not [string]::IsNullOrWhiteSpace($_)
            }) -join [Environment]::NewLine
        }
    }
    finally {
        $process.Dispose()
    }
}

function Get-MuseVersionProbe {
    param([Parameter(Mandatory = $true)][string] $Path)

    $result = Invoke-MuseNativeProcess -FilePath $Path -Arguments @("--version")
    if (-not $result.Started) {
        return [pscustomobject] @{
            Compatible = $false
            Version = $null
            Reason = "could not be started"
        }
    }
    if ($result.TimedOut) {
        return [pscustomobject] @{
            Compatible = $false
            Version = $null
            Reason = "timed out"
        }
    }
    if ($result.ExitCode -ne 0) {
        return [pscustomobject] @{
            Compatible = $false
            Version = $null
            Reason = "returned exit code $($result.ExitCode)"
        }
    }
    if ($result.Output -notmatch '(?m)^\s*Muse Code\s+(?<version>\d+\.\d+\.\d+)(?:\s+\([^\r\n]+\))?\s*$') {
        return [pscustomobject] @{
            Compatible = $false
            Version = $null
            Reason = "did not identify itself as Muse Code"
        }
    }

    $version = [version] $Matches["version"]
    if ($version -lt $script:MuseMinimumVersion) {
        return [pscustomobject] @{
            Compatible = $false
            Version = $version
            Reason = "is older than Muse Code $($script:MuseMinimumVersion)"
        }
    }
    return [pscustomobject] @{
        Compatible = $true
        Version = $version
        Reason = ""
    }
}

function Assert-CompatibleMuseBinary {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Context
    )

    $probe = Get-MuseVersionProbe -Path $Path
    if (-not $probe.Compatible) {
        throw "$Context at '$Path' $($probe.Reason)."
    }
}

function Get-MuseReleaseArtifact {
    param([Parameter(Mandatory = $true)][string] $Architecture)

    try {
        $channel = Invoke-RestMethod `
            -Uri $script:MuseStableChannelUrl `
            -Method Get `
            -TimeoutSec 30 `
            -ErrorAction Stop
    }
    catch {
        throw "Could not fetch Meta's public Muse stable channel."
    }
    Assert-MuseStableChannel -Channel $channel

    try {
        $manifest = Invoke-RestMethod `
            -Uri ([string] $channel.manifest_url) `
            -Method Get `
            -TimeoutSec 30 `
            -ErrorAction Stop
    }
    catch {
        throw "Could not fetch Meta's public Muse release manifest."
    }
    return Resolve-MuseArtifact `
        -Architecture $Architecture `
        -Channel $channel `
        -Manifest $manifest
}

function Save-MuseArtifact {
    param(
        [Parameter(Mandatory = $true)][string] $Url,
        [Parameter(Mandatory = $true)][string] $Destination
    )

    Write-Host "+ Download Meta's Muse executable to a temporary file"
    try {
        Invoke-WebRequest `
            -Uri $Url `
            -OutFile $Destination `
            -TimeoutSec 600 `
            -UseBasicParsing `
            -ErrorAction Stop
    }
    catch {
        throw "Could not download Meta's public Muse Windows executable."
    }
}

function New-MuseOwnershipRecord {
    param([Parameter(Mandatory = $true)] $Artifact)

    return [ordered] @{
        schema_version = $script:MuseOwnerSchemaVersion
        owner = $script:MuseOwner
        release_version = $Artifact.ReleaseVersion
        artifact_key = $Artifact.ArtifactKey
        sha256 = $Artifact.Sha256
        size = [long] $Artifact.Size
    }
}

function Write-MuseRecordContent {
    param(
        [Parameter(Mandatory = $true)][string] $RecordPath,
        [Parameter(Mandatory = $true)][string] $Content
    )

    $temporaryPath = "$RecordPath.$([guid]::NewGuid().ToString('N')).tmp"
    $backupPath = "$RecordPath.$([guid]::NewGuid().ToString('N')).backup"
    try {
        [IO.File]::WriteAllText(
            $temporaryPath,
            $Content,
            [Text.UTF8Encoding]::new($false)
        )
        if (Test-Path -LiteralPath $RecordPath -PathType Leaf) {
            [IO.File]::Replace($temporaryPath, $RecordPath, $backupPath)
            Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
            [IO.File]::Move($temporaryPath, $RecordPath)
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
}

function Write-MuseOwnershipRecord {
    param(
        [Parameter(Mandatory = $true)][string] $RecordPath,
        [Parameter(Mandatory = $true)] $Artifact
    )

    $content = New-MuseOwnershipRecord -Artifact $Artifact | ConvertTo-Json
    Write-MuseRecordContent -RecordPath $RecordPath -Content $content
}

function Restore-MuseOwnershipRecord {
    param(
        [Parameter(Mandatory = $true)][string] $RecordPath,
        [AllowNull()][string] $PreviousContent
    )

    if ($null -eq $PreviousContent) {
        Remove-Item -LiteralPath $RecordPath -Force -ErrorAction SilentlyContinue
        return
    }
    Write-MuseRecordContent -RecordPath $RecordPath -Content $PreviousContent
}

function Test-MuseRecordMatchesArtifact {
    param(
        [Parameter(Mandatory = $true)] $Record,
        [Parameter(Mandatory = $true)] $Artifact
    )

    return (
        $Record.release_version -eq $Artifact.ReleaseVersion -and
        $Record.artifact_key -eq $Artifact.ArtifactKey -and
        $Record.sha256 -ceq $Artifact.Sha256 -and
        [long] $Record.size -eq [long] $Artifact.Size
    )
}

function Test-MuseManagedArtifactCurrent {
    param(
        [Parameter(Mandatory = $true)] $Paths,
        [Parameter(Mandatory = $true)] $Record,
        [Parameter(Mandatory = $true)] $Artifact
    )

    if (-not (Test-MuseRecordMatchesArtifact -Record $Record -Artifact $Artifact)) {
        return $false
    }
    try {
        Assert-MuseArtifactIntegrity `
            -Path $Paths.Executable `
            -ExpectedSize $Artifact.Size `
            -ExpectedSha256 $Artifact.Sha256
        Assert-CompatibleMuseBinary `
            -Path $Paths.Executable `
            -Context "The managed Muse Code executable"
        return $true
    }
    catch {
        return $false
    }
}

function Publish-MuseExecutable {
    param(
        [Parameter(Mandatory = $true)][string] $CandidatePath,
        [Parameter(Mandatory = $true)][string] $ExecutablePath
    )

    $bin = Split-Path -Parent $ExecutablePath
    $stagingPath = Join-Path $bin (".muse-" + [guid]::NewGuid().ToString("N") + ".staging.exe")
    $backupPath = Join-Path $bin (".muse-" + [guid]::NewGuid().ToString("N") + ".backup.exe")
    $replacementCompleted = $false
    $backupRestored = $false
    try {
        Copy-Item -LiteralPath $CandidatePath -Destination $stagingPath
        if (Test-Path -LiteralPath $ExecutablePath -PathType Leaf) {
            try {
                [IO.File]::Replace($stagingPath, $ExecutablePath, $backupPath)
                $replacementCompleted = $true
            }
            catch {
                $publicationError = $_
                if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
                    try {
                        Remove-Item -LiteralPath $ExecutablePath -Force -ErrorAction SilentlyContinue
                        [IO.File]::Move($backupPath, $ExecutablePath)
                        $backupRestored = $true
                    }
                    catch {
                        throw "Muse Code update failed and the previous executable could not be restored. The backup remains at '$backupPath'."
                    }
                }
                throw $publicationError
            }
        }
        else {
            [IO.File]::Move($stagingPath, $ExecutablePath)
            $replacementCompleted = $true
        }
    }
    finally {
        Remove-Item -LiteralPath $stagingPath -Force -ErrorAction SilentlyContinue
        if ($replacementCompleted -or $backupRestored) {
            Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Update-MusePath {
    param([Parameter(Mandatory = $true)][string] $ManagedBin)

    $currentUserPath = [string] (Get-MuseUserPathValue)
    $newUserPath = Add-MusePathValue `
        -PathValue $currentUserPath `
        -ManagedBin $ManagedBin
    if (-not [string]::Equals($currentUserPath, $newUserPath, [StringComparison]::Ordinal)) {
        Set-MuseUserPathValue -Value $newUserPath
    }
    $env:Path = Add-MusePathValue -PathValue ([string] $env:Path) -ManagedBin $ManagedBin
}

function Remove-EmptyMuseInstallDirectories {
    param([Parameter(Mandatory = $true)] $Paths)

    foreach ($directory in @($Paths.Bin, $Paths.Root)) {
        if (
            (Test-Path -LiteralPath $directory -PathType Container) -and
            @(Get-ChildItem -LiteralPath $directory -Force).Count -eq 0
        ) {
            Remove-Item -LiteralPath $directory -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-MuseTemporaryRoot {
    param([Parameter(Mandatory = $true)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return
    }
    $temporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd("\", "/")
    $resolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd("\", "/")
    $requiredPrefix = $temporaryBase + [IO.Path]::DirectorySeparatorChar
    $leafName = [IO.Path]::GetFileName($resolvedPath)
    if (
        (-not $resolvedPath.StartsWith($requiredPrefix, [StringComparison]::OrdinalIgnoreCase)) -or
        (-not $leafName.StartsWith("fcc-muse-", [StringComparison]::Ordinal))
    ) {
        throw "Refusing to remove an unexpected Muse temporary directory: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

function Invoke-MuseInstaller {
    $architecture = Get-MuseNativeArchitecture
    [void] (Get-MuseArtifactKey -Architecture $architecture)
    $paths = Get-MuseInstallPaths

    if ($DryRun) {
        Write-Host "Dry run: inspect existing Muse ownership and commands."
        Write-Host "+ GET $script:MuseStableChannelUrl"
        Write-Host "+ verify release metadata, exact artifact size, SHA-256, and Muse version"
        Write-Host "+ install to '$($paths.Executable)' and prepend '$($paths.Bin)' to the user PATH"
        Write-Host "Dry run complete. No changes were made."
        return
    }

    if (
        (Test-Path -LiteralPath $paths.Root) -and
        (-not (Test-Path -LiteralPath $paths.Root -PathType Container))
    ) {
        throw "The managed Muse Code install path exists but is not owned by FCC: '$($paths.Root)'. Move or remove it, then rerun the installer."
    }

    $record = Read-MuseOwnershipRecord -RecordPath $paths.Record
    if ($null -eq $record) {
        if (
            (Test-Path -LiteralPath $paths.Root -PathType Container) -and
            @(Get-ChildItem -LiteralPath $paths.Root -Force).Count -gt 0
        ) {
            throw "The managed Muse Code directory exists but is not owned by FCC: '$($paths.Root)'. Move or remove it, then rerun the installer."
        }

        $externalCommand = Resolve-MuseExternalCommand
        if ($null -ne $externalCommand) {
            $externalPath = [string] $externalCommand.Source
            $probe = Get-MuseVersionProbe -Path $externalPath
            if (-not $probe.Compatible) {
                throw "A conflicting 'muse' command at '$externalPath' $($probe.Reason). Update or remove that command, then rerun the installer."
            }
            Write-Host "Compatible Muse Code already found at '$externalPath'; leaving it unchanged."
            return
        }
    }

    $artifact = Get-MuseReleaseArtifact -Architecture $architecture
    if (
        $null -ne $record -and
        (Test-MuseManagedArtifactCurrent -Paths $paths -Record $record -Artifact $artifact)
    ) {
        Update-MusePath -ManagedBin $paths.Bin
        Assert-CompatibleMuseBinary `
            -Path $paths.Executable `
            -Context "The managed Muse Code executable"
        Write-Host "Muse Code $($artifact.ReleaseVersion) is already current and verified."
        return
    }

    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("fcc-muse-" + [guid]::NewGuid().ToString("N"))
    $candidatePath = Join-Path $temporaryRoot "muse.exe"
    try {
        New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
        Save-MuseArtifact -Url $artifact.Url -Destination $candidatePath
        Assert-MuseArtifactIntegrity `
            -Path $candidatePath `
            -ExpectedSize $artifact.Size `
            -ExpectedSha256 $artifact.Sha256
        Assert-CompatibleMuseBinary `
            -Path $candidatePath `
            -Context "The downloaded Muse Code executable"

        New-Item -ItemType Directory -Path $paths.Bin -Force | Out-Null
        $previousRecordContent = if (Test-Path -LiteralPath $paths.Record -PathType Leaf) {
            [IO.File]::ReadAllText($paths.Record)
        }
        else {
            $null
        }
        $recordWritten = $false
        try {
            Write-MuseOwnershipRecord -RecordPath $paths.Record -Artifact $artifact
            $recordWritten = $true
            Publish-MuseExecutable `
                -CandidatePath $candidatePath `
                -ExecutablePath $paths.Executable
        }
        catch {
            $publicationError = $_
            if ($recordWritten) {
                try {
                    Restore-MuseOwnershipRecord `
                        -RecordPath $paths.Record `
                        -PreviousContent $previousRecordContent
                }
                catch {
                    throw "Muse Code publication failed and the previous ownership record could not be restored."
                }
            }
            Remove-EmptyMuseInstallDirectories -Paths $paths
            throw $publicationError
        }

        Update-MusePath -ManagedBin $paths.Bin
        Assert-CompatibleMuseBinary `
            -Path $paths.Executable `
            -Context "The published Muse Code executable"
    }
    finally {
        Remove-MuseTemporaryRoot -Path $temporaryRoot
    }

    Write-Host "Muse Code $($artifact.ReleaseVersion) was installed and verified at '$($paths.Executable)'."
}

if ($MyInvocation.InvocationName -ne ".") {
    if ($Help) {
        Show-MuseInstallerUsage
        return
    }
    if ($RemainingArgs.Count -gt 0) {
        Show-MuseInstallerUsage
        throw "Unknown option: $($RemainingArgs -join ' ')"
    }
    Invoke-MuseInstaller
}
