param(
    [string[]] $Only = @(),
    [string[]] $Skip = @(),
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CheckOrder = @(
    "suppressions",
    "ruff-format",
    "ruff-check",
    "ty",
    "pytest",
    "playwright"
)

function Show-Usage {
    @"
Usage: ci.ps1 [options]

Runs the local sequence for the same check IDs enforced by GitHub CI.
Requires uv on PATH when running ruff, ty, pytest, or Playwright checks.
Local ruff checks repair formatting and autofixable lint before later checks.

Checks (in order):
  suppressions   Ban type ignores and legacy future annotations
  ruff-format    uv run ruff format
  ruff-check     uv run ruff check --fix
  ty             uv run ty check
  pytest         uv run pytest -v --tb=short
  playwright     Install Chromium and run deterministic Admin browser tests

Options:
  -Only ID              Run only the given check (repeatable)
  -Skip ID              Skip the given check (repeatable)
  -DryRun               Print commands without running them.
  -Help                 Show this help text.
"@
}

function Write-Step {
    param([string] $Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Format-Argument {
    param([string] $Value)

    if ($Value -match '^[A-Za-z0-9_./:@%+=,\[\]-]+$') {
        return $Value
    }

    return "'" + ($Value -replace "'", "''") + "'"
}

function Invoke-CiCommand {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $parts = @($FilePath) + $Arguments
    $commandText = ($parts | ForEach-Object { Format-Argument ([string] $_) }) -join " "
    Write-Host "+ $commandText"

    if (-not $DryRun) {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $commandText"
        }
    }
}

function Test-ValidCheckId {
    param([string] $CheckId)

    return $CheckOrder -contains $CheckId
}

function Assert-ValidCheckId {
    param([string] $CheckId)

    if (-not (Test-ValidCheckId $CheckId)) {
        throw "unknown check id: $CheckId (expected one of: $($CheckOrder -join ', '))"
    }
}

function Test-ShouldRunCheck {
    param([string] $CheckId)

    if ($Only.Count -gt 0 -and ($Only -notcontains $CheckId)) {
        return $false
    }

    if ($Skip -contains $CheckId) {
        return $false
    }

    return $true
}

function Assert-UvAvailable {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required but was not found on PATH. Install uv first (see README or scripts/install.ps1)."
    }
}

function Test-SelectedChecksNeedUv {
    if ($DryRun) {
        return $false
    }

    foreach ($checkId in $CheckOrder) {
        if ((Test-ShouldRunCheck $checkId) -and $checkId -ne "suppressions") {
            return $true
        }
    }

    return $false
}

function Invoke-SuppressionsCheck {
    Write-Step "Ban suppressions and legacy annotations"
    $pattern = '# type: ignore|# ty: ignore|from __future__ import annotations'
    Write-Host "+ Get-ChildItem -Recurse -Filter *.py (excluding .venv, .git) | Select-String '$pattern'"

    if (-not $DryRun) {
        $matches = Get-ChildItem -Path . -Recurse -Filter *.py -File |
            Where-Object {
                $full = $_.FullName
                $full -notmatch '[\\/]\.venv[\\/]' -and
                    $full -notmatch '[\\/]\.git[\\/]'
            } |
            Select-String -Pattern $pattern

        if ($matches) {
            $matches | ForEach-Object { Write-Host $_.Line }
            throw "type: ignore / ty: ignore comments and legacy future annotations are not allowed. Fix the underlying type/import issue instead."
        }
    }
}

function Invoke-RuffFormatCheck {
    Write-Step "ruff format"
    Invoke-CiCommand -FilePath "uv" -Arguments @("run", "ruff", "format")
}

function Invoke-RuffLintCheck {
    Write-Step "ruff check --fix"
    Invoke-CiCommand -FilePath "uv" -Arguments @("run", "ruff", "check", "--fix")
}

function Invoke-TyCheck {
    Write-Step "ty check"
    Invoke-CiCommand -FilePath "uv" -Arguments @("run", "ty", "check")
}

function Invoke-PytestCheck {
    Write-Step "pytest"
    Invoke-CiCommand -FilePath "uv" -Arguments @("run", "pytest", "-v", "--tb=short")
}

function Invoke-PlaywrightCheck {
    Write-Step "playwright"
    Invoke-CiCommand -FilePath "uv" -Arguments @("run", "playwright", "install", "chromium")
    Invoke-CiCommand -FilePath "uv" -Arguments @(
        "run", "pytest", "e2e", "-v", "--tb=short",
        "--tracing=retain-on-failure", "--screenshot=only-on-failure",
        "--full-page-screenshot", "--output=test-results"
    )
}

function Invoke-Check {
    param([string] $CheckId)

    switch ($CheckId) {
        "suppressions" { Invoke-SuppressionsCheck }
        "ruff-format" { Invoke-RuffFormatCheck }
        "ruff-check" { Invoke-RuffLintCheck }
        "ty" { Invoke-TyCheck }
        "pytest" { Invoke-PytestCheck }
        "playwright" { Invoke-PlaywrightCheck }
        default { throw "unknown check id: $CheckId" }
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

foreach ($checkId in $Only) {
    Assert-ValidCheckId $checkId
}

foreach ($checkId in $Skip) {
    Assert-ValidCheckId $checkId
}

if (Test-SelectedChecksNeedUv) {
    Assert-UvAvailable
}

foreach ($checkId in $CheckOrder) {
    if (Test-ShouldRunCheck $checkId) {
        Invoke-Check $checkId
    }
}

Write-Host ""
Write-Host "All selected CI checks passed."
