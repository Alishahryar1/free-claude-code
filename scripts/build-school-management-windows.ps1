$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "Ce script doit être exécuté sur Windows."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv est requis pour construire l'application. Installez-le depuis https://docs.astral.sh/uv/ puis relancez ce script."
}

uv run --no-project --python 3.14.7 python scripts\build-school-icon.py
if ($LASTEXITCODE -ne 0) {
    throw "La création de l'icône Windows a échoué (code $LASTEXITCODE)."
}

uv run --no-project --python 3.14.7 `
    --with fastapi `
    --with uvicorn `
    --with pyinstaller `
    pyinstaller --noconfirm --clean --distpath dist --workpath build\school-management school-management.spec

if ($LASTEXITCODE -ne 0) {
    throw "La génération de GestionScolaire.exe a échoué (code $LASTEXITCODE)."
}

$executable = Join-Path $repoRoot "dist\GestionScolaire\GestionScolaire.exe"
if (-not (Test-Path $executable)) {
    throw "Le build n'a pas produit le fichier attendu : $executable"
}

$quickStart = Join-Path $repoRoot "dist\GestionScolaire\Lisez-moi-Windows.txt"
Copy-Item (Join-Path $repoRoot "school-management\START-WINDOWS.txt") $quickStart -Force
$installer = Join-Path $repoRoot "dist\GestionScolaire\Installer-GestionScolaire.ps1"
Copy-Item (Join-Path $repoRoot "scripts\install-school-management-windows.ps1") $installer -Force
$archive = Join-Path $repoRoot "dist\GestionScolaire-Windows.zip"
Compress-Archive -Path (Join-Path $repoRoot "dist\GestionScolaire\*") -DestinationPath $archive -Force

Write-Host ""
Write-Host "Application prête : $executable"
Write-Host "Guide de démarrage : $quickStart"
Write-Host "Installateur utilisateur avec raccourcis : $installer"
Write-Host "Archive Windows prête à distribuer : $archive"
Write-Host "Double-cliquez sur GestionScolaire.exe; le navigateur s'ouvrira sur http://127.0.0.1:8765."
