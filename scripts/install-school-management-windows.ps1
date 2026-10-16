$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "Cet installateur doit être exécuté sur Windows."
}

$sourceDirectory = $PSScriptRoot
$sourceExecutable = Join-Path $sourceDirectory "GestionScolaire.exe"
if (-not (Test-Path $sourceExecutable)) {
    throw "GestionScolaire.exe est introuvable. Extrayez d'abord tout le contenu du paquet ZIP."
}

$installDirectory = Join-Path $env:LOCALAPPDATA "GestionScolaire"
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Gestion scolaire.lnk"
$startMenuDirectory = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$startMenuShortcut = Join-Path $startMenuDirectory "Gestion scolaire.lnk"

try {
    New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
    Get-ChildItem -LiteralPath $sourceDirectory -Force |
        Copy-Item -Destination $installDirectory -Recurse -Force

    $executable = Join-Path $installDirectory "GestionScolaire.exe"
    $shell = New-Object -ComObject WScript.Shell
    foreach ($shortcutPath in @($desktopShortcut, $startMenuShortcut)) {
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $executable
        $shortcut.WorkingDirectory = $installDirectory
        $shortcut.Description = "Gestion locale du complexe scolaire"
        $shortcut.IconLocation = "$executable,0"
        $shortcut.Save()
    }
} catch {
    throw "Installation impossible. Fermez l'application si elle est ouverte et réessayez. Détail : $($_.Exception.Message)"
}

Write-Host "Installation terminée dans : $installDirectory"
Write-Host "Raccourcis créés sur le Bureau et dans le menu Démarrer."
Write-Host "Les données et sauvegardes restent dans %LOCALAPPDATA%\FreeClaudeCode."
Write-Host "L'icône de l'application et des raccourcis est intégrée au paquet."
