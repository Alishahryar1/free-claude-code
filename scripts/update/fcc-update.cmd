@echo off
setlocal
set "FCC_UPDATE_CHECKER=%~dp0_fcc-update-check.exe"
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Continue'; $release = & $env:FCC_UPDATE_CHECKER --print-version; $status = $LASTEXITCODE; $ErrorActionPreference = 'Stop'; if ($status -isnot [int]) { throw 'FCC update preflight could not run.' }; if ($status -eq 0) { exit 0 }; if ($status -ne 10) { exit $status }; if ($release -isnot [string] -or $release -notmatch '\A(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\z') { throw 'Invalid FCC update version.' }; $env:FCC_INSTALL_VERSION = $release; & ([scriptblock]::Create((Invoke-RestMethod ('https://raw.githubusercontent.com/Alishahryar1/free-claude-code/v' + $release + '/scripts/install.ps1'))))" %*
  call exit /b %%errorlevel%%
)
