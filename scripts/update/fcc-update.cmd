@echo off
setlocal
rem The installer replaces this file, so cmd must parse the whole block first.
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; & ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install.ps1'))) -Update" %*
  call exit /b %%errorlevel%%
)
