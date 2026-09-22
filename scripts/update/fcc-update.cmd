@echo off
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; & ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install.ps1')))" %*
  call exit /b %%errorlevel%%
)
