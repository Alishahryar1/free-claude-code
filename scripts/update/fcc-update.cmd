@echo off
"%~dp0_fcc-update-check.exe"
if %errorlevel%==0 exit /b 0
if not %errorlevel%==10 exit /b %errorlevel%
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; & ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install.ps1')))" %*
  call exit /b %%errorlevel%%
)
