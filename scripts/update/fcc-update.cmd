@echo off
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0fcc-update-windows.ps1" %*
  call exit /b %%errorlevel%%
)
