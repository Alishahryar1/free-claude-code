@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $env:UV_LINK_MODE = 'copy'; $installer = [scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/mridwanfirdaus2603/free-claude-code/main/scripts/install.ps1')); & $installer @args" %*
exit /b %errorlevel%
