@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Fieldnotes.ps1"
if errorlevel 1 (
  echo.
  echo Fieldnotes could not start. See the message above.
  pause
  exit /b 1
)
exit /b 0
