@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment was not found. Run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u main.py
if errorlevel 1 (
  echo.
  echo The program stopped because of an error. The traceback is above.
  echo Details are also written to logs\app.log
  pause
  exit /b 1
)
exit /b 0
