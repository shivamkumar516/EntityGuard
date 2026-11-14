@echo off
rem EntityGuard launcher for Windows: double-click this file.
rem First run: creates .venv and installs requirements.txt (a few minutes). Later runs start in seconds.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation.
  goto :fail
)

if not exist ".venv\Scripts\python.exe" (
  %PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
  if errorlevel 1 (
    echo EntityGuard needs Python 3.11 or newer. Install it from https://www.python.org/downloads/
    goto :fail
  )
  echo Creating the virtual environment in .venv ...
  %PY% -m venv .venv || goto :fail
)

fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul
if errorlevel 1 (
  echo Installing packages from requirements.txt ...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :fail
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

echo Starting EntityGuard (close this window to stop it) ...
".venv\Scripts\python.exe" app.py --open %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo EntityGuard could not start. See the message above.
pause
exit /b 1
