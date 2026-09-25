@echo off
rem Start SPECTRA on http://127.0.0.1:5000 (Windows).
rem
rem   run.bat           set up .venv if needed, then run the app
rem   run.bat --clean   also delete old uploads and results first
rem
rem The first run needs the internet (pip installs requirements.txt into
rem .venv). Later runs skip pip while requirements.txt is unchanged, so
rem they work offline.
setlocal
cd /d "%~dp0"

set CLEAN=0
:args
if "%~1"=="" goto args_done
if /i "%~1"=="--clean" (
  set CLEAN=1
) else (
  echo Unknown option: %~1 ^(use --clean^)
  exit /b 2
)
shift
goto args
:args_done

if exist ".venv\Scripts\python.exe" goto venv_ready

rem A Python 3.12+: the py launcher first, then python on the PATH.
set PYTHON=
py -3 -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && set PYTHON=py -3
if not defined PYTHON (
  python -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && set PYTHON=python
)
if not defined PYTHON (
  echo SPECTRA needs Python 3.12 or newer. Install it from python.org and try again.
  exit /b 1
)
echo Creating the virtual environment in .venv ...
%PYTHON% -m venv .venv
if errorlevel 1 exit /b 1

:venv_ready
call ".venv\Scripts\activate.bat"

fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if errorlevel 1 (
  echo Installing requirements ...
  python -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 exit /b 1
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

if "%CLEAN%"=="1" python scripts\clean_runs.py

echo.
echo   SPECTRA: open http://127.0.0.1:5000
echo.
python app.py
