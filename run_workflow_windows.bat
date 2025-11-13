@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run everything relative to this script's directory
cd /d "%~dp0"

for /f %%i in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%i"
set "LOG=run_log_%TIMESTAMP%.txt"

call :log "NHANES Multimarker Workflow (Windows batch)"

call :find_python
if errorlevel 1 (
  call :log "ERROR: Python 3 interpreter not found on PATH (python3, python, py -3)."
  exit /b 1
)

call :log "Using Python command: %PY_CMD%"

rem Ensure a local virtual environment with required packages
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

call :setup_venv
if errorlevel 1 (
  call :log "ERROR: Failed to set up Python virtual environment."
  exit /b 1
)

rem From this point on, always run inside the venv
set "PY_CMD=%VENV_PY%"
call :log "Activated venv Python: %PY_CMD%"

set "NEED_DOWNLOAD=0"
if not exist "nhanes_data" (
  set "NEED_DOWNLOAD=1"
) else (
  dir /b "nhanes_data" >nul 2>&1 || set "NEED_DOWNLOAD=1"
)

if "%NEED_DOWNLOAD%"=="1" (
  call :run_py "Downloading NHANES source files..." "1_download_data.py"
) else (
  call :log "Data directory detected; skipping download."
)

call :run_py "Merging and preprocessing data..." "2_merge_preprocess_multimarker.py"
call :run_py "Running regression models..." "3_regression_multimarker.py"
call :run_py "Generating figures..." "4_visualize_multimarker.py"

call :log "Done. Review output_data/, output_figures/, and %LOG% for details."
exit /b 0


:find_python
for %%P in (python3 python) do (
  %%P --version >nul 2>&1
  if not errorlevel 1 (
    set "PY_CMD=%%P"
    goto :eof
  )
)
py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=py -3"
  goto :eof
)
exit /b 1


:run_py
set "STEP_MSG=%~1"
set "SCRIPT=%~2"
call :log "%STEP_MSG%"
call %PY_CMD% "%SCRIPT%" >> "%LOG%" 2>&1
if errorlevel 1 (
  call :log "ERROR: %SCRIPT% failed. Check %LOG% for details."
  exit /b 1
)
exit /b 0


:setup_venv
if exist "%VENV_PY%" (
  rem Refresh dependencies to avoid missing modules
  call :log "Ensuring Python deps (pip install -r requirements.txt)…"
  "%VENV_PY%" -m pip install --disable-pip-version-check -q -r requirements.txt >> "%LOG%" 2>&1
  goto :eof
)

call :log "Creating virtual environment at %VENV_DIR%…"
call %PY_CMD% -m venv "%VENV_DIR%" >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

call :log "Bootstrapping pip/setuptools…"
"%VENV_PY%" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

call :log "Installing required packages…"
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1
goto :eof


:log
set "MSG=%~1"
echo %MSG%
>> "%LOG%" echo %MSG%
exit /b 0
