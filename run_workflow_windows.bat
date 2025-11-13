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


:log
set "MSG=%~1"
echo %MSG%
>> "%LOG%" echo %MSG%
exit /b 0
