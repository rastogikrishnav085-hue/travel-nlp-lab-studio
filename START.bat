@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Travel NLP Lab Studio

:MENU
cls
echo ============================================================
echo   TRAVEL NLP LAB STUDIO
echo ============================================================
echo.
echo   1  Open Studio                          (start here)
echo   2  Train on my own CSVs, then open      (put CSVs in my_data first)
echo   3  Train on the ATIS sample, then open  (about 30 seconds)
echo   4  Run the self-test
echo   5  Quit
echo.
choice /c 12345 /n /m "Choose 1-5: "
if errorlevel 5 exit /b 0
if errorlevel 4 set "MODE=test" & goto :GO
if errorlevel 3 set "MODE=atis" & goto :GO
if errorlevel 2 set "MODE=mine" & goto :GO
set "MODE=open"

:GO
echo.
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do if not defined PY (py -%%V --version >nul 2>&1 && set "PY=py -%%V")
if not defined PY (python --version >nul 2>&1 && set "PY=python")
if not defined PY goto :NO_PYTHON

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating the private Python environment ^(first time only^)...
    %PY% -m venv .venv
    if errorlevel 1 goto :FAIL
) else (
    echo [1/3] Python environment found.
)
set "VPY=%~dp0.venv\Scripts\python.exe"

set "REQ=requirements.txt"
if "%MODE%"=="mine" set "REQ=requirements-train.txt"
if "%MODE%"=="atis" set "REQ=requirements-train.txt"
if "%MODE%"=="test" set "REQ=requirements-dev.txt"
fc /b "%REQ%" ".venv\%REQ%.installed" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing packages ^(first time only, 2 to 5 minutes, do not close^)...
    "%VPY%" -m pip install --upgrade pip >nul 2>&1
    "%VPY%" -m pip install -r "%REQ%"
    if errorlevel 1 goto :FAIL
    copy /y "%REQ%" ".venv\%REQ%.installed" >nul
) else (
    echo [2/3] Packages already installed.
)

if "%MODE%"=="test" goto :TEST
if "%MODE%"=="atis" (
    echo Training on the ATIS sample...
    "%VPY%" tools\run_notebook_local.py --atis
    if errorlevel 1 goto :FAIL
)
if "%MODE%"=="mine" (
    echo Training on the CSV files in my_data...
    "%VPY%" tools\run_notebook_local.py
    if errorlevel 1 goto :FAIL
)

set "PORT="
for /f %%P in ('powershell -NoProfile -Command "foreach($p in 8501..8510){ if(-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)){ $p; break } }"') do set "PORT=%%P"
if not defined PORT set "PORT=8501"

echo [3/3] Starting Studio...
echo.
echo   Your browser opens by itself. If not, go to:  http://127.0.0.1:%PORT%
echo   To stop Studio: click this window and press Ctrl + C
echo.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:%PORT%'"
"%VPY%" -m streamlit run streamlit_app.py --server.port %PORT% --server.address 127.0.0.1
echo.
echo Studio stopped.
pause
goto :MENU

:TEST
echo [3/3] Running the self-test...
"%VPY%" -m pytest -q
echo.
pause
goto :MENU

:NO_PYTHON
echo ERROR: Python was not found.
echo Install Python 3.12 from https://www.python.org/downloads/
echo On the first installer screen, tick "Add python.exe to PATH", then click Install.
pause
goto :MENU

:FAIL
echo.
echo Something went wrong. Copy the text above and send it to your helper.
pause
goto :MENU
