@echo off
REM ============================================================
REM  Fashion3D - double-click to run (Windows)
REM  First run: creates a Python environment and installs packages.
REM  NVIDIA GPU found -> CUDA PyTorch (~3 GB, 5-15 min), otherwise CPU PyTorch (~200 MB, 2-5 min).
REM  Force one with:  set FASHION3D_DEVICE=cpu   (or gpu)  before running this file.
REM  Then opens http://localhost:8517 in your browser.
REM  Close this window to stop the app.
REM ============================================================
cd /d "%~dp0"

REM The environment lives in a short path: PyTorch has very long file names and Windows
REM refuses paths over 260 characters if the project sits deep inside other folders.
set VENV=%LOCALAPPDATA%\Fashion3D\venv
if exist "%VENV%\installed.ok" goto run

set PY=
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V
    )
)
if not defined PY (
    python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" >nul 2>&1 && set PY=python
)
if not defined PY (
    echo.
    echo  Python 3.10, 3.11 or 3.12 is needed. Get it from https://www.python.org/downloads/
    echo  ^(tick "Add python.exe to PATH" during install^), then run this file again.
    pause
    exit /b 1
)

REM ---- GPU or CPU build of PyTorch? ----
set REQ=requirements.txt
set KIND=CPU
if /i "%FASHION3D_DEVICE%"=="cpu" goto have_req
if /i "%FASHION3D_DEVICE%"=="gpu" goto use_gpu
nvidia-smi -L >nul 2>&1 || goto have_req
:use_gpu
set REQ=requirements-gpu.txt
set KIND=GPU (NVIDIA CUDA)
:have_req

echo Creating the Python environment in %VENV% with %PY% ...
%PY% -m venv "%VENV%" || goto fail
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
echo Installing packages for %KIND% (first run only, this can take several minutes) ...
"%VENV%\Scripts\python.exe" -m pip install -r %REQ%
if errorlevel 1 (
    if "%REQ%"=="requirements.txt" goto fail
    echo.
    echo  The GPU install failed - falling back to the CPU version ...
    "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
    set KIND=CPU
)
echo %KIND%> "%VENV%\installed.ok"

:run
set /p KIND=<"%VENV%\installed.ok"
echo.
echo  Fashion3D is starting on http://localhost:8517  (PyTorch: %KIND%; close this window to stop it)
echo.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8517"
"%VENV%\Scripts\python.exe" -m streamlit run app.py --server.port 8517 --server.headless true
pause
exit /b 0

:fail
echo.
echo  Installation failed - see the messages above.
pause
exit /b 1
