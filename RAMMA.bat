@echo off
setlocal EnableExtensions
title RAMMA - Stem Engine

rem ===========================================================================
rem  RAMMA launcher
rem
rem  Put this next to ramma.py and double-click it.
rem
rem  First run: creates the venv, installs everything from requirements.txt,
rem  and offers to install the CUDA build of PyTorch.
rem  Later runs: checks the venv is intact, then starts the app.
rem
rem  Nothing here needs PowerShell, so the execution-policy error does not
rem  apply. The venv's python.exe is called directly rather than "activated",
rem  which also makes it impossible to start the app on the wrong Python.
rem ===========================================================================

cd /d "%~dp0"

set "APP=ramma.py"
set "VENV=venv"
set "PY=%VENV%\Scripts\python.exe"
set "REQ=requirements.txt"
set "MODELS=models"

rem --- The app itself has to be here ----------------------------------------
if not exist "%APP%" (
    echo [ERROR] %APP% not found in "%CD%".
    echo         Put RAMMA.bat in the same folder as %APP%.
    goto :end
)

rem --- A command-line switch skips straight to a task -----------------------
if /I "%~1"=="setup"   goto :setup
if /I "%~1"=="cuda"    goto :cuda
if /I "%~1"=="check"   goto :check
if /I "%~1"=="update"  goto :update

rem --- No venv yet? Set one up ----------------------------------------------
if not exist "%PY%" (
    echo No virtual environment found - setting one up.
    echo.
    goto :setup
)

goto :run


rem ===========================================================================
:setup
rem ===========================================================================
echo ==========================================================
echo  RAMMA - first-time setup
echo ==========================================================
echo.

rem Find a Python 3.10+ to build the venv with. 3.12 first, then 3.11, 3.10,
rem then whatever "py" and "python" point at. bs-roformer-infer will not
rem install on 3.9 or older.
set "BOOTPY="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined BOOTPY (
        py -%%V -c "import sys" >nul 2>&1 && set "BOOTPY=py -%%V"
    )
)
if not defined BOOTPY (
    py -c "import sys" >nul 2>&1 && set "BOOTPY=py"
)
if not defined BOOTPY (
    python -c "import sys" >nul 2>&1 && set "BOOTPY=python"
)
if not defined BOOTPY (
    echo [ERROR] No Python found. Install Python 3.11 or 3.12 from python.org
    echo         and tick "Add python.exe to PATH" during installation.
    echo         Avoid the Microsoft Store build - it sandboxes file writes.
    goto :end
)

echo Using %BOOTPY% to create the environment...
%BOOTPY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if not "%ERRORLEVEL%"=="0" (
    echo [ERROR] That Python is older than 3.10, which bs-roformer-infer needs.
    echo         Install Python 3.11 or 3.12 and run this again.
    goto :end
)

if not exist "%PY%" (
    %BOOTPY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        goto :end
    )
)

echo.
echo Upgrading pip...
"%PY%" -m pip install --upgrade pip --quiet

echo.
if exist "%REQ%" (
    echo Installing packages from %REQ% - this takes a while the first time.
    "%PY%" -m pip install -r "%REQ%"
) else (
    echo %REQ% not found - installing the package list directly.
    "%PY%" -m pip install customtkinter sounddevice soundfile cffi numpy scipy ^
        torch bs-roformer-infer ml-collections PyYAML pydub librosa
)
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed. Scroll up for the reason.
    goto :end
)

echo.
echo ==========================================================
echo  GPU support
echo ==========================================================
echo The install above gives the CPU build of PyTorch. Separation
echo runs many times faster on an NVIDIA GPU.
echo.
choice /C YN /M "Install the CUDA build now"
if errorlevel 2 goto :check
goto :cuda


rem ===========================================================================
:cuda
rem ===========================================================================
echo.
echo Checking your driver...

rem nvidia-smi is not always on PATH, so look in the usual places too.
set "NVSMI="
for /f "delims=" %%P in ('where nvidia-smi 2^>nul') do if not defined NVSMI set "NVSMI=%%P"
if not defined NVSMI if exist "%SystemRoot%\System32\nvidia-smi.exe" set "NVSMI=%SystemRoot%\System32\nvidia-smi.exe"
if not defined NVSMI if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "NVSMI=%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
if not defined NVSMI if exist "%ProgramW6432%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "NVSMI=%ProgramW6432%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

set "GPUNAME="
set "DRVCUDA="
if defined NVSMI (
    for /f "tokens=*" %%G in ('"%NVSMI%" --query-gpu^=name --format^=csv^,noheader 2^>nul') do if not defined GPUNAME set "GPUNAME=%%G"
    for /f "tokens=2 delims=:" %%C in ('"%NVSMI%" 2^>nul ^| findstr /C:"CUDA Version"') do set "DRVCUDA=%%C"
)

if defined GPUNAME (
    echo   GPU    : %GPUNAME%
) else (
    echo   GPU    : not reported by nvidia-smi
)
if defined DRVCUDA (
    echo   Driver supports CUDA: %DRVCUDA%
)

if not defined NVSMI (
    echo.
    echo   [!] nvidia-smi was not found. That usually means the tool is not on
    echo       PATH rather than that the card is missing, so the choice below
    echo       is still offered. If you do have an NVIDIA GPU, pick the build
    echo       matching your driver; check it in GeForce Experience or with
    echo       Device Manager if you are unsure.
)

echo.
echo Pick a build at or below the CUDA version your driver supports.
echo.
echo   1) cu126  (drivers ~525+)      2) cu128  (~570+)
echo   3) cu130  (~580+)              4) cu132  (newest)
echo   5) skip - stay on CPU
choice /C 12345 /M "Which build"
set "SEL=%ERRORLEVEL%"
set "CUTAG="
if "%SEL%"=="1" set "CUTAG=cu126"
if "%SEL%"=="2" set "CUTAG=cu128"
if "%SEL%"=="3" set "CUTAG=cu130"
if "%SEL%"=="4" set "CUTAG=cu132"
if "%SEL%"=="5" goto :check
if not defined CUTAG (
    echo Nothing selected - staying on CPU.
    goto :check
)

echo.
echo You chose option %SEL% - installing torch for %CUTAG%.
echo This is a large download.
"%PY%" -m pip uninstall -y torch
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/%CUTAG%
if errorlevel 1 (
    echo.
    echo [!] That build could not be installed. Check the list at
    echo     https://pytorch.org and try another one with: RAMMA.bat cuda
)
goto :check


rem ===========================================================================
:update
rem ===========================================================================
echo Reinstalling packages from %REQ%...
"%PY%" -m pip install --upgrade -r "%REQ%"
goto :check


rem ===========================================================================
:check
rem ===========================================================================
echo.
echo ==========================================================
echo  Environment
echo ==========================================================
"%PY%" -c "import sys; print('Python     :', sys.version.split()[0]); print('Executable :', sys.executable)"
"%PY%" -c "import torch; print('PyTorch    :', torch.__version__); print('CUDA build :', torch.version.cuda); print('GPU usable :', torch.cuda.is_available()); print('GPU        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>nul
if errorlevel 1 echo PyTorch    : NOT INSTALLED
"%PY%" -c "import bs_roformer, os; print('bs_roformer:', os.path.dirname(bs_roformer.__file__))" 2>nul
if errorlevel 1 echo bs_roformer: NOT INSTALLED
"%PY%" -c "import librosa; print('librosa    :', librosa.__version__)" 2>nul
if errorlevel 1 echo librosa    : NOT INSTALLED - Mel-Band (karaoke) models will not load

rem --- mel_band_roformer.py: pip cannot install it, so it is checked here ---
rem Ask Python where the package actually lives rather than assuming the path.
set "BSRDIR="
for /f "delims=" %%P in ('"%PY%" -c "import bs_roformer,os;print(os.path.dirname(bs_roformer.__file__))" 2^>nul') do set "BSRDIR=%%P"
if not defined BSRDIR set "BSRDIR=%CD%\%VENV%\Lib\site-packages\bs_roformer"

call :melcheck
if "%MELOK%"=="1" echo mel_band   : present
if "%MELOK%"=="1" goto :melok

echo mel_band   : MISSING - mel_band_roformer.py - copy to the bs_roformer
echo              folder that's located in venv\Lib\site-packages
echo.
echo              Full path on this machine:
echo                %BSRDIR%
echo.
echo              Take the file from ZFTurbo's Music-Source-Separation-Training
echo              repo (models\bs_roformer\mel_band_roformer.py). Without it the
echo              karaoke model cannot be built, so FRT and BG VOCALS stay empty.
echo.
echo              Press a key to continue when you've copied mel_band_roformer.py
echo              into the bs_roformer folder. You only need to do this once.
pause >nul

call :melcheck
if "%MELOK%"=="1" (
    echo mel_band   : present - thanks, that is sorted
) else (
    echo mel_band   : still missing - RAMMA will run, but the karaoke split
    echo              will be unavailable until the file is in place.
)
goto :melok

:melok

echo.
echo ----------------------------------------------------------
where nvidia-smi >nul 2>&1 && (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul) || echo nvidia-smi  : not on PATH
echo.
echo If "GPU usable" above says False but you have an NVIDIA card,
echo run:  RAMMA.bat cuda
echo.
echo ==========================================================
echo  Models in "%MODELS%"
echo ==========================================================
if not exist "%MODELS%" (
    echo [!] No models folder. Create one and put your .ckpt and .yaml pairs in it:
    echo       BS-Rofo-SW-Fixed.ckpt / .yaml                 six stems
    echo       BS-Roformer-Resurrection-Inst.ckpt / .yaml    instrumental
    echo       mel_band_roformer_karaoke_becruily.ckpt/.yaml lead + backing vocals
) else (
    dir /b "%MODELS%\*.ckpt" 2>nul
    dir /b "%MODELS%\*.yaml" 2>nul
    dir /b "%MODELS%\*.ckpt" >nul 2>&1
    if errorlevel 1 echo [!] No .ckpt files found - the six-stem model will download itself,
    if errorlevel 1 echo     but the instrumental and karaoke cells need local files.
)

if /I "%~1"=="check" goto :end
echo.
pause


rem ===========================================================================
:run
rem ===========================================================================
echo.
echo Starting RAMMA...
echo.
"%PY%" "%APP%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo ==========================================================
    echo  RAMMA exited with code %RC%
    echo ==========================================================
    echo  Scroll up for the traceback. Useful commands:
    echo    RAMMA.bat check    show the environment and model files
    echo    RAMMA.bat update   reinstall the packages
    echo    RAMMA.bat cuda     install or change the CUDA build
    echo    RAMMA.bat setup    rebuild the environment from scratch
    echo.
    pause
)

goto :end


:melcheck
rem Sets MELOK=1 when bs_roformer.mel_band_roformer can be imported.
set "MELOK=0"
"%PY%" -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('bs_roformer.mel_band_roformer') else 1)" 2>nul
if not errorlevel 1 set "MELOK=1"
goto :eof


:end
endlocal
