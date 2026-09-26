@echo off
setlocal EnableExtensions
title RAMMA - build start exe

rem ===========================================================================
rem  Turns ramma_start.py into RAMMA.exe: a launcher that simply starts the
rem  app with the venv's Python. Around 10 MB, builds in well under a minute.
rem
rem  Run this once. Afterwards you only need RAMMA.exe.
rem ===========================================================================

cd /d "%~dp0"

if not exist "ramma_start.py" (
    echo [ERROR] ramma_start.py not found in "%CD%".
    goto :end
)

set "BUILDPY="
if exist "venv\Scripts\python.exe" set "BUILDPY=venv\Scripts\python.exe"
if not defined BUILDPY (
    py -c "import sys" >nul 2>&1 && set "BUILDPY=py"
)
if not defined BUILDPY (
    python -c "import sys" >nul 2>&1 && set "BUILDPY=python"
)
if not defined BUILDPY (
    echo [ERROR] No Python found to build with.
    goto :end
)

echo Installing PyInstaller...
%BUILDPY% -m pip install --upgrade pyinstaller --quiet
if not "%ERRORLEVEL%"=="0" (
    echo [ERROR] Could not install PyInstaller.
    goto :end
)

set "ICON="
if exist "ramma.ico" set "ICON=--icon ramma.ico"

echo.
echo Building RAMMA.exe...
echo.
%BUILDPY% -m PyInstaller ramma_start.py --onefile --name RAMMA --console %ICON% --noconfirm --clean
if not "%ERRORLEVEL%"=="0" (
    echo.
    echo [ERROR] Build failed - see above.
    goto :end
)

if exist "dist\RAMMA.exe" copy /Y "dist\RAMMA.exe" "RAMMA.exe" >nul

echo.
echo ==========================================================
echo  Done - RAMMA.exe is in this folder
echo ==========================================================
echo  Double-click it to start the app.
echo  build\, dist\ and RAMMA.spec can be deleted.
echo.
pause

:end
endlocal
