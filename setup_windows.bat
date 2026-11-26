@echo off
REM ============================================================
REM  Paradox Trading Terminal - Windows one-time setup
REM  Run from the prototype/ folder:  setup_windows.bat
REM  Requires: Python 3.12+ from python.org (tick "Add to PATH")
REM ============================================================
where python >nul 2>nul || (echo ERROR: Python not found. Install from python.org & pause & exit /b 1)

echo [1/4] Creating virtual environment...
python -m venv .venv || (echo ERROR: venv failed & pause & exit /b 1)

echo [2/4] Installing dependencies (2-5 minutes)...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (echo ERROR: install failed & pause & exit /b 1)

echo [3/4] Creating .env from template...
if not exist .env copy .env.example .env

echo [4/4] Done.
echo.
echo OPTIONAL - live MT5 data + execution (Windows-only feature):
echo   1. Install the MetaTrader 5 terminal from your broker
echo   2. .venv\Scripts\pip install MetaTrader5
echo   3. Fill MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in .env
echo.
echo START THE PLATFORM:
echo   start_windows.bat
echo.
pause
