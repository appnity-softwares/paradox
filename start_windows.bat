@echo off
REM Starts the Paradox terminal on http://127.0.0.1:8799
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run setup_windows.bat first & pause & exit /b 1)
set DSH_ADMIN_PASSWORD=test-pass-1234
.venv\Scripts\python -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8799
pause
