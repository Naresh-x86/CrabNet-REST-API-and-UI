@echo off
REM Start the CrabNet API server

echo ========================================
echo CrabNet API Server
echo ========================================
echo.

cd /d %~dp0\..

REM Activate virtual environment
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call .venv\scripts\activate
)

echo Starting API server on http://localhost:8000
echo.
echo Interactive docs: http://localhost:8000/docs
echo.

cd API_backend
python main.py

pause
