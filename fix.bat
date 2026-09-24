@echo off
echo ===================================================
echo FANet Industrial System - Diagnostics ^& Fix Utility
echo ===================================================
echo.

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please ensure you run setup_env.bat or manually create a virtual environment first.
    pause
    exit /b
)

call venv\Scripts\activate.bat
python fix_system.py

echo.
pause
