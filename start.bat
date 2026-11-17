@echo off
title AIOps Log Anomaly Detector
cd /d "%~dp0"

echo ========================================================
echo   Drift-Aware, Explainable AIOps Log Anomaly Detector
echo   Starting Full System (FastAPI Backend + Streamlit UI)
echo ========================================================
echo.

:: 1. Detect Python Executable
set "PYTHON_EXE="
if exist "C:\Users\Admin\anaconda3\envs\mlproject\python.exe" (
    set "PYTHON_EXE=C:\Users\Admin\anaconda3\envs\mlproject\python.exe"
) else if exist "%USERPROFILE%\anaconda3\envs\mlproject\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\mlproject\python.exe"
) else if defined CONDA_PREFIX (
    set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [OK] Using Python: %PYTHON_EXE%
echo.

:: 2. Launch FastAPI Backend in background
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "AIOps FastAPI Backend" /min "%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000

:: 3. Brief wait for backend socket binding
timeout /t 2 /nobreak >nul

:: 4. Launch Streamlit Dashboard in foreground and open browser
echo [2/2] Starting Streamlit Dashboard on http://localhost:8501 ...
echo Application is now live. Press Ctrl+C in this window to stop the dashboard.
echo.
start "" "http://localhost:8501"
"%PYTHON_EXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless false

pause
