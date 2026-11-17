@echo off
title AIOps Log Anomaly Detector
cd /d "%~dp0"

echo ========================================================
echo   Drift-Aware, Explainable AIOps Log Anomaly Detector
echo   Starting Full System (FastAPI + Streamlit Dashboard)
echo ========================================================
echo.

:: Detect Python Executable
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

:: 1. Start FastAPI in a background window
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "AIOps FastAPI Backend (Port 8000)" /min "%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000

:: Safe pause for 2 seconds
ping -n 3 127.0.0.1 >nul

:: 2. Start Streamlit Dashboard (Streamlit will auto-open the browser when ready)
echo [2/2] Starting Streamlit Dashboard on http://127.0.0.1:8501 ...
echo.
echo ========================================================
echo   Dashboard URL: http://127.0.0.1:8501
echo   FastAPI Docs:  http://127.0.0.1:8000/docs
echo   (Keep this window open while using the application)
echo ========================================================
echo.

"%PYTHON_EXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless false --server.address 127.0.0.1

pause
