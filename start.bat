@echo off
title AIOps Log Anomaly Detector
cd /d "%~dp0"

echo ========================================================
echo   Drift-Aware, Explainable AIOps Log Anomaly Detector
echo   Starting application in offline-ready local mode...
echo ========================================================
echo.

set PYTHON_EXE=C:\Users\Admin\anaconda3\envs\mlproject\python.exe

if not exist "%PYTHON_EXE%" (
    echo Python environment not found at: %PYTHON_EXE%
    echo Please make sure the conda environment 'mlproject' is installed.
    pause
    exit /b 1
)

echo Starting Streamlit Dashboard on http://localhost:8501...
echo (Press Ctrl+C to stop)
echo.

start "" "http://localhost:8501"
"%PYTHON_EXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless false

pause
