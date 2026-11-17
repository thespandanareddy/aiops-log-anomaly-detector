@echo off
title AIOps Log Anomaly Detector - Dashboard
cd /d "%~dp0"

echo ========================================================
echo   Drift-Aware, Explainable AIOps Log Anomaly Detector
echo   Starting application on F:\AIOps_Log_Anomaly_Detector
echo ========================================================
echo.

set PYTHON_EXE=C:\Users\Admin\anaconda3\envs\mlproject\python.exe

if not exist "%PYTHON_EXE%" (
    if defined CONDA_PREFIX (
        set PYTHON_EXE=%CONDA_PREFIX%\python.exe
    ) else (
        set PYTHON_EXE=python
    )
)

echo Using Python: %PYTHON_EXE%
echo Starting Streamlit Dashboard on http://localhost:8501...
echo (Press Ctrl+C to stop)
echo.

start "" "http://localhost:8501"
"%PYTHON_EXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless false

pause
