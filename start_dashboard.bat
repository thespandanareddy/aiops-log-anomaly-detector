@echo off
title AIOps Log Anomaly Detector - Dashboard
cd /d "%~dp0"

echo ========================================================
echo   Drift-Aware, Explainable AIOps Log Anomaly Detector
echo   Starting Streamlit Dashboard Only...
echo ========================================================
echo.

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
echo Starting Streamlit Dashboard on http://127.0.0.1:8501 ...
echo.
echo ========================================================
echo   Dashboard URL: http://127.0.0.1:8501
echo   (Keep this window open while using the application)
echo ========================================================
echo.

"%PYTHON_EXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless false --server.address 127.0.0.1

pause
