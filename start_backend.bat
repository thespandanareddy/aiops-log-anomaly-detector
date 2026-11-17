@echo off
title AIOps Detector - FastAPI Backend Only
cd /d "%~dp0"

echo ========================================================
echo   AIOps Detector - FastAPI REST Backend Only
echo   Starting server on http://127.0.0.1:8000
echo   OpenAPI Docs: http://127.0.0.1:8000/docs
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

echo Using Python: %PYTHON_EXE%
start "" "http://127.0.0.1:8000/docs"
"%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

pause
