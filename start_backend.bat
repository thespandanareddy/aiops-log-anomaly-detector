@echo off
title AIOps Detector - FastAPI Backend
cd /d "%~dp0"

echo ========================================================
echo   AIOps Detector - FastAPI REST Backend
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

echo [OK] Using Python: %PYTHON_EXE%
echo.
echo ========================================================
echo   FastAPI Swagger UI: http://127.0.0.1:8000/docs
echo   Health Check:       http://127.0.0.1:8000/api/health
echo   (Keep this window open while using the API)
echo ========================================================
echo.

"%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

pause
