@echo off
title AIOps Detector - FastAPI Backend
cd /d "%~dp0"

echo ========================================================
echo   AIOps Detector - FastAPI REST Backend
echo   Starting server on http://127.0.0.1:8000
echo   OpenAPI Docs: http://127.0.0.1:8000/docs
echo ========================================================
echo.

set PYTHON_EXE=C:\Users\Admin\anaconda3\envs\mlproject\python.exe

start "" "http://127.0.0.1:8000/docs"
"%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

pause
