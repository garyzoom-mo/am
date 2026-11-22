@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title AlphaMaster Stopper
echo.
".venv\Scripts\python.exe" "tools\launch.py" --stop %*
echo.
echo Press any key to close this window.
pause >nul
