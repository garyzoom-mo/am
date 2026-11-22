@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title AlphaMaster Launcher
echo.
".venv\Scripts\python.exe" "tools\launch.py" %*
echo.
echo Press any key to close this window.
echo (Web and training keep running in background.)
pause >nul
