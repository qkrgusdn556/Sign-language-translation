@echo off
cd /d "%~dp0"
powershell -NoProfile -File "%~dp0scripts\start-web.ps1"
pause
