@echo off
rem Takip ve ayar arayuzu: http://127.0.0.1:7870
setlocal
chcp 65001 >nul
cd /d "%~dp0"
start "" http://127.0.0.1:7870
uv run --project ..\..\index-tts python ui\server.py %*
if errorlevel 1 pause
