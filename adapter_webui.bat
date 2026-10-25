@echo off
rem Orijinal IndexTTS-2.5 + dil adaptoru: http://127.0.0.1:7862  (secenekler: --gpu 1  --port 7863  --adapter "adapters\tr_v2.safetensors")
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\index-tts --extra webui python adapter_webui.py %*
if errorlevel 1 pause
