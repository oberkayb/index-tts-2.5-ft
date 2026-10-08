@echo off
rem Fine-tune WebUI: http://127.0.0.1:7860  (secenekler: --gpu 1  --port 7861  --qwen-emo  --checkpoint "C:\yol\best.pt")
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\index-tts --extra webui python webui_ft.py %*
if errorlevel 1 pause
