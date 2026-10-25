@echo off
rem Egitim grafikleri: tarayicida http://localhost:6006
setlocal
cd /d "%~dp0"
uv run --project ..\..\index-tts tensorboard --logdir work\runs --port 6006
