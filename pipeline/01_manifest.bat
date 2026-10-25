@echo off
rem dataset.jsonl -> filtrelenmis manifest. Yeniden uretmek icin: 01_manifest.bat --force
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\..\index-tts python pipeline.py manifest %*
if errorlevel 1 (
  echo.
  echo [HATA] Adim basarisiz. Ayrinti icin: work\logs
  pause
  exit /b 1
)
