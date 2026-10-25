@echo off
rem Son export'u olc. Baska model: 06_eval.bat --checkpoint "C:\yol\gpt.pth" --label eski_29k
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\..\index-tts python pipeline.py eval %*
if errorlevel 1 (
  echo.
  echo [HATA] Adim basarisiz. Ayrinti icin: work\logs
  pause
  exit /b 1
)
