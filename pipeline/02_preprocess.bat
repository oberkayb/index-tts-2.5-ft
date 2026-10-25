@echo off
rem GPU feature cache. Yarida kalirsa tekrar calistir, kaldigi yerden devam eder. Deneme: --limit 200
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\..\index-tts python pipeline.py preprocess %*
if errorlevel 1 (
  echo.
  echo [HATA] Adim basarisiz. Ayrinti icin: work\logs
  pause
  exit /b 1
)
