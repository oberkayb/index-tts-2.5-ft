@echo off
rem Egitim (kaldigi yerden devam eder). Once olcum: 04_train.bat --probe 20
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\..\index-tts python pipeline.py train %*
if errorlevel 1 (
  echo.
  echo [HATA] Adim basarisiz. Ayrinti icin: work\logs
  pause
  exit /b 1
)
