@echo off
rem Son checkpoint'i gpt.pth'ye cevir. Belirli adim: 05_export.bat --step 12000
setlocal
chcp 65001 >nul
cd /d "%~dp0"
uv run --project ..\..\index-tts python pipeline.py export %*
if errorlevel 1 (
  echo.
  echo [HATA] Adim basarisiz. Ayrinti icin: work\logs
  pause
  exit /b 1
)
