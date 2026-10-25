@echo off
rem Ortam (uv sync), model agirliklari, stok model dogrulamasi, dataset kontrolu, smoke test.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo == uv sync: ..\..\index-tts ortami ==
uv sync --project ..\..\index-tts
if errorlevel 1 goto :fail
uv run --project ..\..\index-tts python pipeline.py setup %*
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo [HATA] Kurulum basarisiz. Ayrinti icin: work\logs
pause
exit /b 1
