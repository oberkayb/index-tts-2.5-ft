@echo off
rem IndexTTS-2.5 TR MCP sunucusu. stdio (varsayilan) ya da: mcp_server.bat --transport http --port 8765
rem Secenekler: --gpu 0  --checkpoint "C:\yol\gpt.pth"  --voices "C:\sesler"  --default-voice ad
uv run --quiet --project "%~dp0..\index-tts" --with mcp==2.3.0 python "%~dp0mcp_server.py" %*
