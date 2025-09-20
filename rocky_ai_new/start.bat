@echo off
if "%DEEPSEEK_API_KEY%"=="" (
  echo WARNING: DEEPSEEK_API_KEY is not set. Set it before starting for full AI responses.
)
python server.py

