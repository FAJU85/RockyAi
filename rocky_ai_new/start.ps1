$ErrorActionPreference = "Stop"
$env:DEEPSEEK_API_KEY = $env:DEEPSEEK_API_KEY -as [string]
if (-not $env:DEEPSEEK_API_KEY -or $env:DEEPSEEK_API_KEY.Trim().Length -eq 0) {
  Write-Host "WARNING: DEEPSEEK_API_KEY is not set. Set it before starting for full AI responses." -ForegroundColor Yellow
}
python .\server.py

