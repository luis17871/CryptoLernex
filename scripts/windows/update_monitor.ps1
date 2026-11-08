# Update the backtest monitor page without stopping the backtests that are running.
#   irm https://raw.githubusercontent.com/luis17871/CryptoLernex/claude/lucid-rubin-ltu6sf/scripts/windows/update_monitor.ps1 | iex
$ErrorActionPreference = "Stop"
$Branch = "claude/lucid-rubin-ltu6sf"
$Dir = Join-Path $env:USERPROFILE "kronos-lab"
$Zip = Join-Path $env:TEMP "kronos-lab.zip"
Invoke-WebRequest "https://github.com/luis17871/CryptoLernex/archive/refs/heads/$Branch.zip" -OutFile $Zip
$Tmp = Join-Path $env:TEMP "kronos-lab-src"
if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
Expand-Archive $Zip $Tmp
$Src = (Get-ChildItem $Tmp | Select-Object -First 1).FullName
foreach ($p in "app", "web", "scripts") { Copy-Item (Join-Path $Src $p) $Dir -Recurse -Force }
# restart only the monitor (port 8020); running tests keep going and the next ones use the new code
Get-NetTCPConnection -LocalPort 8020 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep 1
$env:KRONOS_BACKTEST_DIR = Join-Path $Dir "results\backtest\long"
Start-Process (Join-Path $Dir ".venv\Scripts\python.exe") -ArgumentList "-m", "uvicorn", "app.monitor:app", "--host", "0.0.0.0", "--port", "8020" -WorkingDirectory $Dir -WindowStyle Hidden
Start-Sleep 3
Start-Process "http://localhost:8020"
Write-Host "Monitor updated: http://localhost:8020"
