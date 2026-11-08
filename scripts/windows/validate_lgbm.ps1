# Validate the LightGBM + derivatives result on more coins (does the 55% on BTC hold elsewhere?).
#   irm https://raw.githubusercontent.com/luis17871/CryptoLernex/claude/lucid-rubin-ltu6sf/scripts/windows/validate_lgbm.ps1 | iex
$ErrorActionPreference = "Stop"
$Dir = Join-Path $env:USERPROFILE "kronos-lab"
# refresh code without touching results
irm https://raw.githubusercontent.com/luis17871/CryptoLernex/claude/lucid-rubin-ltu6sf/scripts/windows/update_monitor.ps1 | iex
Set-Location $Dir
$VPy = Join-Path $Dir ".venv\Scripts\python.exe"
$env:KRONOS_BACKTEST_DIR = Join-Path $Dir "results\backtest\long"
Write-Host "-- LightGBM on 8 more coins, horizons 5 and 10 days" -ForegroundColor Yellow
& $VPy -m app.strategy deriv --symbols ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT --horizons 5,10
Write-Host "-- Trading simulation (now with the trend-only control)" -ForegroundColor Yellow
& $VPy -m app.strategy trade
Write-Host "`n== RESULTS (copy only the lines with lgbm, TRADE lgbm and POOLED) ==" -ForegroundColor Green
& $VPy scripts\summarize_long.py results\backtest\long | Select-String -Pattern "lgbm|POOLED"
