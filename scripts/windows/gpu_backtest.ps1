# Long walk-forward backtest of Kronos on an NVIDIA GPU (Windows).
# Run in PowerShell:
#   irm https://raw.githubusercontent.com/luis17871/CryptoLernex/claude/lucid-rubin-ltu6sf/scripts/windows/gpu_backtest.ps1 | iex
# It downloads the project to %USERPROFILE%\kronos-lab, installs Python packages with CUDA,
# downloads the Kronos models and tests ~11 months of BTC and SOL. Safe to re-run: finished
# tests are skipped.
$ErrorActionPreference = "Stop"
$Branch = "claude/lucid-rubin-ltu6sf"
$Dir = Join-Path $env:USERPROFILE "kronos-lab"

function Find-Python {
    foreach ($v in "3.12", "3.11", "3.10") {
        try { $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    }
    try { $p = & python -c "import sys; print(sys.executable if (3,10) <= sys.version_info[:2] <= (3,12) else '')" 2>$null; if ($p) { return $p.Trim() } } catch {}
    return $null
}

Write-Host "== 1/5 Python" -ForegroundColor Cyan
$Py = Find-Python
if (-not $Py) {
    Write-Host "Python 3.12 not found. Installing with winget (accept the prompts)..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $Py = Find-Python
    if (-not $Py) { throw "Python 3.12 still not found. Close and reopen PowerShell, then run the command again." }
}
Write-Host "Using $Py"

Write-Host "== 2/5 Project files" -ForegroundColor Cyan
$Zip = Join-Path $env:TEMP "kronos-lab.zip"
Invoke-WebRequest "https://github.com/luis17871/CryptoLernex/archive/refs/heads/$Branch.zip" -OutFile $Zip
$Tmp = Join-Path $env:TEMP "kronos-lab-src"
if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
Expand-Archive $Zip $Tmp
$Src = Get-ChildItem $Tmp | Select-Object -First 1
New-Item -ItemType Directory -Force $Dir | Out-Null
Copy-Item (Join-Path $Src.FullName "*") $Dir -Recurse -Force   # keeps .venv, weights and results
Set-Location $Dir

Write-Host "== 3/5 Packages (PyTorch with CUDA, ~3 GB the first time)" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { & $Py -m venv .venv }
$VPy = Join-Path $Dir ".venv\Scripts\python.exe"
& $VPy -m pip install -q -U pip
Write-Host "Downloading PyTorch CUDA (~2.5 GB). This can take a while on slow connections."
& $VPy -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --progress-bar on
& $VPy -m pip install -q numpy pandas einops huggingface_hub safetensors tqdm httpx fastapi uvicorn chronos-forecasting timesfm lightgbm scikit-learn
& $VPy -c "import torch; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT AVAILABLE'); raise SystemExit(0 if ok else 3)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch does not see the GPU. Update the NVIDIA driver (GeForce Experience or nvidia.com) and run again." }

Write-Host "== 4/5 Kronos models" -ForegroundColor Cyan
$env:HF_HUB_DISABLE_XET = "1"
foreach ($m in "kronos-small", "kronos-base") {
    if (-not (Test-Path "weights\$m\model\config.json")) { & $VPy scripts\download_weights.py $m .\weights }
}

Write-Host "== 5/5 Backtests (BTC and SOL, ~11 months, rounds that never overlap)" -ForegroundColor Cyan
$Out = "results\backtest\long"
New-Item -ItemType Directory -Force $Out | Out-Null
$Tests = @(
    @{n="small_BTC_h24"; s="BTCUSDT"; m="kronos-small"; h=24; r=330;  k=30},
    @{n="small_SOL_h24"; s="SOLUSDT"; m="kronos-small"; h=24; r=330;  k=30},
    @{n="small_BTC_h4";  s="BTCUSDT"; m="kronos-small"; h=4;  r=2000; k=30},
    @{n="small_SOL_h4";  s="SOLUSDT"; m="kronos-small"; h=4;  r=2000; k=30},
    @{n="base_BTC_h24";  s="BTCUSDT"; m="kronos-base";  h=24; r=330;  k=30},
    @{n="base_SOL_h24";  s="SOLUSDT"; m="kronos-base";  h=24; r=330;  k=30},
    @{n="chronos_BTC_h24"; s="BTCUSDT"; m="chronos-2";   h=24; r=330;  k=30},
    @{n="chronos_BTC_h4";  s="BTCUSDT"; m="chronos-2";   h=4;  r=2000; k=30},
    @{n="chronos_SOL_h24"; s="SOLUSDT"; m="chronos-2";   h=24; r=330;  k=30},
    @{n="chronos_SOL_h4";  s="SOLUSDT"; m="chronos-2";   h=4;  r=2000; k=30},
    @{n="timesfm_BTC_h24"; s="BTCUSDT"; m="timesfm-2.5"; h=24; r=330;  k=30},
    @{n="timesfm_BTC_h4";  s="BTCUSDT"; m="timesfm-2.5"; h=4;  r=2000; k=30},
    @{n="timesfm_SOL_h24"; s="SOLUSDT"; m="timesfm-2.5"; h=24; r=330;  k=30},
    @{n="timesfm_SOL_h4";  s="SOLUSDT"; m="timesfm-2.5"; h=4;  r=2000; k=30}
)
# 3) daily candles, horizons 5 / 10 / 20 days (all available history up to the same end date)
foreach ($m in @(@("kronos-small", "small"), @("chronos-2", "chronos"), @("timesfm-2.5", "timesfm"))) {
    foreach ($s in "BTCUSDT", "SOLUSDT") {
        foreach ($h in 5, 10, 20) {
            $Tests += @{n="$($m[1])_$($s.Replace('USDT',''))_d$h"; s=$s; m=$m[0]; h=$h; r=600; k=30; i="1d"}
        }
    }
}
# results already computed in the cloud (same window): reuse them instead of recomputing
$RepoRes = Join-Path $Src.FullName "results\backtest\long"
if (Test-Path $RepoRes) {
    Get-ChildItem $RepoRes -File -Include *.json, *.csv -Recurse | Where-Object { $_.Name -ne "plan.json" } |
        ForEach-Object { $dst = Join-Path $Out $_.Name; if (-not (Test-Path $dst)) { Copy-Item $_.FullName $dst } }
}
# plan for the monitor page, then start the monitor (http://localhost:8020) and open it
$Tests | ForEach-Object { @{name=$_.n; symbol=$_.s; model=$_.m; horizon=$_.h; interval=$(if ($_.i) { $_.i } else { "1h" })} } |
    ConvertTo-Json | Set-Content "$Out\plan.json" -Encoding UTF8
$env:KRONOS_BACKTEST_DIR = Join-Path $Dir $Out
$busy = Get-NetTCPConnection -LocalPort 8020 -State Listen -ErrorAction SilentlyContinue
if (-not $busy) {
    Start-Process $VPy -ArgumentList "-m", "uvicorn", "app.monitor:app", "--host", "0.0.0.0", "--port", "8020" -WorkingDirectory $Dir -WindowStyle Hidden
    Start-Sleep 3
}
Start-Process "http://localhost:8020"
Write-Host "Monitor: http://localhost:8020 (also from other devices: http://<this-PC-IP>:8020)"

foreach ($t in $Tests) {
    $json = "$Out\$($t.n).json"
    if (Test-Path $json) { Write-Host "skip $($t.n) (done)"; continue }
    Write-Host "-- $($t.n)" -ForegroundColor Yellow
    $wd = if ($t.m -like "kronos-*") { "weights\$($t.m)" } else { "" }
    $iv = if ($t.i) { $t.i } else { "1h" }
    & $VPy -m app.backtest --symbol $t.s --interval $iv --model $t.m --weights-dir $wd `
        --horizon $t.h --step $t.h --rounds $t.r --samples $t.k --device cuda:0 --end 2026-11-08T00:00Z `
        --json $json --csv "$Out\$($t.n).csv"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "test $($t.n) failed, continuing" -ForegroundColor Red
        Set-Content "$Out\$($t.n).failed" "exit code $LASTEXITCODE"
    } else { Remove-Item "$Out\$($t.n).failed" -ErrorAction SilentlyContinue }
}

$env:KRONOS_BACKTEST_DIR = Join-Path $Dir $Out
Write-Host "-- 4) LightGBM with derivatives data (funding, premium, taker flow)" -ForegroundColor Yellow
if (-not (Test-Path "$Out\lgbm_SOL_d10.json")) { & $VPy -m app.strategy deriv --symbols BTCUSDT,SOLUSDT --horizons 5,10 }
Write-Host "-- 2) 20 coins, weekly long top-5 / short bottom-5" -ForegroundColor Yellow
if (-not (Test-Path "$Out\xsec.json")) { & $VPy -m app.strategy xsec --models kronos-small,chronos-2,timesfm-2.5 --weeks 104 --device cuda }
Write-Host "-- 1) Trading simulation of every test (costs, strong signals, trend filter, stops)" -ForegroundColor Yellow
& $VPy -m app.strategy trade

Write-Host "`n== RESULTS (copy everything below and paste it in the chat) ==" -ForegroundColor Green
& $VPy scripts\summarize_long.py $Out
Write-Host "Files: $Dir\$Out"
