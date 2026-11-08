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
& $VPy -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126
& $VPy -m pip install -q numpy pandas einops huggingface_hub safetensors tqdm httpx
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
    @{n="base_SOL_h24";  s="SOLUSDT"; m="kronos-base";  h=24; r=330;  k=30}
)
foreach ($t in $Tests) {
    $json = "$Out\$($t.n).json"
    if (Test-Path $json) { Write-Host "skip $($t.n) (done)"; continue }
    Write-Host "-- $($t.n)" -ForegroundColor Yellow
    & $VPy -m app.backtest --symbol $t.s --interval 1h --model $t.m --weights-dir "weights\$($t.m)" `
        --horizon $t.h --step $t.h --rounds $t.r --samples $t.k --device cuda:0 `
        --json $json --csv "$Out\$($t.n).csv"
    if ($LASTEXITCODE -ne 0) { Write-Host "test $($t.n) failed, continuing" -ForegroundColor Red }
}

Write-Host "`n== RESULTS (copy everything below and paste it in the chat) ==" -ForegroundColor Green
& $VPy scripts\summarize_long.py $Out
Write-Host "Files: $Dir\$Out"
