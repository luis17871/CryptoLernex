#!/usr/bin/env bash
# Long backtests for Chronos-2 and TimesFM-2.5 over the same window as the Kronos GPU run.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
OUT=results/backtest/long; END=2026-11-08T00:00Z
for m in chronos-2 timesfm-2.5; do
  for s in BTCUSDT SOLUSDT; do
    for h in 24 4; do
      r=$([ "$h" = 24 ] && echo 330 || echo 2000)
      n="${m%%-*}_${s%USDT}_h${h}"
      [ -f "$OUT/$n.json" ] && continue
      python3 -m app.backtest --symbol "$s" --interval 1h --model "$m" --horizon "$h" --step "$h" --rounds "$r" \
        --samples 30 --end "$END" --json "$OUT/$n.json" --csv "$OUT/$n.csv" > "$OUT/$n.log" 2>&1 && echo "done $n" || echo "FAILED $n"
    done
  done
done
echo ALL_DONE
