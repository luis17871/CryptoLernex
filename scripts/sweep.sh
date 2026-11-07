#!/usr/bin/env bash
# Backtest sweep over Kronos configurations (same 2-week window, 28 rounds every 12h).
set -u
cd "$(dirname "$0")/.."
OUT=results/backtest/sweep; mkdir -p "$OUT"
run() { # name symbol model T top_p samples
  local n=$1; shift
  [ -f "$OUT/$n.json" ] && return
  python3 -m app.backtest --symbol "$1" --interval 1h --model "$2" --weights-dir "weights/$2" \
    --temperature "$3" --top-p "$4" --samples "$5" --rounds 28 --step 12 \
    --json "$OUT/$n.json" --csv "$OUT/$n.csv" > "$OUT/$n.log" 2>&1
  echo "done $n"
}
run small_T1.0_p1.0   BTCUSDT kronos-small 1.0 1.0 30
run small_T1.5_p1.0   BTCUSDT kronos-small 1.5 1.0 30
run mini_T1.0_p0.9    BTCUSDT kronos-mini  1.0 0.9 30
run small_eth_T1.0_p0.9 ETHUSDT kronos-small 1.0 0.9 30
run base_T1.0_p0.9    BTCUSDT kronos-base  1.0 0.9 20
echo SWEEP_FINISHED
