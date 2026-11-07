"""Forward (live, out-of-sample) test. Designed to run hourly from CI for a week.

    python -m app.forward step     # record a forecast for the newest closed candle (if not done yet)
    python -m app.forward report   # grade every recorded round against real candles -> REPORT.md

State is plain JSON lines (git-friendly). Nothing is graded until the candles really exist, so
there is no look-ahead: each forecast is written before its future is known.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics
from .config import INTERVALS, MODELS
from .data import MarketData
from .engine import BootstrapEngine, KronosEngine

OUT = Path(os.environ.get("KRONOS_FORWARD_DIR", "results/forward"))


def _rounds_file(symbol: str, interval: str) -> Path:
    return OUT / f"rounds_{symbol}_{interval}.jsonl"


def load_rounds(symbol: str, interval: str) -> list[dict]:
    f = _rounds_file(symbol, interval)
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def step(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = MarketData(a.source)
    model_key = a.model
    ctx = min(a.context, MODELS[model_key]["max_context"])
    df = md.candles(a.symbol, a.interval, ctx)
    anchor = int(df["timestamps"].iloc[-1].timestamp())
    done = {(r["anchor_ts"], r["engine"]) for r in load_rounds(a.symbol, a.interval)}
    sec = INTERVALS[a.interval]
    y_ts = pd.Series(pd.date_range(df["timestamps"].iloc[-1] + pd.Timedelta(seconds=sec), periods=a.horizon,
                                   freq=f"{sec}s"))
    engines = {}
    if (anchor, model_key) not in done:
        engines[model_key] = lambda: KronosEngine(model_key, a.weights_dir, "", 50)
    if (anchor, "bootstrap") not in done:
        engines["bootstrap"] = BootstrapEngine
    if not engines:
        print(f"anchor {df['timestamps'].iloc[-1]} already recorded")
        return
    with _rounds_file(a.symbol, a.interval).open("a") as f:
        for name, make in engines.items():
            eng = make()
            t0 = time.time()
            paths = eng.sample(df, y_ts, a.samples, a.temperature, a.top_p, anchor % 100000)
            summ = analytics.summarize(paths, df["close"].to_numpy())
            rec = {"engine": name, "symbol": a.symbol, "interval": a.interval, "anchor_ts": anchor,
                   "recorded_at": int(time.time()), "source": md.active, "last_close": summ["last_close"],
                   "horizon": a.horizon, "samples": a.samples, "temperature": a.temperature, "top_p": a.top_p,
                   "seconds": round(time.time() - t0, 1), "median": summ["median"], "p10": summ["final"]["p10"],
                   "p90": summ["final"]["p90"], "prob_up": summ["prob_up"], "regime": summ["regime"],
                   "paths": np.round(paths[..., 3], 2).tolist()}
            f.write(json.dumps(rec) + "\n")
            print(f"{name}: anchor {df['timestamps'].iloc[-1]} last {summ['last_close']:.2f} "
                  f"median {summ['median']:.2f} P(up) {summ['prob_up']:.0%} ({rec['seconds']}s)")


def grade(rounds: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    ts = (df["timestamps"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().astype(int).to_numpy()
    close = df["close"].to_numpy()
    rows = []
    for r in rounds:
        fut = close[ts > r["anchor_ts"]][: r["horizon"]]
        if len(fut) < r["horizon"]:
            continue  # future not complete yet
        sc = analytics.score(np.asarray(r["paths"]), r["last_close"], fut)
        rows.append({"engine": r["engine"], "anchor": pd.to_datetime(r["anchor_ts"], unit="s", utc=True),
                     "last_close": r["last_close"], "median": r["median"], "p10": r["p10"], "p90": r["p90"],
                     "prob_up": r["prob_up"], "regime": r["regime"], "actual": sc["actual_last"],
                     "direction_hit": sc["direction_hit"], "coverage": sc["coverage"],
                     "inside_final": sc["inside_final"], "crps_skill": sc["crps_skill"], "skill": sc["skill"],
                     "pit": sc["pit"]})
    return pd.DataFrame(rows)


def binom_p(k: int, n: int) -> float:
    """One-sided P(X >= k) for X ~ Binomial(n, 0.5): is the hit-rate better than a coin?"""
    from math import comb
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n) if n else 1.0


def report(a) -> None:
    rounds = load_rounds(a.symbol, a.interval)
    if not rounds:
        print("no rounds recorded yet")
        return
    sec = INTERVALS[a.interval]
    first = min(r["anchor_ts"] for r in rounds)
    need = int((time.time() - first) // sec) + 5
    df = MarketData(a.source).candles(a.symbol, a.interval, min(need, 5000))
    g = grade(rounds, df)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"# Forward test {a.symbol} {a.interval}", "",
             f"Actualizado: {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC. Rondas registradas: "
             f"{len(rounds)}; evaluadas (24 velas completas): {len(g)}.", ""]
    if len(g):
        g.drop(columns=["pit"]).to_csv(OUT / f"graded_{a.symbol}_{a.interval}.csv", index=False)
        lines += ["| motor | rondas | hit-rate dir | p-valor vs moneda | cobertura p10-p90 | final dentro banda |"
                  " CRPS skill | MAE skill |", "|---|---|---|---|---|---|---|---|"]
        for eng, d in g.groupby("engine"):
            k, n = int(d.direction_hit.sum()), len(d)
            lines.append(f"| {eng} | {n} | {k / n:.1%} | {binom_p(k, n):.3f} | {d.coverage.mean():.1%} | "
                         f"{d.inside_final.mean():.1%} | {d.crps_skill.mean():+.1%} | {d.skill.mean():+.1%} |")
        lines += ["", "Ideal: hit-rate > 50 % con p-valor < 0,05; cobertura cercana a 80 %; skills > 0.",
                  "Nota: rondas horarias con horizonte de 24 h se solapan, así que no son independientes;"
                  " el p-valor es optimista."]
    (OUT / f"REPORT_{a.symbol}_{a.interval}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step", "report"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--model", default="kronos-small")
    ap.add_argument("--weights-dir", default=os.environ.get("KRONOS_WEIGHTS_DIR", ""))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    a = ap.parse_args()
    step(a) if a.cmd == "step" else report(a)


if __name__ == "__main__":
    main()
