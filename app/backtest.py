"""Walk-forward backtest: is the model actually better than a statistical baseline?

    python -m app.backtest --symbol BTCUSDT --interval 1h --rounds 60 --step 12

For each of ``rounds`` anchors (spaced ``step`` candles apart, oldest first) the engines only
see candles up to the anchor, forecast ``horizon`` steps, and are graded on what happened.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import analytics
from .config import INTERVALS, settings
from .data import MarketData
from .engine import BootstrapEngine, KronosEngine


def run(df: pd.DataFrame, engines: dict, horizon: int, rounds: int, step: int, context: int, samples: int,
        temperature: float, top_p: float, seed: int = 0, progress: bool = True) -> dict:
    sec = int((df["timestamps"].iloc[1] - df["timestamps"].iloc[0]).total_seconds())
    last_anchor = len(df) - horizon - 1
    anchors = [last_anchor - i * step for i in range(rounds)][::-1]
    anchors = [a for a in anchors if a >= context - 1]
    if not anchors:
        raise SystemExit("not enough history for that context/horizon; fetch more candles")
    res = {name: [] for name in engines}
    for j, a in enumerate(anchors):
        hist = df.iloc[a - context + 1: a + 1].reset_index(drop=True)
        fut = df["close"].iloc[a + 1: a + 1 + horizon].to_numpy()
        y_ts = pd.Series(pd.date_range(hist["timestamps"].iloc[-1] + pd.Timedelta(seconds=sec), periods=horizon,
                                       freq=f"{sec}s"))
        last = float(hist["close"].iloc[-1])
        for name, eng in engines.items():
            t0 = time.time()
            paths = eng.sample(hist, y_ts, samples, temperature, top_p, seed + j)
            sc = analytics.score(paths[..., 3], last, fut)
            sc["seconds"] = time.time() - t0
            res[name].append(sc)
        if progress:
            line = " | ".join(f"{n}: dir {'HIT ' if r[-1]['direction_hit'] else 'miss'} cov {r[-1]['coverage']:.0%}"
                              for n, r in res.items())
            print(f"[{j + 1:>3}/{len(anchors)}] {hist['timestamps'].iloc[-1]:%Y-%m-%d %H:%M}  {line}", flush=True)
    return {n: analytics.aggregate(r) | {"avg_seconds": float(np.mean([x["seconds"] for x in r]))}
            for n, r in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=settings.symbol)
    ap.add_argument("--interval", default=settings.interval, choices=list(INTERVALS))
    ap.add_argument("--source", default=settings.source)
    ap.add_argument("--model", default=settings.model)
    ap.add_argument("--weights-dir", default=settings.weights_dir)
    ap.add_argument("--horizon", type=int, default=settings.horizon)
    ap.add_argument("--context", type=int, default=settings.lookback)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=settings.temperature)
    ap.add_argument("--top-p", type=float, default=settings.top_p)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--json", help="write results to this file")
    a = ap.parse_args()

    engines: dict = {}
    if not a.baseline_only:
        k = KronosEngine(a.model, a.weights_dir, settings.device, settings.batch_size)
        engines[k.name] = k
        a.context = min(a.context, k.max_context)
    engines["bootstrap"] = BootstrapEngine()
    need = a.context + a.horizon + a.rounds * a.step + 5
    df = MarketData(a.source).candles(a.symbol, a.interval, need)
    print(f"{len(df)} candles {a.symbol} {a.interval}: {df.timestamps.iloc[0]} -> {df.timestamps.iloc[-1]}")
    out = run(df, engines, a.horizon, a.rounds, a.step, a.context, a.samples, a.temperature, a.top_p)
    print("\nengine          rounds  hit-rate  cover(p10-p90)  CRPS-skill  MAE-skill  s/round")
    for n, m in out.items():
        print(f"{n:<15} {m['rounds']:>6}  {m['hit_rate']:>8.1%}  {m['coverage']:>14.1%}  {m['crps_skill']:>10.1%}"
              f"  {m['skill']:>9.1%}  {m['avg_seconds']:>7.2f}")
    print("\nhit-rate > 50% = direction edge; coverage ~80% = honest uncertainty; skill > 0 = beats 'no change'.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
