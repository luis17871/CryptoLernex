"""Walk-forward backtest: is the model actually better than a statistical baseline?

    python -m app.backtest --symbol BTCUSDT --interval 1h --rounds 60 --step 12

For each of ``rounds`` anchors (spaced ``step`` candles apart, oldest first) the engines only
see candles up to the anchor, forecast ``horizon`` steps, and are graded on what happened.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import analytics
from .config import INTERVALS, settings
from .data import MarketData
from .engine import BootstrapEngine, KronosEngine


def run(df: pd.DataFrame, engines: dict, horizon: int, rounds: int, step: int, context: int, samples: int,
        temperature: float, top_p: float, seed: int = 0, progress: bool = True,
        records: list | None = None, progress_file: str | None = None, meta: dict | None = None) -> dict:
    sec = int((df["timestamps"].iloc[1] - df["timestamps"].iloc[0]).total_seconds())
    last_anchor = len(df) - horizon - 1
    anchors = [last_anchor - i * step for i in range(rounds)][::-1]
    anchors = [a for a in anchors if a >= context - 1]
    if not anchors:
        raise SystemExit("not enough history for that context/horizon; fetch more candles")
    res = {name: [] for name in engines}
    t_start = time.time()
    compact: list = []  # per round and engine: [round, engine, hit, coverage, crps_skill] for the monitor
    every = max(1, len(anchors) // 200)
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
            pc = paths[:, -1, 3]
            p10, p50, p90 = (float(v) for v in np.percentile(pc, [10, 50, 90]))
            # [round, engine, hit, coverage, crps_skill, anchor_ts, last, median, p10, p90, actual, P(up)]
            compact.append([j, name, int(sc["direction_hit"]), round(sc["coverage"], 4), round(sc["crps_skill"], 4),
                            int(hist["timestamps"].iloc[-1].timestamp()), round(last, 6), round(p50, 6),
                            round(p10, 6), round(p90, 6), round(float(fut[-1]), 6),
                            round(float((pc > last).mean()), 3)])
            if records is not None:
                summ = analytics.summarize(paths, hist["close"].to_numpy())
                records.append({"engine": name, "anchor": str(hist["timestamps"].iloc[-1]), "last_close": last,
                                "median": summ["median"], "p10": summ["final"]["p10"], "p90": summ["final"]["p90"],
                                "prob_up": summ["prob_up"], "regime": summ["regime"], "actual": float(fut[-1]),
                                **{k: sc[k] for k in ("coverage", "inside_final", "direction_hit", "crps_skill",
                                                      "skill", "seconds")}})
        if progress_file and ((j + 1) % every == 0 or j + 1 == len(anchors)):
            _write_progress(progress_file, meta or {}, j + 1, len(anchors), t_start,
                            str(hist["timestamps"].iloc[-1]), compact)
        if progress and ((j + 1) % max(1, len(anchors) // 40) == 0 or j + 1 == len(anchors)):
            line = " | ".join(f"{n}: dir {'HIT ' if r[-1]['direction_hit'] else 'miss'} cov {r[-1]['coverage']:.0%}"
                              for n, r in res.items())
            print(f"[{j + 1:>3}/{len(anchors)}] {hist['timestamps'].iloc[-1]:%Y-%m-%d %H:%M}  {line}", flush=True)
    return {n: analytics.aggregate(r) | {"avg_seconds": float(np.mean([x["seconds"] for x in r]))}
            for n, r in res.items()}


def _write_progress(path: str, meta: dict, done: int, total: int, t_start: float, last_anchor: str,
                    compact: list) -> None:
    """Small JSON the monitor page polls while a test runs (written atomically)."""
    el = time.time() - t_start
    doc = {"status": "running" if done < total else "finishing", "done": done, "total": total,
           "elapsed_s": round(el, 1), "eta_s": round(el / done * (total - done), 1) if done else None,
           "updated": time.time(), "last_anchor": last_anchor, "meta": meta, "rounds": compact}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


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
    ap.add_argument("--device", default=settings.device, help="cuda:0, cpu, mps (default: auto)")
    ap.add_argument("--json", help="write aggregate results to this file")
    ap.add_argument("--csv", help="write one row per engine and round to this file")
    ap.add_argument("--progress", help="live progress JSON for the monitor (default: <json>.progress.json)")
    a = ap.parse_args()

    engines: dict = {}
    if not a.baseline_only:
        k = KronosEngine(a.model, a.weights_dir, a.device, settings.batch_size)
        print(f"model {a.model} on {k.device}")
        engines[k.name] = k
        a.context = min(a.context, k.max_context)
    engines["bootstrap"] = BootstrapEngine()
    need = a.context + a.horizon + a.rounds * a.step + 5
    df = MarketData(a.source).candles(a.symbol, a.interval, need)
    print(f"{len(df)} candles {a.symbol} {a.interval}: {df.timestamps.iloc[0]} -> {df.timestamps.iloc[-1]}")
    recs: list = []
    prog = a.progress or (a.json[:-5] + ".progress.json" if a.json and a.json.endswith(".json") else None)
    meta = {"symbol": a.symbol, "interval": a.interval, "model": a.model, "horizon": a.horizon, "step": a.step,
            "samples": a.samples, "first": str(df.timestamps.iloc[0]), "last": str(df.timestamps.iloc[-1])}
    out = run(df, engines, a.horizon, a.rounds, a.step, a.context, a.samples, a.temperature, a.top_p, records=recs,
              progress_file=prog, meta=meta)
    if a.csv:
        pd.DataFrame(recs).to_csv(a.csv, index=False)
    print("\nengine          rounds  hit-rate  cover(p10-p90)  CRPS-skill  MAE-skill  s/round")
    for n, m in out.items():
        print(f"{n:<15} {m['rounds']:>6}  {m['hit_rate']:>8.1%}  {m['coverage']:>14.1%}  {m['crps_skill']:>10.1%}"
              f"  {m['skill']:>9.1%}  {m['avg_seconds']:>7.2f}")
    independent = a.step >= a.horizon
    print(f"\nsignificance ({'independent rounds' if independent else 'OVERLAPPING rounds: optimistic'}):")
    for n in out:
        hits = [r["direction_hit"] for r in recs if r["engine"] == n]
        k_, n_ = int(sum(hits)), len(hits)
        lo, hi = analytics.wilson(k_, n_)
        pv = analytics.binom_p_greater(k_, n_)
        verdict = "ventaja" if lo > 0.5 else "peor que azar" if hi < 0.5 else "no concluyente"
        out[n] |= {"hits": k_, "ci_low": lo, "ci_high": hi, "p_value": pv, "verdict": verdict,
                   "independent": independent}
        print(f"{n:<15} {k_}/{n_} = {k_ / n_:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  p={pv:.4f}  -> {verdict}")
    out["_config"] = {k: v for k, v in vars(a).items() if k not in ("json", "csv", "weights_dir")} | {
        "first": str(df.timestamps.iloc[0]), "last": str(df.timestamps.iloc[-1])}
    print("\nhit-rate > 50% = direction edge; coverage ~80% = honest uncertainty; skill > 0 = beats 'no change'.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    if prog and os.path.exists(prog):  # keep the per-round curve for the monitor, mark it done
        with open(prog) as f:
            doc = json.load(f)
        doc["status"] = "done"
        with open(prog, "w") as f:
            json.dump(doc, f)


if __name__ == "__main__":
    main()
