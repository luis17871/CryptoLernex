"""Live monitor for long backtests. Light: no model, no scheduler.

    python -m uvicorn app.monitor:app --port 8020      # then open http://localhost:8020

Reads results/backtest/long (or KRONOS_BACKTEST_DIR): plan.json (queued tests), <name>.progress.json
(written while a test runs) and <name>.json (final results).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import analytics
from .config import ROOT

DIR = Path(os.environ.get("KRONOS_BACKTEST_DIR", ROOT / "results" / "backtest" / "long"))
STALE_S = 15 * 60

app = FastAPI(title="Kronos backtests")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))  # PowerShell writes a BOM
    except (OSError, ValueError):
        return None


def _engine_stats(rounds: list, engine: str) -> dict:
    rs = [r for r in rounds if r[1] == engine]
    n = len(rs)
    k = sum(r[2] for r in rs)
    lo, hi = analytics.wilson(k, n)
    curve, hits = [], 0
    stride = max(1, n // 200)
    for i, r in enumerate(rs, 1):
        hits += r[2]
        if i % stride == 0 or i == n:
            curve.append([r[0] + 1, hits / i, *analytics.wilson(hits, i)])
    verdict = "pocos datos" if n < 30 else "ventaja" if lo > 0.5 else "peor que azar" if hi < 0.5 else "no concluyente"
    out = {"n": n, "hits": k, "hit_rate": k / n if n else None, "ci_low": lo, "ci_high": hi,
           "p_value": analytics.binom_p_greater(k, n), "coverage": sum(r[3] for r in rs) / n if n else None,
           "crps_skill": sum(r[4] for r in rs) / n if n else None, "verdict": verdict, "curve": curve}
    rich = [r for r in rs if len(r) >= 12]
    if rich:
        # per-round forecast vs reality, at most 400 points for the hero chart
        st = max(1, len(rich) // 400)
        out["series"] = [[r[5], r[6], r[7], r[8], r[9], r[10], r[2], r[11]] for r in rich[::st]]
        # rolling hit-rate over the last 50 rounds
        win, roll = 50, []
        for i in range(win, len(rich) + 1, max(1, len(rich) // 200)):
            roll.append([rich[i - 1][0] + 1, sum(x[2] for x in rich[i - win:i]) / win])
        out["rolling"] = roll
        # calibration: predicted P(up) vs how often price really went up
        bins = []
        for lo_b in (0.0, 0.2, 0.4, 0.6, 0.8):
            b = [r for r in rich if lo_b <= r[11] < lo_b + 0.2 or (lo_b == 0.8 and r[11] == 1.0)]
            if b:
                bins.append([lo_b + 0.1, sum(1 for r in b if r[10] > r[6]) / len(b), len(b),
                             sum(r[11] for r in b) / len(b)])
        out["calibration"] = bins
    return out


def _rounds_from_csv(p: Path) -> list:
    import pandas as pd
    try:
        df = pd.read_csv(p)
    except (OSError, ValueError):
        return []
    need = {"engine", "anchor", "last_close", "median", "p10", "p90", "actual", "direction_hit", "prob_up",
            "coverage", "crps_skill"}
    if not need <= set(df.columns):
        return []
    df["i"] = df.groupby("engine").cumcount()
    ts = (pd.to_datetime(df["anchor"], utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds().astype(int)
    return [[int(r.i), r.engine, int(bool(r.direction_hit)), float(r.coverage), float(r.crps_skill), int(t),
             float(r.last_close), float(r.median), float(r.p10), float(r.p90), float(r.actual), float(r.prob_up)]
            for r, t in zip(df.itertuples(index=False), ts)]


def collect(d: Path = DIR) -> dict:
    plan = _load(d / "plan.json") or []
    names = [t["name"] for t in plan]
    for p in sorted(d.glob("*.json")):
        stem = p.name[:-len(".progress.json")] if p.name.endswith(".progress.json") else p.stem
        if stem != "plan" and stem not in names:
            names.append(stem)
    tests = []
    for name in names:
        info = next((t for t in plan if t["name"] == name), {"name": name})
        final = _load(d / f"{name}.json")
        prog = _load(d / f"{name}.progress.json")
        failed = (d / f"{name}.failed").exists()
        t = {"name": name, "plan": info, "status": "queued", "done": 0, "total": None, "eta_s": None,
             "elapsed_s": None, "updated": None, "meta": {}, "engines": {}}
        rounds = (prog or {}).get("rounds", [])
        if (not rounds or len(rounds[0]) < 12) and (d / f"{name}.csv").exists():
            rounds = _rounds_from_csv(d / f"{name}.csv") or rounds  # older runs: rebuild detail from the CSV
        if prog or rounds:
            prog = prog or {"status": "done", "updated": time.time()}
            t.update({k: prog.get(k) for k in ("done", "total", "eta_s", "elapsed_s", "updated", "meta", "last_anchor")})
            t["done"] = t["done"] or 0
            t["meta"] = t["meta"] or {}
            engines = sorted({r[1] for r in rounds}, key=lambda e: e == "bootstrap")
            t["engines"] = {e: _engine_stats(rounds, e) for e in engines}
            stale = time.time() - (prog.get("updated") or 0) > STALE_S
            t["status"] = "done" if prog.get("status") == "done" else "stopped" if stale else "running"
        if final:
            t["status"] = "done"
            cfg = final.get("_config", {})
            t["meta"] = t["meta"] or cfg
            for e, m in final.items():
                if e.startswith("_"):
                    continue
                s = t["engines"].setdefault(e, {"curve": []})
                s.update({"n": m.get("rounds"), "hit_rate": m.get("hit_rate"), "ci_low": m.get("ci_low"),
                          "ci_high": m.get("ci_high"), "p_value": m.get("p_value"), "coverage": m.get("coverage"),
                          "crps_skill": m.get("crps_skill"), "mae_skill": m.get("skill"),
                          "verdict": m.get("verdict"), "seconds": m.get("avg_seconds")})
            t["done"] = t["total"] = max((m.get("rounds") or 0) for e, m in final.items() if not e.startswith("_"))
        if failed and t["status"] != "done":
            t["status"] = "failed"
        tests.append(t)
    return {"dir": str(d), "now": time.time(), "tests": tests}


@app.get("/")
def page():
    return FileResponse(ROOT / "web" / "backtests.html")


@app.get("/api/backtests")
def api():
    return collect()
