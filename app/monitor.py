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
    return {"n": n, "hits": k, "hit_rate": k / n if n else None, "ci_low": lo, "ci_high": hi,
            "p_value": analytics.binom_p_greater(k, n), "coverage": sum(r[3] for r in rs) / n if n else None,
            "crps_skill": sum(r[4] for r in rs) / n if n else None, "verdict": verdict, "curve": curve}


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
        if prog:
            t.update({k: prog.get(k) for k in ("done", "total", "eta_s", "elapsed_s", "updated", "meta", "last_anchor")})
            engines = sorted({r[1] for r in prog.get("rounds", [])}, key=lambda e: e == "bootstrap")
            t["engines"] = {e: _engine_stats(prog["rounds"], e) for e in engines}
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
