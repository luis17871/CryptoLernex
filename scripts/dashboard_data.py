"""Build the CSV files behind the validation dashboard from results/.

    python scripts/dashboard_data.py OUT_DIR

Writes forward.csv (every live round, graded or not), backtest.csv (the 2-week walk-forward)
and sweep.csv (one row per configuration and engine, finished or pending).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.data import MarketData  # noqa: E402
from app.forward import grade, load_rounds  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "dashboard_data")
out.mkdir(parents=True, exist_ok=True)

# ---- forward test
rounds = load_rounds("BTCUSDT", "1h")
graded = pd.DataFrame()
if rounds:
    first = min(r["anchor_ts"] for r in rounds)
    df = MarketData("auto").candles("BTCUSDT", "1h", min(int((time.time() - first) // 3600) + 30, 5000))
    graded = grade(rounds, df)
rows = []
for r in rounds:
    anchor = pd.to_datetime(r["anchor_ts"], unit="s", utc=True)
    g = graded[(graded.engine == r["engine"]) & (graded.anchor == anchor)] if len(graded) else graded
    row = {"round": f"{anchor:%Y-%m-%d %H:%M} {r['engine']}", "engine": r["engine"],
           "anchor": anchor.strftime("%Y-%m-%dT%H:%MZ"), "last_close": round(r["last_close"], 2),
           "median": round(r["median"], 2), "p10": round(r["p10"], 2), "p90": round(r["p90"], 2),
           "prob_up": round(r["prob_up"], 3), "regime": r["regime"], "status": "graded" if len(g) else "open"}
    if len(g):
        x = g.iloc[0]
        row |= {"actual": round(x.actual, 2), "direction_hit": int(x.direction_hit),
                "coverage": round(x.coverage, 3), "inside_final": int(x.inside_final),
                "crps_skill": round(x.crps_skill, 4), "mae_skill": round(x.skill, 4)}
    rows.append(row)
cols = ["round", "engine", "anchor", "last_close", "median", "p10", "p90", "prob_up", "regime", "status",
        "actual", "direction_hit", "coverage", "inside_final", "crps_skill", "mae_skill"]
pd.DataFrame(rows, columns=cols).to_csv(out / "forward.csv", index=False)

# ---- backtest
b = pd.read_csv(ROOT / "results/backtest/btc_1h_small.csv")
b["anchor"] = pd.to_datetime(b["anchor"]).dt.strftime("%Y-%m-%dT%H:%MZ")
b["round"] = b["anchor"] + " " + b["engine"]
b = b.rename(columns={"skill": "mae_skill"})
b["direction_hit"] = b["direction_hit"].astype(int)
b["inside_final"] = b["inside_final"].astype(int)
b[["round", "engine", "anchor", "last_close", "median", "p10", "p90", "prob_up", "regime", "actual",
   "direction_hit", "coverage", "inside_final", "crps_skill", "mae_skill"]].round(4).to_csv(out / "backtest.csv",
                                                                                          index=False)

# ---- sweep
CONFIGS = [  # (id, label, symbol, model, T, top_p)
    ("viral_small_T1.0_p0.9", "Panel viral: small, T 1.0, top_p 0.9", "BTCUSDT", "kronos-small", 1.0, 0.9),
    ("small_T1.0_p1.0", "small, T 1.0, top_p 1.0", "BTCUSDT", "kronos-small", 1.0, 1.0),
    ("small_T1.5_p1.0", "small, T 1.5, top_p 1.0", "BTCUSDT", "kronos-small", 1.5, 1.0),
    ("mini_T1.0_p0.9", "mini, T 1.0, top_p 0.9", "BTCUSDT", "kronos-mini", 1.0, 0.9),
    ("small_eth_T1.0_p0.9", "small en ETH, T 1.0, top_p 0.9", "ETHUSDT", "kronos-small", 1.0, 0.9),
    ("base_T1.0_p0.9", "base, T 1.0, top_p 0.9", "BTCUSDT", "kronos-base", 1.0, 0.9),
]
srows = []
for cid, label, sym, model, t, p in CONFIGS:
    f = ROOT / ("results/backtest/btc_1h_small.json" if cid.startswith("viral")
                else f"results/backtest/sweep/{cid}.json")
    res = json.loads(f.read_text()) if f.exists() else {}
    for eng in (model, "bootstrap"):
        m = res.get(eng, {})
        srows.append({"row": f"{cid}|{eng}", "config": label, "symbol": sym, "engine": eng,
                      "kind": "kronos" if eng != "bootstrap" else "base", "status": "done" if m else "pending",
                      "rounds": m.get("rounds"), "hit_rate": m.get("hit_rate"), "coverage": m.get("coverage"),
                      "crps_skill": m.get("crps_skill"), "mae_skill": m.get("skill")})
pd.DataFrame(srows).round(4).to_csv(out / "sweep.csv", index=False)
print("wrote", *(p.name for p in out.glob("*.csv")))
