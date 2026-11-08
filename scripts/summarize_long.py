"""Print one table from results/backtest/long/*.json (to paste back into the chat)."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/backtest/long")
rows = []
for f in sorted(d.glob("*.json")):
    res = json.loads(f.read_text())
    cfg = res.pop("_config", {})
    for eng, m in res.items():
        rows.append((f.stem, eng, m["rounds"], m["hit_rate"], m.get("ci_low", 0), m.get("ci_high", 1),
                     m.get("p_value", 1), m["coverage"], m["crps_skill"], m["skill"], m.get("verdict", "-"),
                     cfg.get("first", "")[:10], cfg.get("last", "")[:10]))
print(f"{'prueba':<26}{'motor':<14}{'n':>5} {'acierto':>8} {'IC95%':>15} {'p':>7} {'cober':>6} {'CRPS':>7} "
      f"{'MAE':>7}  veredicto        periodo")
for r in rows:
    print(f"{r[0]:<26}{r[1]:<14}{r[2]:>5} {r[3]:>8.1%} [{r[4]:>5.1%},{r[5]:>6.1%}] {r[6]:>7.4f} {r[7]:>6.1%} "
          f"{r[8]:>+7.1%} {r[9]:>+7.1%}  {r[10]:<16} {r[11]}..{r[12]}")
