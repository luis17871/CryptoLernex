"""Live paper trading of the LightGBM + derivatives strategy (no real money).

Once a day, after the daily candle closes, for every symbol:
  1. download daily futures candles, funding, premium and taker flow (Binance, public);
  2. train LightGBM on all history whose 5-day outcome is already known;
  3. if flat and the signal is strong (P(up) >= 65% or <= 35%) and agrees with the trend
     (price vs 50-day average), open a paper position for 5 days with a 1.5-sigma stop;
  4. mark everything to market. Controls run alongside: trend-only and buy & hold.

State lives in data/paper.json. Costs: 0.14% per round trip (fees + slippage).
    python -m app.paper            # one step now
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

STATE = Path(os.environ.get("KRONOS_PAPER_STATE", ROOT / "data" / "paper.json"))
SYMBOLS = [s for s in os.environ.get("KRONOS_PAPER_SYMBOLS", "BTCUSDT,SOLUSDT").split(",") if s]
H, COST, STRONG, STOP_K = 5, 0.0014, 0.15, 1.5


def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"created": time.time(), "symbols": {}}


def _save(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, STATE)


def _fresh_frame(symbol: str) -> pd.DataFrame:
    from . import strategy

    end = pd.Timestamp.now("UTC").floor("D").strftime("%Y-%m-%dT%H:%MZ")
    for old in strategy.CACHE.glob(f"deriv_{symbol}_*.csv"):  # keep only today's download
        if end.replace(":", "") not in old.name:
            old.unlink(missing_ok=True)
    return strategy.deriv_frame(symbol, end)


def step_symbol(symbol: str, s: dict) -> dict:
    import lightgbm as lgb

    from .strategy import features

    d = _fresh_frame(symbol)
    x = features(d)
    y = (d["close"].shift(-H) > d["close"]).astype(float)
    y[d["close"].shift(-H).isna()] = np.nan
    i = len(d) - 1
    day = str(d["date"].iloc[i])[:10]
    if s.get("last_day") == day:
        return s
    ok = y.notna() & x.notna().all(axis=1)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    model.fit(x[ok], y[ok])
    pu = float(model.predict_proba(x.iloc[[i]])[0, 1])
    close, high, low = (float(d[c].iloc[i]) for c in ("close", "high", "low"))
    sma50 = float(d["close"].iloc[i - 49: i + 1].mean())
    trend = 1 if close > sma50 else -1
    sigma = float(np.log(d["close"]).diff().iloc[i - 29: i + 1].std())

    eq, teq, beq = s.get("equity", 1.0), s.get("trend_equity", 1.0), s.get("bh_equity", 1.0)
    prev_close = s.get("prev_close")
    pos = s.get("position")
    trades = s.setdefault("trades", [])
    # 1) mark open position: stop first, then time exit
    if pos and prev_close:
        side = pos["side"]
        stop_hit = (side > 0 and low <= pos["stop"]) or (side < 0 and high >= pos["stop"])
        px = pos["stop"] if stop_hit else close
        eq *= 1 + side * (px / prev_close - 1)
        pos["days"] += 1
        if stop_hit or pos["days"] >= H:
            eq *= 1 - COST
            ret = side * (px / pos["entry"] - 1) - COST
            trades.append({"symbol": symbol, "side": side, "open": pos["date"], "close": day, "entry": pos["entry"],
                           "exit": px, "ret": ret, "reason": "stop" if stop_hit else "5 días", "p_up": pos["p_up"]})
            pos = None
    # 2) controls
    if prev_close:
        beq *= close / prev_close
        if s.get("trend_pos", 0) > 0:
            teq *= close / prev_close
    new_trend_pos = 1 if trend > 0 else 0
    if new_trend_pos != s.get("trend_pos", 0):
        teq *= 1 - COST / 2
    # 3) new decision when flat
    signal = "sin señal"
    if pos is None:
        side = 1 if pu > 0.5 else -1
        if abs(pu - 0.5) >= STRONG and side == trend:
            stop = close * (1 - STOP_K * sigma * math.sqrt(H)) if side > 0 else close * (1 + STOP_K * sigma * math.sqrt(H))
            pos = {"side": side, "entry": close, "stop": stop, "date": day, "days": 0, "p_up": pu}
            signal = "abre largo" if side > 0 else "abre corto"
        else:
            signal = "fuera: señal débil" if abs(pu - 0.5) < STRONG else "fuera: contra la tendencia"
    else:
        signal = "mantiene " + ("largo" if pos["side"] > 0 else "corto")
    hist = s.setdefault("history", [])
    hist.append({"day": day, "close": close, "p_up": pu, "trend": trend, "sma50": sma50, "signal": signal,
                 "equity": eq, "trend_equity": teq, "bh_equity": beq})
    s.update({"last_day": day, "prev_close": close, "position": pos, "equity": eq, "trend_equity": teq,
              "bh_equity": beq, "trend_pos": new_trend_pos, "p_up": pu, "trend": trend, "signal": signal,
              "top_features": [f for f, _ in sorted(zip(x.columns, model.feature_importances_), key=lambda t: -t[1])[:5]]})
    return s


def step() -> dict:
    st = _load()
    for sym in SYMBOLS:
        try:
            st["symbols"][sym] = step_symbol(sym, st["symbols"].get(sym, {}))
            st["symbols"][sym].pop("error", None)
        except Exception as e:  # noqa: BLE001
            st["symbols"].setdefault(sym, {})["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    st["updated"] = time.time()
    _save(st)
    return st


def state() -> dict:
    return _load()


if __name__ == "__main__":
    out = step()
    for k, v in out["symbols"].items():
        print(k, v.get("signal"), f"P(up) {v.get('p_up', 0):.0%}", f"equity {v.get('equity', 1):.4f}", v.get("error", ""))
