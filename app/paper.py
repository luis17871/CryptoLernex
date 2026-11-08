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


def _save_to(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, path)


def _save(st: dict) -> None:
    _save_to(STATE, st)


def _synthetic_frame(symbol: str) -> pd.DataFrame:
    """Offline stand-in (KRONOS_PAPER_SOURCE=synthetic) for demos and tests."""
    from .data import fetch_synthetic
    d = fetch_synthetic(symbol, "1d", 1500).rename(columns={"timestamps": "date"})
    rng = np.random.default_rng(sum(map(ord, symbol)))
    d["taker_buy"] = d["volume"] * rng.uniform(0.4, 0.6, len(d))
    d["funding"] = rng.normal(1e-4, 2e-4, len(d))
    d["premium"] = rng.normal(0, 1e-3, len(d))
    return d


def _fresh_frame(symbol: str) -> pd.DataFrame:
    from . import strategy

    if os.environ.get("KRONOS_PAPER_SOURCE") == "synthetic":
        return _synthetic_frame(symbol)

    end = pd.Timestamp.now("UTC").floor("D").strftime("%Y-%m-%dT%H:%MZ")
    for old in strategy.CACHE.glob(f"deriv_{symbol}_*.csv"):  # keep only today's download
        if end.replace(":", "") not in old.name:
            old.unlink(missing_ok=True)
    return strategy.deriv_frame(symbol, end)


def _model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _advance(symbol: str, s: dict, d: pd.DataFrame, i: int, pu: float, top: list) -> dict:
    """One trading day for one symbol: mark position, controls, new decision. Shared by live and simulation."""
    day = str(d["date"].iloc[i])[:10]
    close, high, low = (float(d[c].iloc[i]) for c in ("close", "high", "low"))
    sma50 = float(d["close"].iloc[i - 49: i + 1].mean())
    trend = 1 if close > sma50 else -1
    sigma = float(np.log(d["close"]).diff().iloc[i - 29: i + 1].std())
    eq, teq, beq = s.get("equity", 1.0), s.get("trend_equity", 1.0), s.get("bh_equity", 1.0)
    prev_close = s.get("prev_close")
    pos = s.get("position")
    trades = s.setdefault("trades", [])
    if pos and prev_close:
        side = pos["side"]
        stop_hit = (side > 0 and low <= pos["stop"]) or (side < 0 and high >= pos["stop"])
        px = pos["stop"] if stop_hit else close
        eq *= 1 + side * (px / prev_close - 1)
        pos["days"] += 1
        if stop_hit or pos["days"] >= H:
            eq *= 1 - COST
            trades.append({"symbol": symbol, "side": side, "open": pos["date"], "close": day, "entry": pos["entry"],
                           "exit": px, "ret": side * (px / pos["entry"] - 1) - COST,
                           "reason": "stop" if stop_hit else "5 días", "p_up": pos["p_up"]})
            pos = None
    if prev_close:
        beq *= close / prev_close
        if s.get("trend_pos", 0) > 0:
            teq *= close / prev_close
    new_trend_pos = 1 if trend > 0 else 0
    if new_trend_pos != s.get("trend_pos", 0) and prev_close:
        teq *= 1 - COST / 2
    if pos is None:
        side = 1 if pu > 0.5 else -1
        if abs(pu - 0.5) >= STRONG and side == trend:
            k = STOP_K * sigma * math.sqrt(H)
            pos = {"side": side, "entry": close, "stop": close * (1 - k) if side > 0 else close * (1 + k),
                   "date": day, "days": 0, "p_up": pu}
            signal = "abre largo" if side > 0 else "abre corto"
        else:
            signal = "fuera: señal débil" if abs(pu - 0.5) < STRONG else "fuera: contra la tendencia"
    else:
        signal = "mantiene " + ("largo" if pos["side"] > 0 else "corto")
    s.setdefault("history", []).append({"day": day, "close": close, "p_up": pu, "trend": trend, "signal": signal,
                                        "equity": eq, "trend_equity": teq, "bh_equity": beq})
    s.update({"last_day": day, "prev_close": close, "position": pos, "equity": eq, "trend_equity": teq,
              "bh_equity": beq, "trend_pos": new_trend_pos, "p_up": pu, "trend": trend, "signal": signal,
              "top_features": top})
    return s


def _xy(d: pd.DataFrame):
    from .strategy import features
    x = features(d)
    y = (d["close"].shift(-H) > d["close"]).astype(float)
    y[d["close"].shift(-H).isna()] = np.nan
    return x, y


def _fit(x, y, upto: int):
    """Train on rows whose 5-day outcome is known at day ``upto`` (no look-ahead)."""
    rows = slice(0, max(0, upto - H + 1))
    xt, yt = x.iloc[rows], y.iloc[rows]
    ok = yt.notna() & xt.notna().all(axis=1)
    m = _model()
    m.fit(xt[ok], yt[ok])
    top = [f for f, _ in sorted(zip(x.columns, m.feature_importances_), key=lambda t: -t[1])[:5]]
    return m, top


def step_symbol(symbol: str, s: dict) -> dict:
    d = _fresh_frame(symbol)
    i = len(d) - 1
    if s.get("last_day") == str(d["date"].iloc[i])[:10]:
        return s
    x, y = _xy(d)
    m, top = _fit(x, y, i)
    return _advance(symbol, s, d, i, float(m.predict_proba(x.iloc[[i]])[0, 1]), top)


SIM = STATE.with_name("paper_sim.json")


def simulate(days: int = 365, symbols: list[str] | None = None) -> dict:
    """Replay the exact live rules over the last ``days`` days, retraining every 30 days."""
    symbols = symbols or SYMBOLS
    st = {"created": time.time(), "mode": "simulation", "days": days, "running": True, "done": 0,
          "total": len(symbols), "symbols": {}}
    _save_to(SIM, st)
    for sym in symbols:
        s: dict = {}
        try:
            d = _fresh_frame(sym)
            x, y = _xy(d)
            start = max(250, len(d) - days)
            m, top, trained = None, [], -999
            for i in range(start, len(d)):
                if i - trained >= 30:
                    m, top = _fit(x, y, i)
                    trained = i
                xi = x.iloc[[i]]
                pu = 0.5 if xi.isna().any(axis=1).iloc[0] else float(m.predict_proba(xi)[0, 1])
                s = _advance(sym, s, d, i, pu, top)
        except Exception as e:  # noqa: BLE001
            s["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        st["symbols"][sym] = s
        st["done"] += 1
        st["updated"] = time.time()
        _save_to(SIM, st)
    st["running"] = False
    _save_to(SIM, st)
    return st


def sim_state() -> dict:
    try:
        return json.loads(SIM.read_text())
    except (OSError, ValueError):
        return {"mode": "simulation", "symbols": {}, "running": False}


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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sim":
        out = simulate(int(sys.argv[2]) if len(sys.argv) > 2 else 365)
    else:
        out = step()
    for k, v in out["symbols"].items():
        print(k, v.get("signal"), f"P(up) {v.get('p_up', 0):.0%}", f"equity {v.get('equity', 1):.4f}", v.get("error", ""))
