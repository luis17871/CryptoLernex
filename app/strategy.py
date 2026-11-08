"""Trading experiments on top of the forecasts. Everything writes into results/backtest/long so the
monitor (app.monitor) shows it.

    python -m app.strategy trade  [--dir DIR]          # 1) PnL of every direction test (costs, stops, filters)
    python -m app.strategy xsec   --models kronos-small,chronos-2,timesfm-2.5   # 2) 20-coin weekly long/short
    python -m app.strategy deriv  --symbols BTCUSDT,SOLUSDT --horizons 5,10     # 4) LightGBM on derivatives data

(3, daily candles, is the regular backtest with --interval 1d; see scripts/windows/gpu_backtest.ps1.)
All experiments stop at --end so every model sees the same window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from . import analytics
from .config import INTERVALS, ROOT
from .data import MarketData

OUT = Path(os.environ.get("KRONOS_BACKTEST_DIR", ROOT / "results" / "backtest" / "long"))
CACHE = ROOT / "results" / "cache"
COST = 0.0014  # round trip: 2 x 0.05% taker fee + 2 x 0.02% slippage (Binance USDT-M futures)
UNIVERSE = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "TRX", "LTC", "BCH", "ATOM",
            "NEAR", "UNI", "ETC", "XLM", "FIL", "APT"]


def _end_ms(end: str | None) -> int | None:
    if not end:
        return None
    ts = pd.Timestamp(end)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return int(ts.timestamp() * 1000)


def candles(symbol: str, interval: str, n: int, end: str | None) -> pd.DataFrame:
    """Cached candles (the same window is re-read many times)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{symbol}_{interval}_{n}_{(end or 'now').replace(':', '')}.csv"
    if f.exists() and end:
        df = pd.read_csv(f)
        df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
        return df
    df = MarketData("auto").candles(symbol, interval, n, _end_ms(end))
    df.to_csv(f, index=False)
    return df


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, path)


def perf(rets: np.ndarray, periods_per_year: float) -> dict:
    """Metrics of a series of per-period returns (0 = flat period)."""
    rets = np.asarray(rets, dtype=float)
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    traded = rets[rets != 0]
    wins, losses = traded[traded > 0], traded[traded < 0]
    years = len(rets) / periods_per_year if periods_per_year else 1
    sd = rets.std()
    return {
        "total_return": float(eq[-1] - 1) if len(eq) else 0.0,
        "cagr": float(eq[-1] ** (1 / years) - 1) if len(eq) and years > 0 and eq[-1] > 0 else -1.0,
        "sharpe": float(rets.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "max_dd": float((eq / peak - 1).min()) if len(eq) else 0.0,
        "trades": int(len(traded)), "exposure": float(len(traded) / len(rets)) if len(rets) else 0.0,
        "win_rate": float(len(wins) / len(traded)) if len(traded) else None,
        "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
    }


def _curve(ts: list, rets: np.ndarray, maxn: int = 300) -> list:
    eq = np.cumprod(1 + np.asarray(rets, dtype=float))
    st = max(1, len(eq) // maxn)
    return [[int(ts[i]), round(float(eq[i]), 5)] for i in range(0, len(eq), st)] + \
        ([[int(ts[-1]), round(float(eq[-1]), 5)]] if len(eq) else [])


# ============================================================== 1) trading simulation of direction tests
def simulate(rounds: pd.DataFrame, bars: pd.DataFrame, horizon: int, sec: int, cost: float = COST) -> dict:
    """rounds: one engine's rows (anchor, last_close, actual, prob_up), non-overlapping.
    Variants: every signal / strong signals / + trend filter / + volatility stop."""
    bars = bars.reset_index(drop=True)
    ts = (bars["timestamps"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().astype(int).to_numpy()
    close, high, low = bars["close"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    per_day = max(1, 86400 // sec)
    sma_n, vol_n = 50 * per_day, 30 * per_day
    pos = {int(t): i for i, t in enumerate(ts)}
    out = {k: [] for k in ("todas", "fuertes", "fuertes+tendencia", "fuertes+tendencia+stop")}
    bh, stamps = [], []
    for r in rounds.itertuples(index=False):
        a = int(pd.Timestamp(r.anchor).timestamp())
        i = pos.get(a)
        if i is None or i + horizon >= len(close):
            continue
        entry, final = close[i], close[i + horizon]
        p = float(r.prob_up)
        side = 1 if p > 0.5 else -1 if p < 0.5 else 0
        strong = abs(p - 0.5) >= 0.15
        trend = 1 if close[i] > close[max(0, i - sma_n + 1): i + 1].mean() else -1
        sigma = logret[max(1, i - vol_n + 1): i + 1].std() or 1e-4
        stop_d = 1.5 * sigma * math.sqrt(horizon)
        plain = side * (final / entry - 1) - cost if side else 0.0
        # stop: walk the bars of the holding period; a touched stop exits at the stop price
        exit_px = final
        if side:
            stop = entry * (1 - stop_d) if side > 0 else entry * (1 + stop_d)
            for j in range(i + 1, i + horizon + 1):
                if (side > 0 and low[j] <= stop) or (side < 0 and high[j] >= stop):
                    exit_px = stop
                    break
        stopped = side * (exit_px / entry - 1) - cost if side else 0.0
        out["todas"].append(plain)
        out["fuertes"].append(plain if strong else 0.0)
        ok = strong and side == trend
        out["fuertes+tendencia"].append(plain if ok else 0.0)
        out["fuertes+tendencia+stop"].append(stopped if ok else 0.0)
        bh.append(final / entry - 1)
        stamps.append(a)
    ppy = 365 * 86400 / (horizon * sec)
    res = {k: perf(np.array(v), ppy) | {"curve": _curve(stamps, v)} for k, v in out.items() if v}
    res["_buyhold"] = perf(np.array(bh), ppy) | {"curve": _curve(stamps, bh)} if bh else {}
    return res


def cmd_trade(a) -> None:
    d = Path(a.dir)
    for f in sorted(d.glob("*.csv")):
        name = f.stem
        meta_f = d / f"{name}.json"
        if name.startswith(("trade_", "xsec")) or not meta_f.exists():
            continue
        cfg = json.loads(meta_f.read_text(encoding="utf-8-sig")).get("_config", {})
        sym, iv, h = cfg.get("symbol"), cfg.get("interval", "1h"), int(cfg.get("horizon", 24))
        if not sym or iv not in INTERVALS:
            continue
        df = pd.read_csv(f)
        if "prob_up" not in df.columns:
            continue
        first = pd.Timestamp(df["anchor"].min())
        sec = INTERVALS[iv]
        need = int((pd.Timestamp(cfg.get("last") or a.end) - first).total_seconds() // sec) + 60 * 86400 // sec + h + 10
        try:
            bars = candles(sym, iv, min(need, 20000), cfg.get("last") or a.end)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: no candles ({e})")
            continue
        doc = {"_kind": "trade", "test": name, "meta": cfg | {"cost": a.cost}, "engines": {}}
        for eng, g in df.groupby("engine"):
            doc["engines"][eng] = simulate(g.sort_values("anchor"), bars, h, sec, a.cost)
        _write(d / f"trade_{name}.json", doc)
        best = max(((e, v, m["total_return"]) for e, s in doc["engines"].items() for v, m in s.items()
                    if not v.startswith("_")), key=lambda x: x[2], default=None)
        print(f"trade_{name}: best {best}")


# ============================================================== 2) cross-sectional weekly long/short
def cmd_xsec(a) -> None:
    from .engine import make_engine

    models = [m for m in a.models.split(",") if m]
    hold, weeks, ctx = 7, a.weeks, 512
    data = {}
    for c in UNIVERSE:
        try:
            df = candles(f"{c}USDT", "1d", ctx + weeks * hold + 40, a.end)
            if len(df) > 120:
                data[c] = df.reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            print(f"{c}: skipped ({e})")
    end_ts = max(df["timestamps"].iloc[-1] for df in data.values())
    rebal = [end_ts - pd.Timedelta(days=hold * (k + 1)) for k in range(weeks)][::-1]
    engines = {m: make_engine(m, f"weights/{m}" if m.startswith("kronos") and Path(f"weights/{m}").exists() else "",
                              a.device, 50, ctx) for m in models}
    names = list(engines) + ["momentum_28d", "reversal_7d"]
    rets = {n: [] for n in names}
    ics = {n: [] for n in names}
    stamps = []
    prog = OUT / "xsec.progress.json"
    t0 = time.time()
    for w, t in enumerate(rebal):
        preds = {n: {} for n in names}
        real = {}
        for c, df in data.items():
            idx = df.index[df["timestamps"] == t]
            if not len(idx):
                continue
            i = int(idx[0])
            if i < 60 or i + hold >= len(df):
                continue
            hist = df.iloc[max(0, i - ctx + 1): i + 1].reset_index(drop=True)
            last = float(hist["close"].iloc[-1])
            real[c] = float(df["close"].iloc[i + hold] / last - 1)
            preds["momentum_28d"][c] = last / float(df["close"].iloc[i - 28]) - 1
            preds["reversal_7d"][c] = -(last / float(df["close"].iloc[i - 7]) - 1)
            y_ts = pd.Series(pd.date_range(t + pd.Timedelta(days=1), periods=hold, freq="D"))
            for m, eng in engines.items():
                if len(hist) < 64:
                    continue
                paths = eng.sample(hist, y_ts, a.samples, 1.0, 0.9, w)
                preds[m][c] = float(np.median(paths[:, -1, 3]) / last - 1)
        if len(real) < 12:
            continue
        stamps.append(int(t.timestamp()))
        for n in names:
            p = {c: v for c, v in preds[n].items() if c in real}
            if len(p) < 12:
                rets[n].append(0.0)
                continue
            order = sorted(p, key=p.get)
            k = min(5, len(order) // 4)
            longs, shorts = order[-k:], order[:k]
            r = np.mean([real[c] for c in longs]) - np.mean([real[c] for c in shorts]) - 2 * COST
            rets[n].append(float(r))
            ics[n].append(float(pd.Series(p).rank().corr(pd.Series({c: real[c] for c in p}).rank())))
        el = time.time() - t0
        _write(prog, {"status": "running", "done": w + 1, "total": len(rebal), "elapsed_s": el,
                      "eta_s": el / (w + 1) * (len(rebal) - w - 1), "updated": time.time()})
        print(f"[{w + 1}/{len(rebal)}] {t:%Y-%m-%d} " + " ".join(f"{n}:{rets[n][-1]:+.2%}" for n in names), flush=True)
    doc = {"_kind": "xsec", "meta": {"universe": list(data), "weeks": len(stamps), "hold_days": hold, "end": a.end,
                                     "cost": COST, "long": 5, "short": 5}, "strategies": {}}
    for n in names:
        r = np.array(rets[n])
        ic = np.array(ics[n])
        tstat = ic.mean() / (ic.std(ddof=1) / math.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else 0.0
        doc["strategies"][n] = perf(r, 52) | {"curve": _curve(stamps, r), "ic_mean": float(ic.mean()) if len(ic) else None,
                                               "ic_tstat": float(tstat), "ic_positive": float((ic > 0).mean()) if len(ic) else None}
    _write(OUT / "xsec.json", doc)
    _write(prog, {"status": "done", "done": len(rebal), "total": len(rebal), "updated": time.time()})
    for n, m in doc["strategies"].items():
        print(f"{n:<14} total {m['total_return']:+.1%} sharpe {m['sharpe']:.2f} IC {m['ic_mean']:+.3f} (t {m['ic_tstat']:.2f})")


# ============================================================== 4) derivatives features + LightGBM
FAPI = "https://fapi.binance.com"


def _fapi(path: str, params: dict) -> list:
    r = httpx.get(FAPI + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _paged(path: str, params: dict, start_ms: int, end_ms: int, step_key="startTime", limit=1000) -> list:
    rows, cur = [], start_ms
    while cur < end_ms:
        batch = _fapi(path, params | {step_key: cur, "endTime": end_ms, "limit": limit})
        if not batch:
            break
        rows += batch
        last = batch[-1][0] if isinstance(batch[-1], list) else batch[-1]["fundingTime"]
        if last + 1 <= cur:
            break
        cur = last + 1
        if len(batch) < limit:
            break
        time.sleep(0.15)
    return rows


def deriv_frame(symbol: str, end: str) -> pd.DataFrame:
    """Daily futures klines + taker flow + funding + premium (basis), cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"deriv_{symbol}_{end.replace(':', '')}.csv"
    if f.exists():
        d = pd.read_csv(f, parse_dates=["date"])
        return d
    end_ms, start_ms = _end_ms(end), int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    k = _paged("/fapi/v1/klines", {"symbol": symbol, "interval": "1d"}, start_ms, end_ms)
    kl = pd.DataFrame([[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])] for r in k],
                      columns=["t", "open", "high", "low", "close", "volume", "taker_buy"])
    p = _paged("/fapi/v1/premiumIndexKlines", {"symbol": symbol, "interval": "1d"}, start_ms, end_ms)
    pr = pd.DataFrame([[r[0], float(r[4])] for r in p], columns=["t", "premium"])
    fr = _paged("/fapi/v1/fundingRate", {"symbol": symbol}, start_ms, end_ms)
    fu = pd.DataFrame([[r["fundingTime"], float(r["fundingRate"])] for r in fr], columns=["t", "funding"])
    fu["date"] = pd.to_datetime(fu["t"], unit="ms", utc=True).dt.floor("D")
    fu = fu.groupby("date", as_index=False)["funding"].sum()
    d = kl.merge(pr, on="t", how="left")
    d["date"] = pd.to_datetime(d["t"], unit="ms", utc=True)
    d = d.merge(fu, on="date", how="left").drop(columns="t")
    d[["premium", "funding"]] = d[["premium", "funding"]].fillna(0.0)
    d = d[d["date"] + pd.Timedelta(days=1) <= pd.Timestamp(end_ms, unit="ms", tz="UTC")].reset_index(drop=True)
    d.to_csv(f, index=False)
    return d


def features(d: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=d.index)
    lc = np.log(d["close"])
    r = lc.diff()
    for n in (1, 3, 7, 14, 30):
        x[f"ret_{n}"] = lc - lc.shift(n)
    x["vol_7"], x["vol_30"] = r.rolling(7).std(), r.rolling(30).std()
    x["vol_ratio"] = x["vol_7"] / x["vol_30"]
    x["dist_sma50"] = d["close"] / d["close"].rolling(50).mean() - 1
    x["dist_sma200"] = d["close"] / d["close"].rolling(200).mean() - 1
    x["volume_z"] = (d["volume"] - d["volume"].rolling(30).mean()) / d["volume"].rolling(30).std()
    tb = d["taker_buy"] / d["volume"].replace(0, np.nan)
    x["taker_buy_1"], x["taker_buy_7"] = tb, tb.rolling(7).mean()
    x["funding_1"], x["funding_7"] = d["funding"], d["funding"].rolling(7).sum()
    x["funding_z"] = (d["funding"].rolling(7).mean() - d["funding"].rolling(90).mean()) / d["funding"].rolling(90).std()
    x["premium_1"], x["premium_7"] = d["premium"], d["premium"].rolling(7).mean()
    x["range_1"] = (d["high"] - d["low"]) / d["close"]
    return x


def cmd_deriv(a) -> None:
    import lightgbm as lgb

    for symbol in a.symbols.split(","):
        try:
            d = deriv_frame(symbol, a.end)
        except Exception as e:  # noqa: BLE001
            print(f"{symbol}: futures data not available ({e})")
            continue
        x = features(d)
        for h in (int(v) for v in a.horizons.split(",")):
            y = (d["close"].shift(-h) > d["close"]).astype(float)
            y[d["close"].shift(-h).isna()] = np.nan
            start = 400
            anchors = list(range(len(d) - h - 1, start, -h))[::-1]
            model, trained_at, rows, mom_rows = None, -999, [], []
            sigma = np.log(d["close"]).diff().rolling(30).std()
            for i in anchors:
                if model is None or i - trained_at >= 30:
                    tr = slice(0, i - h + 1)  # only rows whose outcome is known at time i
                    xt, yt = x.iloc[tr], y.iloc[tr]
                    ok = yt.notna() & xt.notna().all(axis=1)
                    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
                    model.fit(xt[ok], yt[ok])
                    trained_at = i
                xi = x.iloc[[i]]
                if xi.isna().any(axis=1).iloc[0]:
                    continue
                pu = float(model.predict_proba(xi)[0, 1])
                last, act = float(d["close"].iloc[i]), float(d["close"].iloc[i + h])
                s = float(sigma.iloc[i]) * math.sqrt(h)
                base = {"anchor": str(d["date"].iloc[i]), "last_close": last, "actual": act, "median": last,
                        "p10": last * math.exp(-1.2816 * s), "p90": last * math.exp(1.2816 * s), "regime": "-",
                        "coverage": float(last * math.exp(-1.2816 * s) <= act <= last * math.exp(1.2816 * s)),
                        "crps_skill": 0.0, "skill": 0.0, "seconds": 0.0}
                base["inside_final"] = bool(base["coverage"])
                rows.append(base | {"engine": "lightgbm", "prob_up": pu,
                                    "direction_hit": int((pu > 0.5) == (act > last))})
                m7 = float(x["ret_7"].iloc[i])
                mom_rows.append(base | {"engine": "momentum_7d", "prob_up": 0.5 + 0.2 * np.sign(m7),
                                        "direction_hit": int((m7 > 0) == (act > last))})
            if not rows:
                continue
            name = f"lgbm_{symbol.replace('USDT', '')}_d{h}"
            df = pd.DataFrame(rows + mom_rows)
            df.to_csv(OUT / f"{name}.csv", index=False)
            out = {}
            for eng, g in df.groupby("engine"):
                k, n = int(g["direction_hit"].sum()), len(g)
                lo, hi = analytics.wilson(k, n)
                out[eng] = {"rounds": n, "hit_rate": k / n, "coverage": float(g["coverage"].mean()), "crps_skill": 0.0,
                            "skill": 0.0, "avg_seconds": 0.0, "hits": k, "ci_low": lo, "ci_high": hi,
                            "p_value": analytics.binom_p_greater(k, n), "independent": True,
                            "verdict": "ventaja" if lo > 0.5 else "peor que azar" if hi < 0.5 else "no concluyente"}
            imp = sorted(zip(x.columns, model.feature_importances_), key=lambda t: -t[1])[:8]
            out["_config"] = {"symbol": symbol, "interval": "1d", "model": "lightgbm", "horizon": h, "step": h,
                              "first": str(d["date"].iloc[anchors[0]]), "last": str(d["date"].iloc[-1]),
                              "features": [f for f, _ in imp]}
            _write(OUT / f"{name}.json", out)
            print(f"{name}: lightgbm {out['lightgbm']['hit_rate']:.1%} [{out['lightgbm']['ci_low']:.0%}-"
                  f"{out['lightgbm']['ci_high']:.0%}] vs momentum {out.get('momentum_7d', {}).get('hit_rate', 0):.1%}"
                  f" | top features {[f for f, _ in imp[:4]]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["trade", "xsec", "deriv"])
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--end", default="2026-11-08T00:00Z")
    ap.add_argument("--cost", type=float, default=COST)
    ap.add_argument("--models", default="kronos-small,chronos-2,timesfm-2.5")
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--device", default="")
    ap.add_argument("--symbols", default="BTCUSDT,SOLUSDT")
    ap.add_argument("--horizons", default="5,10")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"trade": cmd_trade, "xsec": cmd_xsec, "deriv": cmd_deriv}[a.cmd](a)


if __name__ == "__main__":
    main()
