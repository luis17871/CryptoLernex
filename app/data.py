"""Market data: closed OHLCV candles from public exchange APIs (no account needed).

Every source returns a DataFrame with columns
``timestamps, open, high, low, close, volume, amount`` (UTC, oldest first) containing
**closed** candles only. ``synthetic`` is a deterministic offline market so the whole
pipeline (forecast -> live scoring) works without network access.
"""
from __future__ import annotations

import time
from functools import lru_cache

import httpx
import numpy as np
import pandas as pd

from .config import INTERVALS

COLS = ["open", "high", "low", "close", "volume", "amount"]
UA = {"User-Agent": "kronos-lab/1.0"}


class DataError(RuntimeError):
    pass


def _finish(df: pd.DataFrame, interval: str, now: float | None = None) -> pd.DataFrame:
    sec = INTERVALS[interval]
    now = time.time() if now is None else now
    df = df.sort_values("timestamps").drop_duplicates("timestamps")
    open_s = (df["timestamps"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    df = df[(open_s + sec) <= now].copy()  # keep closed candles only
    df[COLS] = df[COLS].astype("float64")
    return df.reset_index(drop=True)


def _get(url: str, params: dict) -> object:
    r = httpx.get(url, params=params, headers=UA, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- Binance
def fetch_binance(symbol: str, interval: str, limit: int, end_ms: int | None = None) -> pd.DataFrame:
    rows: list = []
    hosts = ["https://api.binance.com", "https://data-api.binance.vision"]
    remaining, end = limit + 1, end_ms
    while remaining > 0:
        params = {"symbol": symbol, "interval": interval, "limit": min(1000, remaining)}
        if end is not None:
            params["endTime"] = end
        last_err = None
        for h in hosts:
            try:
                batch = _get(f"{h}/api/v3/klines", params)
                break
            except Exception as e:  # noqa: BLE001 - try the mirror
                last_err = e
        else:
            raise DataError(f"binance: {last_err}")
        if not batch:
            break
        rows = batch + rows
        remaining -= len(batch)
        end = batch[0][0] - 1
        if len(batch) < params["limit"]:
            break
    df = pd.DataFrame([[r[0], r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows],
                      columns=["ms", *COLS])
    df["timestamps"] = pd.to_datetime(df.pop("ms"), unit="ms", utc=True)
    return _finish(df, interval).tail(limit).reset_index(drop=True)


# ---------------------------------------------------------------- Bybit
_BYBIT_IV = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}


def fetch_bybit(symbol: str, interval: str, limit: int, end_ms: int | None = None) -> pd.DataFrame:
    rows: list = []
    remaining, end = limit + 1, end_ms
    while remaining > 0:
        params = {"category": "spot", "symbol": symbol, "interval": _BYBIT_IV[interval],
                  "limit": min(1000, remaining)}
        if end is not None:
            params["end"] = end
        data = _get("https://api.bybit.com/v5/market/kline", params)
        batch = data.get("result", {}).get("list", [])  # newest first
        if not batch:
            break
        rows = list(reversed(batch)) + rows
        remaining -= len(batch)
        end = int(batch[-1][0]) - 1
        if len(batch) < params["limit"]:
            break
    df = pd.DataFrame([[int(r[0]), *r[1:7]] for r in rows], columns=["ms", *COLS])
    df["timestamps"] = pd.to_datetime(df.pop("ms"), unit="ms", utc=True)
    return _finish(df, interval).tail(limit).reset_index(drop=True)


# ---------------------------------------------------------------- OKX
_OKX_IV = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}


def _okx_inst(symbol: str) -> str:
    for q in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if symbol.endswith(q) and len(symbol) > len(q):
            return f"{symbol[:-len(q)]}-{q}"
    return symbol


def fetch_okx(symbol: str, interval: str, limit: int, end_ms: int | None = None) -> pd.DataFrame:
    rows: list = []
    remaining, after = limit + 1, end_ms
    while remaining > 0:
        params = {"instId": _okx_inst(symbol), "bar": _OKX_IV[interval], "limit": min(100, remaining)}
        if after is not None:
            params["after"] = after
        batch = _get("https://www.okx.com/api/v5/market/history-candles", params).get("data", [])
        if not batch:
            break
        rows = list(reversed(batch)) + rows
        remaining -= len(batch)
        after = int(batch[-1][0])
        if len(batch) < params["limit"]:
            break
    df = pd.DataFrame([[int(r[0]), *r[1:6], r[7]] for r in rows], columns=["ms", *COLS])
    df["timestamps"] = pd.to_datetime(df.pop("ms"), unit="ms", utc=True)
    return _finish(df, interval).tail(limit).reset_index(drop=True)


# ---------------------------------------------------------------- Synthetic (offline)
_SYN_ORIGIN = 1767225600  # 2026-01-01 UTC
_SYN_BASE = {"BTC": 80000.0, "ETH": 3000.0, "SOL": 150.0}


@lru_cache(maxsize=32)
def _syn_series(symbol: str, interval: str, n: int) -> tuple[np.ndarray, ...]:
    """Regime-switching stochastic-volatility market, deterministic per (symbol, interval)."""
    seed = sum(map(ord, symbol + interval)) * 7919
    rng = np.random.default_rng(seed)
    sec = INTERVALS[interval]
    scale = np.sqrt(sec / 3600.0)
    regime_len = max(20, int(48 * 3600 / sec))
    n_reg = n // regime_len + 2
    drift = rng.normal(0, 0.00012, n_reg).repeat(regime_len)[:n] * scale**2
    vol_lvl = np.exp(rng.normal(np.log(0.006), 0.35, n_reg)).repeat(regime_len)[:n] * scale
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    # GARCH-like clustering on top of the regime volatility
    h = np.empty(n)
    h[0] = 1.0
    for i in range(1, n):
        h[i] = 0.05 + 0.10 * (z[i - 1] ** 2) * h[i - 1] + 0.85 * h[i - 1]
    shocks = drift + vol_lvl * np.sqrt(h) * z
    base = next((v for k, v in _SYN_BASE.items() if symbol.startswith(k)), 100.0)
    # weak mean reversion of log-price keeps the level realistic over long horizons
    kappa = 0.0015 * scale**2
    logp = np.empty(n)
    lp = 0.0
    for i in range(n):
        lp += shocks[i] - kappa * lp
        logp[i] = lp
    r = np.diff(np.concatenate([[0.0], logp]))
    close = base * np.exp(logp)
    open_ = np.concatenate([[base], close[:-1]])
    wick = np.abs(rng.normal(0, 1, (2, n))) * vol_lvl * np.sqrt(h) * 0.6
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    vol = np.exp(rng.normal(0, 0.4, n)) * (1 + 40 * np.abs(r)) * 1000 / base * 50
    return open_, high, low, close, vol


def fetch_synthetic(symbol: str, interval: str, limit: int, end_ms: int | None = None,
                    now: float | None = None) -> pd.DataFrame:
    sec = INTERVALS[interval]
    now = time.time() if now is None else now
    if end_ms is not None:
        now = min(now, end_ms / 1000 + sec)
    n_closed = int((now - _SYN_ORIGIN) // sec)
    if n_closed <= 0:
        raise DataError("synthetic: clock before origin")
    # Index i is the candle opening at _SYN_ORIGIN + i*sec; history before the origin is
    # simulated by shifting the series so that there is always enough lookback.
    pad = 3000
    total = n_closed + pad
    bucket = (total // 5000 + 1) * 5000  # cache-friendly size; the prefix is stable
    o, h, lo, c, v = (a[:total] for a in _syn_series(symbol, interval, bucket))
    start = max(0, total - limit)
    ts = _SYN_ORIGIN + (np.arange(start, total) - pad) * sec
    df = pd.DataFrame({"timestamps": pd.to_datetime(ts, unit="s", utc=True),
                       "open": o[start:], "high": h[start:], "low": lo[start:], "close": c[start:],
                       "volume": v[start:]})
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df[["timestamps", *COLS]]


SOURCES = {"binance": fetch_binance, "bybit": fetch_bybit, "okx": fetch_okx, "synthetic": fetch_synthetic}


class MarketData:
    """Picks the first working source when ``source='auto'`` and remembers it."""

    def __init__(self, source: str = "auto"):
        self.source = source
        self.active: str | None = None if source == "auto" else source
        self.errors: dict[str, str] = {}

    def candles(self, symbol: str, interval: str, limit: int, end_ms: int | None = None) -> pd.DataFrame:
        if interval not in INTERVALS:
            raise DataError(f"unsupported interval {interval}")
        order = [self.active] if self.active else ["binance", "bybit", "okx", "synthetic"]
        for name in order:
            try:
                df = SOURCES[name](symbol, interval, limit, end_ms)
                if len(df) == 0:
                    raise DataError("empty response")
                self.active = name
                return df
            except Exception as e:  # noqa: BLE001
                self.errors[name] = str(e)[:200]
        if self.source == "auto" and self.active:  # active source died -> rescan next call
            self.active = None
        raise DataError(f"no data source available: {self.errors}")
