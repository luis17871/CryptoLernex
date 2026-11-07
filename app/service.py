"""Orchestrator: forecast rounds, live scoring as candles close, event log."""
from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import analytics
from .config import INTERVALS, Settings
from .data import MarketData
from .engine import BootstrapEngine, Engine, build_engine
from .store import Store

log = logging.getLogger("kronos.service")


class EventLog:
    """Ring buffer of terminal-style events, fan-out to SSE subscribers."""

    def __init__(self, size: int = 400):
        self.buf: collections.deque = collections.deque(maxlen=size)
        self.subs: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.seq = 0

    def emit(self, tag: str, msg: str, level: str = "info") -> None:
        self.seq += 1
        ev = {"seq": self.seq, "t": time.time(), "tag": tag, "msg": msg, "level": level}
        self.buf.append(ev)
        log.info("[%s] %s", tag, msg)
        if self.loop:
            for q in list(self.subs):
                self.loop.call_soon_threadsafe(q.put_nowait, ev)


class Lab:
    def __init__(self, settings: Settings, engine: Engine | None = None, store: Store | None = None,
                 market: MarketData | None = None):
        self.s = settings
        self.events = EventLog()
        self.market = market or MarketData(settings.source)
        self.store = store or Store(settings.db_path)
        self.lock = threading.Lock()  # one inference at a time
        self.engine_warning = None
        self.engine: Engine | None = engine
        self.baseline = BootstrapEngine()
        self.busy = False
        self.round_counter: dict[tuple, int] = {}

    # ------------------------------------------------------------- lifecycle
    def load(self) -> None:
        if self.engine is None:
            self.events.emit("kronos", f"load {self.s.model} - Hugging Face")
            self.engine, self.engine_warning = build_engine(self.s)
            if self.engine_warning:
                self.events.emit("kronos", self.engine_warning, "warn")
        i = self.engine.info()
        self.events.emit("kronos", f"ready: {i['model']} | params {i['params']} | context {i['max_context']}"
                                   f" | device {i['device']}")

    def info(self) -> dict:
        return {"engine": self.engine.info() if self.engine else None, "warning": self.engine_warning,
                "source": self.market.active, "source_errors": self.market.errors,
                "settings": {k: v for k, v in asdict(self.s).items() if k not in ("db_path", "weights_dir")},
                "intervals": list(INTERVALS), "busy": self.busy}

    # ------------------------------------------------------------- forecasting
    def _context_len(self) -> int:
        return min(self.s.lookback, self.engine.info()["max_context"])

    def forecast(self, symbol: str | None = None, interval: str | None = None, horizon: int | None = None,
                 samples: int | None = None, temperature: float | None = None, top_p: float | None = None,
                 seed: int | None = None) -> dict:
        symbol = (symbol or self.s.symbol).upper()
        interval = interval or self.s.interval
        H = int(horizon or self.s.horizon)
        N = int(samples or self.s.samples)
        T = float(temperature if temperature is not None else self.s.temperature)
        P = float(top_p if top_p is not None else self.s.top_p)
        if not (1 <= H <= 240 and 1 <= N <= 1000 and 0.05 <= T <= 3 and 0.05 <= P <= 1):
            raise ValueError("parameters out of range")
        with self.lock:
            self.busy = True
            try:
                return self._forecast(symbol, interval, H, N, T, P, seed)
            finally:
                self.busy = False

    def _forecast(self, symbol, interval, H, N, T, P, seed) -> dict:
        key = (symbol, interval)
        n_round = self.round_counter[key] = self.round_counter.get(key, 0) + 1
        sec = INTERVALS[interval]
        ctx = self._context_len()
        self.events.emit(self.market.active or "data", f"GET klines {symbol} {interval} limit={ctx}")
        df = self.market.candles(symbol, interval, ctx)
        self.events.emit(self.market.active, f"{len(df)} candles - {symbol} {interval} - close {df.close.iloc[-1]:,.2f}")
        last_ts = df["timestamps"].iloc[-1]
        y_ts = pd.Series(pd.date_range(last_ts + pd.Timedelta(seconds=sec), periods=H, freq=f"{sec}s"))
        tag = self.engine.name
        step = "tokenize" if tag.startswith("kronos") else "bootstrap"
        self.events.emit(tag, f"round {n_round}: {step} last {len(df)} candles")
        t0 = time.time()
        paths = self.engine.sample(df, y_ts, N, T, P, seed)
        dt = time.time() - t0
        summ = analytics.summarize(paths, df["close"].to_numpy())
        f = summ["final"]
        self.events.emit(tag, f"predict next {H} - sample_count={N} - {dt:.1f}s")
        self.events.emit(tag, f"median {f['p50']:,.2f} - p10 {f['p10']:,.2f} - p90 {f['p90']:,.2f}")
        self.events.emit(tag, f"{summ['n_up']}/{N} paths up - regime {summ['regime']}")
        anchor = int(last_ts.timestamp())
        params = {"samples": N, "temperature": T, "top_p": P, "context": len(df), "seconds": round(dt, 2)}
        rid = self.store.add(created_at=time.time(), symbol=symbol, interval=interval, engine=self.engine.name,
                             anchor_ts=anchor, last_close=summ["last_close"], horizon=H, params=params,
                             summary=summ, paths=np.round(paths[..., 3], 6).tolist())
        if self.s.shadow_baseline and self.engine.name != self.baseline.name:
            bp = self.baseline.sample(df, y_ts, N, 1.0, 1.0, seed)
            self.store.add(created_at=time.time(), symbol=symbol, interval=interval, engine=self.baseline.name,
                           role="shadow", anchor_ts=anchor, last_close=summ["last_close"], horizon=H,
                           params=params, summary=analytics.summarize(bp, df["close"].to_numpy()),
                           paths=np.round(bp[..., 3], 6).tolist())
        return {"id": rid, "round": n_round}

    # ------------------------------------------------------------- live scoring
    def update_scores(self) -> int:
        """Grade every open round with the candles that have closed since its anchor."""
        finished = 0
        groups: dict[tuple, list] = collections.defaultdict(list)
        for r in self.store.open_rounds():
            groups[(r["symbol"], r["interval"])].append(r)
        for (symbol, interval), rounds in groups.items():
            sec = INTERVALS[interval]
            oldest = min(r["anchor_ts"] for r in rounds)
            need = int((time.time() - oldest) // sec) + 2
            try:
                df = self.market.candles(symbol, interval, min(max(need, 2), 1000))
            except Exception as e:  # noqa: BLE001
                self.events.emit("score", f"{symbol}: cannot fetch candles ({e})", "warn")
                continue
            ts = (df["timestamps"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().astype(int).to_numpy()
            for r in rounds:
                after = df["close"].to_numpy()[ts > r["anchor_ts"]][: r["horizon"]]
                if len(after) == 0 or (r.get("score") and r["score"]["steps"] == len(after)):
                    continue
                sc = analytics.score(np.asarray(r["paths"]), r["last_close"], after)
                done = len(after) >= r["horizon"]
                self.store.set_score(r["id"], sc, done)
                if done and r["role"] == "primary":
                    finished += 1
                    verdict = "inside p10-p90" if sc["inside_final"] else "outside p10-p90"
                    self.events.emit("score", f"round #{r['id']} closed: real {sc['actual_last']:,.2f} - {verdict}"
                                              f" - dir {'HIT' if sc['direction_hit'] else 'MISS'}"
                                              f" - coverage {sc['coverage']:.0%}")
        return finished

    def metrics(self, symbol: str, interval: str, horizon: int | None = None) -> dict:
        out = {}
        for role in ("primary", "shadow"):
            hist = self.store.history(symbol, interval, role, limit=5000, horizon=horizon)
            out[role] = analytics.aggregate([h["score"] for h in hist if h["status"] == "done"])
            out[role]["open"] = sum(1 for h in hist if h["status"] == "open")
            out[role].update(analytics.significance(hist, INTERVALS[interval]))
        return out

    def needs_new_round(self, symbol: str, interval: str, horizon: int | None = None, every_hours: int = 1) -> bool:
        """True when a new candle closed on this track's grid (forecasts issued at UTC multiples of
        ``every_hours``) and no round exists for it yet."""
        sec = INTERVALS[interval]
        newest_closed_open = (int(time.time()) // sec) * sec - sec
        if (newest_closed_open + sec) % (every_hours * 3600) != 0 and every_hours * 3600 > sec:
            return False
        last = self.store.latest(symbol, interval, horizon=horizon)
        return not last or newest_closed_open > last["anchor_ts"]
