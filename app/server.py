"""HTTP API + dashboard.  Run:  uvicorn app.server:app --port 8000"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import INTERVALS, ROOT, settings
from .service import Lab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
lab = Lab(settings)


async def scheduler() -> None:
    """New round whenever a candle closes; grade open rounds continuously."""
    while True:
        try:
            await asyncio.to_thread(lab.update_scores)
            for track in settings.tracks if settings.auto_run else []:
                horizon, every = (int(x) for x in track.split(":"))
                for sym in settings.auto_symbols or [settings.symbol]:
                    if not lab.busy and lab.needs_new_round(sym, settings.interval, horizon, every):
                        await asyncio.to_thread(lab.forecast, sym, settings.interval, horizon)
        except Exception as e:  # noqa: BLE001
            lab.events.emit("scheduler", f"{type(e).__name__}: {e}", "error")
        await asyncio.sleep(settings.poll_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    lab.events.loop = asyncio.get_running_loop()
    await asyncio.to_thread(lab.load)
    task = asyncio.create_task(scheduler())
    yield
    task.cancel()


app = FastAPI(title="Kronos Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/info")
def info():
    return lab.info()


@app.get("/api/state")
def state(symbol: str = Query(None), interval: str = Query(None), candles: int = Query(160, le=1000),
          horizon: int = Query(None)):
    symbol = (symbol or settings.symbol).upper()
    interval = interval or settings.interval
    if interval not in INTERVALS:
        raise HTTPException(400, "bad interval")
    try:
        df = lab.market.candles(symbol, interval, candles)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"market data: {e}") from e
    cdl = [[int(t.timestamp()), o, h, l, c, v] for t, o, h, l, c, v in
           df[["timestamps", "open", "high", "low", "close", "volume"]].itertuples(index=False)]
    rnd = lab.store.latest(symbol, interval, horizon=horizon)
    shadow = lab.store.latest(symbol, interval, "shadow", horizon=horizon)
    if rnd and shadow and shadow["anchor_ts"] != rnd["anchor_ts"]:
        shadow = None
    rounds = lab.store.history(symbol, interval, limit=40, horizon=horizon)
    return {"symbol": symbol, "interval": interval, "seconds": INTERVALS[interval], "candles": cdl,
            "round": rnd, "shadow": {"summary": shadow["summary"], "score": shadow["score"]} if shadow else None,
            "metrics": lab.metrics(symbol, interval, horizon), "rounds": rounds, "info": lab.info(),
            "horizon": horizon}


class ForecastReq(BaseModel):
    symbol: str | None = None
    interval: str | None = None
    horizon: int | None = Field(None, ge=1, le=240)
    samples: int | None = Field(None, ge=1, le=1000)
    temperature: float | None = Field(None, ge=0.05, le=3)
    top_p: float | None = Field(None, ge=0.05, le=1)
    seed: int | None = None


@app.post("/api/forecast")
async def forecast(req: ForecastReq):
    if req.interval and req.interval not in INTERVALS:
        raise HTTPException(400, "bad interval")
    if lab.busy:
        raise HTTPException(409, "a forecast is already running")
    try:
        return await asyncio.to_thread(lab.forecast, req.symbol, req.interval, req.horizon, req.samples,
                                       req.temperature, req.top_p, req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        lab.events.emit("kronos", f"forecast failed: {e}", "error")
        raise HTTPException(502, str(e)) from e


@app.get("/api/rounds/{rid}")
def round_detail(rid: int):
    rows = lab.store._rows("SELECT * FROM rounds WHERE id=?", (rid,))
    if not rows:
        raise HTTPException(404, "not found")
    return rows[0]


@app.get("/api/export.csv")
def export(symbol: str = Query(None), interval: str = Query(None)):
    symbol = (symbol or settings.symbol).upper()
    interval = interval or settings.interval
    rows = []
    for r in lab.store.history(symbol, interval, limit=10000):
        s, sc = r["summary"], r["score"] or {}
        rows.append({"id": r["id"], "anchor": pd.to_datetime(r["anchor_ts"], unit="s", utc=True), "engine": r["engine"],
                     "horizon": r["horizon"],
                     "last_close": r["last_close"], "median": s["median"], "p10": s["final"]["p10"],
                     "p90": s["final"]["p90"], "prob_up": s["prob_up"], "regime": s["regime"], "status": r["status"],
                     "actual": sc.get("actual_last"), "coverage": sc.get("coverage"),
                     "direction_hit": sc.get("direction_hit"), "crps_skill": sc.get("crps_skill")})
    csv = pd.DataFrame(rows).to_csv(index=False)
    return StreamingResponse(iter([csv]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=kronos_{symbol}_{interval}.csv"})


@app.get("/api/events")
async def events():
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    lab.events.subs.add(q)

    async def gen():
        try:
            for ev in list(lab.events.buf)[-120:]:
                yield f"data: {json.dumps(ev)}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            lab.events.subs.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
