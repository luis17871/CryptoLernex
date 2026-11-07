"""SQLite persistence for forecast rounds (so the track record survives restarts)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    engine TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'primary',      -- primary | shadow (baseline benchmark)
    anchor_ts INTEGER NOT NULL,                -- open time (s) of the last known candle
    last_close REAL NOT NULL,
    horizon INTEGER NOT NULL,
    params TEXT NOT NULL,
    summary TEXT NOT NULL,
    paths TEXT NOT NULL,                       -- close paths (N x H), rounded
    status TEXT NOT NULL DEFAULT 'open',       -- open | done
    score TEXT
);
CREATE INDEX IF NOT EXISTS ix_rounds_key ON rounds(symbol, interval, role, id);
"""


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def add(self, **r) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO rounds(created_at,symbol,interval,engine,role,anchor_ts,last_close,horizon,params,summary,paths)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (r["created_at"], r["symbol"], r["interval"], r["engine"], r.get("role", "primary"), r["anchor_ts"],
                 r["last_close"], r["horizon"], json.dumps(r["params"]), json.dumps(r["summary"]),
                 json.dumps(r["paths"])))
            self.db.commit()
            return int(cur.lastrowid)

    def set_score(self, rid: int, score: dict, done: bool) -> None:
        with self.lock:
            self.db.execute("UPDATE rounds SET score=?, status=? WHERE id=?",
                            (json.dumps(score), "done" if done else "open", rid))
            self.db.commit()

    def _rows(self, sql: str, args: tuple) -> list[dict]:
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            for k in ("params", "summary", "paths", "score"):
                if d.get(k):
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def open_rounds(self) -> list[dict]:
        return self._rows("SELECT * FROM rounds WHERE status='open'", ())

    def latest(self, symbol: str, interval: str, role: str = "primary", horizon: int | None = None) -> dict | None:
        hz = "" if horizon is None else " AND horizon=?"
        args = (symbol, interval, role) + (() if horizon is None else (int(horizon),))
        r = self._rows(f"SELECT * FROM rounds WHERE symbol=? AND interval=? AND role=?{hz} ORDER BY id DESC LIMIT 1",
                       args)
        return r[0] if r else None

    def history(self, symbol: str, interval: str, role: str = "primary", limit: int = 200,
                with_paths: bool = False, horizon: int | None = None) -> list[dict]:
        cols = "*" if with_paths else ("id,created_at,symbol,interval,engine,role,anchor_ts,last_close,horizon,"
                                       "params,summary,status,score")
        hz = "" if horizon is None else " AND horizon=?"
        args = (symbol, interval, role) + (() if horizon is None else (int(horizon),)) + (limit,)
        return self._rows(f"SELECT {cols} FROM rounds WHERE symbol=? AND interval=? AND role=?{hz} "
                          "ORDER BY id DESC LIMIT ?", args)
