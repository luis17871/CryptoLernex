import time

import numpy as np
import pandas as pd
import pytest

from app import analytics
from app.backtest import run as backtest_run
from app.config import Settings
from app.data import MarketData, fetch_synthetic
from app.engine import BootstrapEngine, KronosEngine
from app.service import Lab
from app.store import Store


def tiny_kronos():
    torch = pytest.importorskip("torch")
    from app.vendor.kronos import Kronos, KronosTokenizer

    torch.manual_seed(0)
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
                          ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                          beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
    mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0,
                 attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=True)
    return KronosEngine.from_modules(tok, mdl, max_context=128, name="kronos-tiny", batch_size=16)


def _df(n=300):
    return fetch_synthetic("BTCUSDT", "1h", n)


def _yts(df, h):
    return pd.Series(pd.date_range(df.timestamps.iloc[-1] + pd.Timedelta(hours=1), periods=h, freq="h"))


def test_synthetic_is_deterministic_and_closed():
    a = fetch_synthetic("BTCUSDT", "1h", 200)
    b = fetch_synthetic("BTCUSDT", "1h", 200, now=time.time() - 3 * 3600)
    assert len(a) == 200
    assert (a.timestamps.iloc[-1] + pd.Timedelta(hours=1)).timestamp() <= time.time()
    shared = a.merge(b, on="timestamps", suffixes=("a", "b"))
    assert len(shared) > 150 and np.allclose(shared.closea, shared.closeb)
    assert (a.high >= a[["open", "close"]].max(axis=1)).all()


def test_bootstrap_paths_shape_and_ohlc():
    df = _df()
    p = BootstrapEngine().sample(df, _yts(df, 24), 50, 1.0, 1.0, 1)
    assert p.shape == (50, 24, 6)
    assert (p[..., 1] >= p[..., 3]).all() and (p[..., 2] <= p[..., 3]).all()
    assert np.isclose(p[:, 0, 0], df.close.iloc[-1]).all()


def test_kronos_engine_keeps_individual_paths():
    eng = tiny_kronos()
    df = _df(200)
    p = eng.sample(df, _yts(df, 6), 20, 1.0, 0.9, 3)
    assert p.shape == (20, 6, 6)
    assert np.isfinite(p).all()
    assert p[:, -1, 3].std() > 0  # paths differ -> not averaged


def test_crps_matches_bruteforce():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=60), 0.3
    brute = np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()
    assert np.isclose(analytics.crps_ensemble(x, y), brute)


def test_score_calibrated_forecast_has_80pct_coverage():
    rng = np.random.default_rng(1)
    covs = []
    for _ in range(200):
        paths = 100 + rng.normal(size=(400, 10)).cumsum(1)
        actual = 100 + rng.normal(size=10).cumsum()
        covs.append(analytics.score(paths, 100.0, actual)["coverage"])
    assert 0.74 < np.mean(covs) < 0.86


def test_summary_fields():
    df = _df()
    p = BootstrapEngine().sample(df, _yts(df, 24), 100, 1.0, 1.0, 2)
    s = analytics.summarize(p, df.close.to_numpy())
    assert s["n_up"] + s["n_down"] == 100
    assert s["final"]["p10"] <= s["median"] <= s["final"]["p90"]
    assert s["regime"] in {"VOLATILE", "TRENDING UP", "TRENDING DOWN", "COMPRESSED", "CHOPPY"}
    assert 0 <= s["touch"]["+1%"] <= 1


def test_lab_round_trip_and_scoring(monkeypatch):
    s = Settings(source="synthetic", engine="bootstrap", lookback=256, horizon=6, samples=40, db_path=":memory:")
    lab = Lab(s, engine=BootstrapEngine(), store=Store(":memory:"), market=MarketData("synthetic"))
    lab.load()
    out = lab.forecast("BTCUSDT", "1h", seed=1)
    r = lab.store.latest("BTCUSDT", "1h")
    assert r["id"] == out["id"] and len(r["paths"]) == 40 and len(r["paths"][0]) == 6
    assert not lab.needs_new_round("BTCUSDT", "1h")
    # jump 7 hours into the future: all 6 candles closed -> round graded
    future = time.time() + 7 * 3600
    real = fetch_synthetic
    monkeypatch.setattr("app.data.SOURCES", {"synthetic": lambda *a, **k: real(*a, **k, now=future)})
    monkeypatch.setattr("app.service.time.time", lambda: future)
    assert lab.update_scores() == 1
    m = lab.metrics("BTCUSDT", "1h")
    assert m["primary"]["rounds"] == 1 and 0 <= m["primary"]["coverage"] <= 1


def test_backtest_runs_with_both_engines():
    df = _df(400)
    out = backtest_run(df, {"tiny": tiny_kronos(), "bootstrap": BootstrapEngine()}, horizon=4, rounds=3, step=5,
                       context=100, samples=10, temperature=1.0, top_p=0.9, progress=False)
    assert out["tiny"]["rounds"] == 3 and out["bootstrap"]["rounds"] == 3


def test_forward_grade_only_complete_rounds():
    from app.forward import binom_p, grade

    df = _df(300)
    hist = df.iloc[:250]
    last = float(hist.close.iloc[-1])
    paths = BootstrapEngine().sample(hist, _yts(hist, 10), 20, 1.0, 1.0, 0)[..., 3]
    r = {"engine": "b", "anchor_ts": int(hist.timestamps.iloc[-1].timestamp()), "last_close": last, "horizon": 10,
         "paths": paths.tolist(), "median": 0, "p10": 0, "p90": 0, "prob_up": 0.5, "regime": "CHOPPY"}
    late = dict(r, anchor_ts=int(df.timestamps.iloc[-3].timestamp()))
    g = grade([r, late], df)
    assert len(g) == 1 and g.actual.iloc[0] == df.close.iloc[259]
    assert binom_p(0, 10) == 1.0 and binom_p(10, 10) == 1 / 1024


def test_significance_uses_non_overlapping_rounds():
    rows = [{"status": "done", "anchor_ts": i * 3600, "horizon": 24, "score": {"direction_hit": True}}
            for i in range(240)]
    s = analytics.significance(rows, 3600)
    assert s["n_indep"] == 10 and s["hit_indep"] == 1.0
    assert s["ci_low"] > 0.5 and s["verdict"] == "ventaja"
    few = analytics.significance(rows[:48], 3600)
    assert few["n_indep"] == 2 and few["verdict"] == "pocos datos"
    lo, hi = analytics.wilson(5, 10)
    assert lo < 0.5 < hi


def test_scheduler_settings_auto_symbols(monkeypatch):
    monkeypatch.setenv("KRONOS_AUTO_SYMBOLS", "BTCUSDT,ETHUSDT")
    assert Settings().auto_symbols == ["BTCUSDT", "ETHUSDT"]
    monkeypatch.delenv("KRONOS_AUTO_SYMBOLS")
    assert Settings().auto_symbols == []


def test_tracks_grid_and_horizon_filter(monkeypatch):
    s = Settings(source="synthetic", engine="bootstrap", lookback=128, samples=10, db_path=":memory:")
    lab = Lab(s, engine=BootstrapEngine(), store=Store(":memory:"), market=MarketData("synthetic"))
    t0 = 1794081600  # 2026-11-07 20:00 UTC, a multiple of 4h
    monkeypatch.setattr("app.service.time.time", lambda: t0 + 60)
    assert lab.needs_new_round("BTCUSDT", "1h", 4, 4)       # candle 19:00 closed at 20:00 -> on the 4h grid
    lab.forecast("BTCUSDT", "1h", horizon=4, seed=1)
    lab.forecast("BTCUSDT", "1h", horizon=24, seed=1)
    assert not lab.needs_new_round("BTCUSDT", "1h", 4, 4)
    monkeypatch.setattr("app.service.time.time", lambda: t0 + 3600 + 60)
    assert not lab.needs_new_round("BTCUSDT", "1h", 4, 4)   # 21:00 is off the 4h grid
    assert lab.needs_new_round("BTCUSDT", "1h", 24, 1)
    assert lab.store.latest("BTCUSDT", "1h", horizon=4)["horizon"] == 4
    assert len(lab.store.history("BTCUSDT", "1h", horizon=24)) == 1


def test_monitor_collects_progress_and_results(tmp_path):
    import json as _json

    from app.monitor import collect

    df = _df(700)
    backtest_run(df, {"bootstrap": BootstrapEngine()}, horizon=4, rounds=20, step=4, context=100, samples=10,
                 temperature=1.0, top_p=1.0, progress=False, progress_file=str(tmp_path / "t1.progress.json"),
                 meta={"symbol": "BTCUSDT", "model": "kronos-small", "horizon": 4})
    (tmp_path / "plan.json").write_text("﻿" + _json.dumps([{"name": "t1"}, {"name": "t2"}]), encoding="utf-8")
    (tmp_path / "t2.failed").write_text("x")
    out = {t["name"]: t for t in collect(tmp_path)["tests"]}
    assert out["t1"]["done"] == 20 and out["t1"]["engines"]["bootstrap"]["n"] == 20
    assert out["t1"]["engines"]["bootstrap"]["curve"][-1][0] == 20
    assert out["t2"]["status"] == "failed"


def test_quantile_engine_builds_coherent_paths():
    from app.engine import _QuantileEngine

    class Fake(_QuantileEngine):
        levels = [0.1, 0.5, 0.9]

        def _quantiles(self, close, H):
            base = close[-1]
            return np.array([[base * (1 - 0.01 * (h + 1)), base, base * (1 + 0.01 * (h + 1))] for h in range(H)])

    df = _df(200)
    p = Fake().sample(df, _yts(df, 5), 101, 1.0, 1.0, 0)
    c = p[..., 3]
    assert p.shape == (101, 5, 6)
    assert np.isclose(np.median(c[:, -1]), df.close.iloc[-1])
    assert abs(np.mean(c[:, -1] > df.close.iloc[-1]) - 0.5) < 0.02
    assert np.isclose(np.percentile(c[:, -1], 90), df.close.iloc[-1] * 1.05, rtol=2e-3)
    assert (np.diff(c, axis=0) >= -1e-9).all()  # comonotonic: path order is the same at every step
