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
