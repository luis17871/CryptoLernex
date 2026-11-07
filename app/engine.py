"""Forecast engines. Each one returns ``N`` sampled future paths of shape (N, H, 6)
with columns open, high, low, close, volume, amount.

* ``KronosEngine``  - the Kronos foundation model. Unlike ``KronosPredictor.predict`` (which
  averages its samples into a single line) we keep every Monte-Carlo path, which is what makes
  the probability fan, quantiles, up/down counts and calibration possible.
* ``BootstrapEngine`` - a statistical baseline (volatility-scaled block bootstrap of the
  instrument's own returns). It is the fallback when the weights are unavailable and the
  benchmark Kronos must beat to be worth anything.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np
import pandas as pd

from .config import MODELS, Settings

log = logging.getLogger("kronos.engine")
PRICE = ["open", "high", "low", "close", "volume", "amount"]


class Engine(Protocol):
    name: str

    def info(self) -> dict: ...

    def sample(self, df: pd.DataFrame, y_timestamps: pd.Series, n_paths: int,
               temperature: float, top_p: float, seed: int | None) -> np.ndarray: ...


# ---------------------------------------------------------------- Kronos
class KronosEngine:
    def __init__(self, model_key: str = "kronos-small", weights_dir: str = "", device: str = "",
                 batch_size: int = 50):
        import torch

        from .vendor.kronos import Kronos, KronosTokenizer

        import os
        if os.environ.get("KRONOS_TORCH_THREADS"):
            torch.set_num_threads(int(os.environ["KRONOS_TORCH_THREADS"]))
        if model_key not in MODELS:
            raise ValueError(f"unknown model {model_key}; options: {list(MODELS)}")
        spec = MODELS[model_key]
        self.spec, self.model_key = spec, model_key
        self.name = model_key
        self.batch_size = batch_size
        self.device = device or ("cuda:0" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                 else "cpu")
        tok_src = f"{weights_dir}/tokenizer" if weights_dir else spec["tokenizer"]
        mdl_src = f"{weights_dir}/model" if weights_dir else spec["repo"]
        t0 = time.time()
        self.tokenizer = KronosTokenizer.from_pretrained(tok_src).to(self.device).eval()
        self.model = Kronos.from_pretrained(mdl_src).to(self.device).eval()
        self.load_seconds = round(time.time() - t0, 1)
        self.max_context = spec["max_context"]
        log.info("loaded %s on %s in %.1fs", model_key, self.device, self.load_seconds)

    @classmethod
    def from_modules(cls, tokenizer, model, max_context: int = 512, name: str = "kronos-custom",
                     batch_size: int = 50, device: str = "cpu") -> "KronosEngine":
        """Wrap already-built modules (used by tests and fine-tuned checkpoints)."""
        self = cls.__new__(cls)
        self.tokenizer, self.model = tokenizer.to(device).eval(), model.to(device).eval()
        self.max_context, self.name, self.model_key = max_context, name, name
        self.spec = {"repo": name, "tokenizer": "custom", "params": "?", "max_context": max_context}
        self.batch_size, self.device, self.load_seconds = batch_size, device, 0.0
        return self

    def info(self) -> dict:
        return {"engine": "kronos", "model": self.model_key, "repo": self.spec["repo"],
                "tokenizer": self.spec["tokenizer"], "params": self.spec["params"],
                "max_context": self.max_context, "device": self.device}

    def sample(self, df, y_timestamps, n_paths, temperature=1.0, top_p=0.9, seed=None):
        import torch

        from .vendor.kronos import auto_regressive_inference, calc_time_stamps

        df = df.tail(self.max_context)
        x = df[PRICE].to_numpy(np.float64)
        mean, std = x.mean(0), x.std(0)
        xn = np.clip((x - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        xs = calc_time_stamps(pd.Series(df["timestamps"].values)).to_numpy(np.float32)
        ys = calc_time_stamps(pd.Series(pd.to_datetime(y_timestamps).values)).to_numpy(np.float32)
        H = len(ys)
        if seed is not None:
            torch.manual_seed(seed)
        out = []
        for start in range(0, n_paths, self.batch_size):
            b = min(self.batch_size, n_paths - start)
            xb = torch.from_numpy(xn).unsqueeze(0).repeat(b, 1, 1).to(self.device)
            xsb = torch.from_numpy(xs).unsqueeze(0).repeat(b, 1, 1).to(self.device)
            ysb = torch.from_numpy(ys).unsqueeze(0).repeat(b, 1, 1).to(self.device)
            # sample_count=1 per row -> the internal mean is a no-op and each row is one path
            preds = auto_regressive_inference(self.tokenizer, self.model, xb, xsb, ysb, self.max_context, H,
                                              clip=5, T=temperature, top_k=0, top_p=top_p, sample_count=1)
            out.append(preds[:, -H:, :])
        paths = np.concatenate(out, 0).astype(np.float64) * (std + 1e-5) + mean
        return _sanitize(paths)


# ---------------------------------------------------------------- Bootstrap baseline
class BootstrapEngine:
    """Block bootstrap of log-returns, rescaled to the current (EWMA) volatility.

    Captures fat tails and short-range autocorrelation of the instrument itself; zero drift
    by construction (a martingale benchmark that is hard to beat on direction).
    """

    name = "bootstrap"

    def __init__(self, block: int = 6, ewma_lambda: float = 0.94):
        self.block, self.lam = block, ewma_lambda

    def info(self) -> dict:
        return {"engine": "bootstrap", "model": "vol-scaled block bootstrap", "repo": "-",
                "tokenizer": "-", "params": "0", "max_context": 2000, "device": "cpu"}

    def sample(self, df, y_timestamps, n_paths, temperature=1.0, top_p=0.9, seed=None):
        rng = np.random.default_rng(seed)
        H = len(y_timestamps)
        c = df["close"].to_numpy(np.float64)
        o = df["open"].to_numpy(np.float64)
        r = np.diff(np.log(c))
        r = r - r.mean()
        # standardise history by its own EWMA vol, then rescale to today's vol
        var = np.empty_like(r)
        var[0] = r[: min(50, len(r))].var() + 1e-12
        for i in range(1, len(r)):
            var[i] = self.lam * var[i - 1] + (1 - self.lam) * r[i - 1] ** 2
        z = r / np.sqrt(var)
        cur_vol = np.sqrt(self.lam * var[-1] + (1 - self.lam) * r[-1] ** 2) * temperature
        n_blocks = int(np.ceil(H / self.block))
        starts = rng.integers(0, len(z) - self.block, size=(n_paths, n_blocks))
        idx = (starts[..., None] + np.arange(self.block)).reshape(n_paths, -1)[:, :H]
        rets = z[idx] * cur_vol
        close = c[-1] * np.exp(np.cumsum(rets, axis=1))
        open_ = np.concatenate([np.full((n_paths, 1), c[-1]), close[:, :-1]], axis=1)
        # wicks and volume resampled from the same historical bars
        hi_ratio = (df["high"].to_numpy() / np.maximum(o, c))[1:][idx]
        lo_ratio = (df["low"].to_numpy() / np.minimum(o, c))[1:][idx]
        high = np.maximum(open_, close) * hi_ratio
        low = np.minimum(open_, close) * lo_ratio
        vol = df["volume"].to_numpy()[1:][idx]
        amount = vol * (open_ + high + low + close) / 4
        return _sanitize(np.stack([open_, high, low, close, vol, amount], axis=-1))


def _sanitize(p: np.ndarray) -> np.ndarray:
    """Enforce OHLC consistency (high >= max(o,c), low <= min(o,c), non-negative volume)."""
    o, h, l, c = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    p[..., 1] = np.maximum.reduce([h, o, c])
    p[..., 2] = np.minimum.reduce([l, o, c])
    p[..., 4:] = np.maximum(p[..., 4:], 0)
    return p


def build_engine(s: Settings) -> tuple[Engine, str | None]:
    """Returns (engine, warning). Falls back to the bootstrap engine when Kronos can't load."""
    if s.engine == "bootstrap":
        return BootstrapEngine(), None
    try:
        return KronosEngine(s.model, s.weights_dir, s.device, s.batch_size), None
    except Exception as e:  # noqa: BLE001
        if s.engine == "kronos":
            raise
        msg = f"Kronos weights unavailable ({type(e).__name__}: {str(e)[:160]}); using bootstrap engine"
        log.warning(msg)
        return BootstrapEngine(), msg
