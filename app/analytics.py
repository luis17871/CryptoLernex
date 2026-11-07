"""Turning sampled paths into decisions, and grading forecasts against reality."""
from __future__ import annotations

import numpy as np

QS = [5, 10, 25, 50, 75, 90, 95]


def realized_vol(close: np.ndarray, window: int = 96) -> float:
    r = np.diff(np.log(close[-window - 1:]))
    return float(r.std()) if len(r) > 1 else 0.0


def summarize(paths: np.ndarray, history_close: np.ndarray) -> dict:
    """paths: (N, H, 6). Returns everything the dashboard shows for one forecast."""
    close = paths[..., 3]
    hi, lo = paths[..., 1], paths[..., 2]
    last = float(history_close[-1])
    N, H = close.shape
    q = np.percentile(close, QS, axis=0)  # (len(QS), H)
    band = {f"p{p}": q[i].tolist() for i, p in enumerate(QS)}
    final = close[:, -1]
    fq = {f"p{p}": float(np.percentile(final, p)) for p in QS}
    n_up = int((final > last).sum())
    ret = final / last - 1

    # probability that the price TOUCHES a level at any time within the horizon
    path_max, path_min = hi.max(axis=1), lo.min(axis=1)
    touch = {}
    for pct in (0.5, 1, 2, 3, 5):
        touch[f"+{pct}%"] = float((path_max >= last * (1 + pct / 100)).mean())
        touch[f"-{pct}%"] = float((path_min <= last * (1 - pct / 100)).mean())

    # regime from the shape of the distribution vs. what history implies
    hv = realized_vol(history_close)
    hist_band = last * hv * np.sqrt(H) * 2 * 1.2816 if hv > 0 else 1e-9  # p10-p90 width under a RW
    width = fq["p90"] - fq["p10"]
    trend = (fq["p50"] - last) / width if width > 0 else 0.0
    vol_ratio = width / hist_band if hist_band > 0 else 1.0
    prob_up = n_up / N
    if vol_ratio > 1.6:
        regime = "VOLATILE"
    elif trend > 0.2 and prob_up >= 0.62:
        regime = "TRENDING UP"
    elif trend < -0.2 and prob_up <= 0.38:
        regime = "TRENDING DOWN"
    elif vol_ratio < 0.6:
        regime = "COMPRESSED"
    else:
        regime = "CHOPPY"

    # step at which the median path peaks / bottoms (useful for timing)
    med = q[QS.index(50)]
    return {
        "last_close": last, "n_paths": N, "horizon": H,
        "band": band, "final": fq,
        "median": fq["p50"], "spread": width,
        "n_up": n_up, "n_down": N - n_up, "prob_up": prob_up,
        "exp_return": float(ret.mean()), "median_return": float(np.median(ret)),
        "var95": float(np.percentile(ret, 5)), "cvar95": float(ret[ret <= np.percentile(ret, 5)].mean()),
        "touch": touch,
        "regime": regime, "trend_score": float(trend), "vol_ratio": float(vol_ratio),
        "hist_vol": hv, "median_peak_step": int(np.argmax(med)) + 1, "median_trough_step": int(np.argmin(med)) + 1,
    }


def crps_ensemble(samples: np.ndarray, y: float) -> float:
    """CRPS of an ensemble forecast (lower is better): E|X-y| - 0.5 E|X-X'|."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(x)
    term1 = np.abs(x - y).mean()
    # E|X-X'| via sorted-sample identity, O(n log n)
    term2 = (2 * np.arange(1, n + 1) - n - 1).dot(x) * 2 / (n * n)
    return float(term1 - 0.5 * term2)


def score(paths_close: np.ndarray, last_close: float, actual: np.ndarray) -> dict:
    """Grade a forecast with the k realised closes available so far (k <= H).

    * coverage   - share of realised closes inside the p10-p90 band (well calibrated ~= 0.80)
    * direction  - did the median call the sign of the move at the last available step?
    * crps       - proper scoring rule for the whole distribution (lower is better)
    * skill      - 1 - MAE(median) / MAE(naive "price stays flat"); > 0 beats no-change
    """
    k = len(actual)
    pc = paths_close[:, :k]
    p10, p50, p90 = np.percentile(pc, [10, 50, 90], axis=0)
    inside = (actual >= p10) & (actual <= p90)
    crps = np.mean([crps_ensemble(pc[:, i], actual[i]) for i in range(k)])
    crps_naive = np.mean(np.abs(actual - last_close))  # CRPS of a point mass at last close
    mae = np.mean(np.abs(p50 - actual))
    mae_naive = np.mean(np.abs(last_close - actual))
    pred_dir = np.sign(p50[-1] - last_close)
    real_dir = np.sign(actual[-1] - last_close)
    # PIT of each realised value (uniform if calibrated) - for the calibration histogram
    pit = [(pc[:, i] < actual[i]).mean() for i in range(k)]
    return {
        "steps": k,
        "coverage": float(inside.mean()),
        "inside_final": bool(inside[-1]),
        "direction_hit": bool(pred_dir == real_dir and real_dir != 0),
        "crps": float(crps), "crps_naive": float(crps_naive),
        "crps_skill": float(1 - crps / crps_naive) if crps_naive > 0 else 0.0,
        "mae": float(mae), "mae_naive": float(mae_naive),
        "skill": float(1 - mae / mae_naive) if mae_naive > 0 else 0.0,
        "actual_last": float(actual[-1]), "pit": [float(v) for v in pit],
    }


def aggregate(scores: list[dict]) -> dict:
    done = [s for s in scores if s]
    if not done:
        return {"rounds": 0}
    pit = np.concatenate([s["pit"] for s in done])
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    return {
        "rounds": len(done),
        "hit_rate": float(np.mean([s["direction_hit"] for s in done])),
        "coverage": float(np.mean([s["coverage"] for s in done])),
        "inside_final": float(np.mean([s["inside_final"] for s in done])),
        "crps": float(np.mean([s["crps"] for s in done])),
        "crps_skill": float(np.mean([s["crps_skill"] for s in done])),
        "skill": float(np.mean([s["skill"] for s in done])),
        "pit_hist": (hist / max(1, hist.sum())).tolist(),
    }
