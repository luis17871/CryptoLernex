"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"KRONOS_{name}", default)


# Kronos model zoo (https://github.com/shiyu-coder/Kronos). Kronos-large is not public.
MODELS: dict[str, dict] = {
    "kronos-mini": {"repo": "NeoQuasar/Kronos-mini", "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
                    "params": "4.1M", "max_context": 2048},
    "kronos-small": {"repo": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                     "params": "24.7M", "max_context": 512},
    "kronos-base": {"repo": "NeoQuasar/Kronos-base", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                    "params": "102.3M", "max_context": 512},
}

INTERVALS: dict[str, int] = {  # interval -> seconds
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


@dataclass
class Settings:
    model: str = field(default_factory=lambda: _env("MODEL", "kronos-small"))
    # "auto" = Kronos if weights load, otherwise the statistical bootstrap engine.
    engine: str = field(default_factory=lambda: _env("ENGINE", "auto"))
    weights_dir: str = field(default_factory=lambda: _env("WEIGHTS_DIR", ""))  # optional local snapshot
    device: str = field(default_factory=lambda: _env("DEVICE", ""))
    source: str = field(default_factory=lambda: _env("SOURCE", "auto"))  # auto|binance|bybit|okx|synthetic
    symbols: list[str] = field(default_factory=lambda: _env("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(","))
    symbol: str = field(default_factory=lambda: _env("SYMBOL", "BTCUSDT"))
    interval: str = field(default_factory=lambda: _env("INTERVAL", "1h"))
    lookback: int = field(default_factory=lambda: int(_env("LOOKBACK", "512")))
    horizon: int = field(default_factory=lambda: int(_env("HORIZON", "24")))
    samples: int = field(default_factory=lambda: int(_env("SAMPLES", "100")))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "1.0")))
    top_p: float = field(default_factory=lambda: float(_env("TOP_P", "0.9")))
    batch_size: int = field(default_factory=lambda: int(_env("BATCH", "50")))
    auto_run: bool = field(default_factory=lambda: _env("AUTO_RUN", "1") == "1")
    poll_seconds: int = field(default_factory=lambda: int(_env("POLL_SECONDS", "30")))
    shadow_baseline: bool = field(default_factory=lambda: _env("SHADOW_BASELINE", "1") == "1")
    db_path: str = field(default_factory=lambda: _env("DB", str(ROOT / "data" / "kronos_lab.sqlite")))


settings = Settings()
