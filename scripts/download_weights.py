"""Download Kronos weights once so the lab can run offline:

    python scripts/download_weights.py kronos-small ./weights
    KRONOS_WEIGHTS_DIR=./weights/kronos-small uvicorn app.server:app
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import MODELS  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "kronos-small"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "weights") / key
spec = MODELS[key]
snapshot_download(spec["repo"], local_dir=out / "model")
snapshot_download(spec["tokenizer"], local_dir=out / "tokenizer")
print(f"saved to {out}  ->  export KRONOS_WEIGHTS_DIR={out} KRONOS_MODEL={key}")
