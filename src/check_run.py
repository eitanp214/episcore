"""Report what a checkpointed run actually captured."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR  # noqa: E402

for tag in ("factory051", "factory051_v2"):
    emb = EMB_DIR / f"{tag}.npy"
    man = EMB_DIR / f"{tag}.parquet"
    print(f"\n=== {tag}")
    if not emb.exists():
        print("  missing")
        continue

    vectors = np.load(emb)
    manifest = pd.read_parquet(man)
    print(f"  frames={vectors.shape[0]}  dims={vectors.shape[1]}")
    print(f"  clips={manifest['clip'].nunique()}  workers={manifest['worker'].nunique()}")
    print(f"  columns={list(manifest.columns)}")
    print(f"  per-worker frames:")
    for worker, count in manifest.groupby("worker").size().items():
        print(f"    w{worker}: {count}")

    if "sharpness" in manifest.columns:
        s = manifest["sharpness"]
        print(
            f"  sharpness: min={s.min():.1f} p10={s.quantile(.1):.1f} "
            f"p50={s.median():.1f} p90={s.quantile(.9):.1f} max={s.max():.1f}"
        )
