"""How much does the headline number depend on the blur gate?

Filtering the blurriest frames RAISED measured redundancy (24.1% -> 27.2%).
A reviewer will read that as the filter being tuned to flatter the result.
The defence is not an argument, it is this table: if the figure moves
smoothly and modestly across filter strengths, the choice of cutoff is not
carrying the claim. If it jumps, the claim is fragile and we say so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD, SIM_THRESHOLDS  # noqa: E402
from dedup import measure  # noqa: E402

BLUR_LEVELS = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Blur-gate sensitivity.")
    parser.add_argument("--tag", type=str, default="factory051_v2")
    args = parser.parse_args()

    embeddings = np.load(EMB_DIR / f"{args.tag}.npy")
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")
    if "sharpness" not in manifest.columns:
        print("no sharpness column in this run", file=sys.stderr)
        return 1

    sharp = manifest["sharpness"].to_numpy()
    rows = []
    for level in BLUR_LEVELS:
        cutoff = float(np.percentile(sharp, level)) if level > 0 else -np.inf
        keep = sharp >= cutoff
        result = measure(embeddings[keep])
        row = {"blur_dropped_%": level, "frames": int(keep.sum())}
        row.update({f"@{t:.2f}": round(result.curve[t] * 100, 1) for t in SIM_THRESHOLDS})
        rows.append(row)
        print(f"  {level:4.0f}% dropped -> {keep.sum():5d} frames", flush=True)

    table = pd.DataFrame(rows)
    print("\nREDUNDANCY (%) BY BLUR-GATE STRENGTH\n")
    print(table.to_string(index=False))

    headline_col = f"@{HEADLINE_THRESHOLD:.2f}"
    values = table[headline_col].to_numpy()
    spread = float(values.max() - values.min())
    print(f"\nheadline column {headline_col}: min={values.min():.1f} "
          f"max={values.max():.1f} spread={spread:.1f} points")

    if spread <= 5.0:
        print("STABLE — the cutoff is not carrying the claim.")
    elif spread <= 10.0:
        print("MODERATE — quote a range, not a point estimate.")
    else:
        print("FRAGILE — do not publish a single number; publish the curve only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
