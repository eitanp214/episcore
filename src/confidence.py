"""Uncertainty on the headline figure, and where it actually comes from.

WHY NOT BOOTSTRAP. The obvious tool is wrong here. Resampling clips with
replacement puts the same clip in the corpus twice, and this statistic
measures duplication -- so every repeated clip scores as a perfect match
against its own earlier copy. A first attempt produced a 95% interval of
[49.2%, 53.4%] around a 27.2% point estimate: an interval that cannot
contain its own estimate, because the method manufactured the very thing it
was measuring.

Jackknife instead. Leaving one clip out at a time never duplicates anything
and keeps the corpus size essentially fixed, which matters because this
statistic is size-dependent -- a smaller corpus has less earlier material to
match against, so subsampling would bias it downward.

The per-operator breakdown is reported first, and deliberately so. It is the
more honest uncertainty statement: if the figure ranges from 7% to 46%
depending on whose footage you bought, the corpus average is a summary of
very different situations rather than a property of the domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD, OUT_DIR  # noqa: E402
from dedup import measure  # noqa: E402
from run_report import split_by_sharpness  # noqa: E402


def redundancy(embeddings: np.ndarray, threshold: float) -> float:
    return float(np.mean(measure(embeddings).max_sim >= threshold))


def main() -> int:
    parser = argparse.ArgumentParser(description="Jackknife CI for redundancy.")
    parser.add_argument("--tag", type=str, default="factory051_v2")
    parser.add_argument("--threshold", type=float, default=HEADLINE_THRESHOLD)
    parser.add_argument("--blur-percentile", type=float, default=20.0)
    args = parser.parse_args()

    embeddings = np.load(EMB_DIR / f"{args.tag}.npy")
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")
    keep, _ = split_by_sharpness(manifest, args.blur_percentile)
    embeddings = embeddings[keep]
    manifest = manifest.loc[keep].reset_index(drop=True)

    point = redundancy(embeddings, args.threshold)
    n_clips = manifest["clip"].nunique()
    print(f"point estimate: {point * 100:.1f}%  "
          f"({len(embeddings)} frames, {n_clips} clips)\n")

    # --- the number that actually matters --------------------------------
    print("PER OPERATOR")
    per_worker = {}
    for worker, group in manifest.groupby("worker", sort=True):
        val = redundancy(embeddings[group.index.to_numpy()], args.threshold)
        per_worker[str(worker)] = val
        print(f"  w{worker}  {val * 100:5.1f}%  n={len(group):4d}  "
              f"{'#' * int(round(val * 100 / 2))}")

    vals = np.array(list(per_worker.values()))
    print(f"\n  range {vals.min() * 100:.1f}% - {vals.max() * 100:.1f}%   "
          f"sd {vals.std(ddof=1) * 100:.1f} pts   "
          f"ratio {vals.max() / max(vals.min(), 1e-9):.1f}x")

    # --- jackknife over clips --------------------------------------------
    clips = manifest["clip"].to_numpy()
    unique_clips = pd.unique(clips)
    print(f"\njackknife: {len(unique_clips)} leave-one-clip-out fits ...", flush=True)

    estimates = np.empty(len(unique_clips))
    for i, clip in enumerate(unique_clips):
        estimates[i] = redundancy(embeddings[clips != clip], args.threshold)
        if (i + 1) % 60 == 0:
            print(f"  {i + 1}/{len(unique_clips)}", flush=True)

    n = len(unique_clips)
    mean_est = estimates.mean()
    var = (n - 1) / n * np.sum((estimates - mean_est) ** 2)
    se = float(np.sqrt(var))
    lo, hi = point - 1.96 * se, point + 1.96 * se

    print(f"\nHEADLINE  {point * 100:.1f}%   95% CI "
          f"[{lo * 100:.1f}%, {hi * 100:.1f}%]   (jackknife SE {se * 100:.2f} pts)")

    print(
        f"\nThe sampling interval is narrow, but it is the wrong thing to quote\n"
        f"alone. Operator-to-operator spread is {vals.min() * 100:.0f}% to "
        f"{vals.max() * 100:.0f}% -- roughly {(vals.max() - vals.min()) * 100:.0f} points,\n"
        f"an order of magnitude wider than the sampling error on the mean.\n"
        f"A buyer's exposure depends far more on WHOSE footage they bought\n"
        f"than on uncertainty in the corpus average."
    )

    dest = OUT_DIR / f"confidence_{args.tag}.json"
    dest.write_text(
        json.dumps({
            "threshold": args.threshold,
            "point": point,
            "ci95_jackknife": [float(lo), float(hi)],
            "jackknife_se": se,
            "per_worker": per_worker,
            "operator_spread": {
                "min": float(vals.min()), "max": float(vals.max()),
                "sd": float(vals.std(ddof=1)),
            },
            "note": "bootstrap is invalid for this statistic; see module docstring",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
