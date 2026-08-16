"""When does buying another hour from the same operator stop paying?

Corpus-level redundancy says a quarter of the footage repeats. It does not
say what a buyer should do about it. This does: process each worker's clips
in collection order and, for every clip, measure what fraction of its frames
are novel against everything already collected from that worker.

The result is a decay curve. Where it flattens is the point at which more
hours from the same operator stop adding information -- which is a
procurement decision, not a research one.

Scoped per worker on purpose. Cross-worker novelty barely decays (workers do
not duplicate each other), so mixing them would hide the effect that matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD, OUT_DIR  # noqa: E402
from run_report import split_by_sharpness  # noqa: E402


def clip_novelty(embeddings: np.ndarray, clip_ids: np.ndarray,
                 threshold: float) -> list[dict]:
    """Per-clip novelty against everything earlier from the same worker."""
    x = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatIP(x.shape[1])

    rows = []
    for position, clip in enumerate(pd.unique(clip_ids), start=1):
        mask = clip_ids == clip
        block = x[mask]
        if block.shape[0] == 0:
            continue

        if index.ntotal == 0:
            novel = 1.0
        else:
            sims, _ = index.search(block, 1)
            novel = float(np.mean(sims[:, 0] < threshold))

        rows.append({"position": position, "clip": clip,
                     "frames": int(block.shape[0]), "novel_share": novel})
        index.add(block)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Marginal novelty per operator.")
    parser.add_argument("--tag", type=str, default="factory051_v2")
    parser.add_argument("--threshold", type=float, default=HEADLINE_THRESHOLD)
    parser.add_argument("--blur-percentile", type=float, default=20.0)
    args = parser.parse_args()

    embeddings = np.load(EMB_DIR / f"{args.tag}.npy")
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")

    keep, _ = split_by_sharpness(manifest, args.blur_percentile)
    embeddings = embeddings[keep]
    manifest = manifest.loc[keep].reset_index(drop=True)

    per_worker = {}
    for worker, group in manifest.groupby("worker", sort=True):
        idx = group.index.to_numpy()
        rows = clip_novelty(embeddings[idx], group["clip"].to_numpy(), args.threshold)
        per_worker[worker] = rows
        print(f"  w{worker}: {len(rows)} clips", flush=True)

    # Frame-weighted mean novelty at each clip position across workers.
    max_pos = max(len(r) for r in per_worker.values())
    curve = []
    for pos in range(1, max_pos + 1):
        vals, weights = [], []
        for rows in per_worker.values():
            if pos <= len(rows):
                vals.append(rows[pos - 1]["novel_share"])
                weights.append(rows[pos - 1]["frames"])
        if vals:
            curve.append({
                "position": pos,
                "novel_share": float(np.average(vals, weights=weights)),
                "workers": len(vals),
            })

    print(f"\nMARGINAL NOVELTY per clip, threshold {args.threshold}\n")
    print("  clip #   novel %   bar")
    for row in curve:
        if row["position"] <= 12 or row["position"] % 5 == 0:
            pct = row["novel_share"] * 100
            print(f"  {row['position']:5d}   {pct:6.1f}%   {'#' * int(round(pct / 2.5))}")

    first = curve[0]["novel_share"]
    print(f"\nfirst clip:  {first * 100:.1f}% novel (by definition ~100%)")

    for target in (0.75, 0.60, 0.50):
        hit = next((r for r in curve if r["novel_share"] < target), None)
        if hit:
            print(f"drops below {target * 100:.0f}%: clip {hit['position']}")
        else:
            print(f"drops below {target * 100:.0f}%: never within this sample")

    tail = [r["novel_share"] for r in curve[-10:]]
    print(f"last 10 clips average: {np.mean(tail) * 100:.1f}% novel")

    cumulative_frames = sum(r["frames"] for rows in per_worker.values() for r in rows)
    novel_frames = sum(
        r["frames"] * r["novel_share"] for rows in per_worker.values() for r in rows
    )
    print(
        f"\nacross the whole sample: {novel_frames:.0f} of {cumulative_frames} frames "
        f"novel ({novel_frames / cumulative_frames * 100:.1f}%)"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"novelty_{args.tag}.json"
    dest.write_text(
        json.dumps({"threshold": args.threshold, "curve": curve,
                    "per_worker": per_worker}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
