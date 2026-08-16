"""Turn embeddings into the redundancy report.

Usage:
    python src/run_report.py --tag run1

Reports three scopes separately, because they answer different questions:
  within-worker   one person's own footage repeating itself
  within-factory  workers in one plant covering the same ground
  global          the corpus as a whole
A single blended number would hide which of the three is doing the work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD, OUT_DIR, SIM_THRESHOLDS  # noqa: E402
from dedup import measure, measure_by_group, summarise  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report corpus redundancy.")
    parser.add_argument("--tag", type=str, default="run1")
    parser.add_argument("--examples", type=int, default=40)
    parser.add_argument(
        "--blur-percentile",
        type=float,
        default=20.0,
        help="drop the blurriest N%% of frames before measuring redundancy",
    )
    return parser.parse_args()


def split_by_sharpness(
    manifest: pd.DataFrame, percentile: float
) -> tuple[np.ndarray, float]:
    """Return a keep-mask for frames sharp enough to measure, plus the cutoff.

    Motion blur is a confound, not just noise: blurred frames lose the detail
    that distinguishes them, so they match each other and inflate redundancy.
    Visual inspection confirmed a blur-only pair at 0.906 outranking a genuine
    same-workstation pair at 0.872.

    The cutoff is a percentile of this corpus rather than an absolute value,
    because Laplacian variance scales with resolution and scene content.
    """
    if "sharpness" not in manifest.columns:
        return np.ones(len(manifest), dtype=bool), float("nan")
    values = manifest["sharpness"].to_numpy()
    cutoff = float(np.percentile(values, percentile))
    return values >= cutoff, cutoff


def format_curve(curve: dict[float, float]) -> str:
    if not curve:
        return "    (no data)"
    lines = []
    for t in SIM_THRESHOLDS:
        pct = curve.get(t, float("nan")) * 100
        bar = "#" * int(round(pct / 2))
        marker = "  <-- headline" if abs(t - HEADLINE_THRESHOLD) < 1e-9 else ""
        lines.append(f"    sim >= {t:.2f}   {pct:5.1f}%  {bar}{marker}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    emb_path = EMB_DIR / f"{args.tag}.npy"
    man_path = EMB_DIR / f"{args.tag}.parquet"
    if not emb_path.exists():
        print(f"missing {emb_path} — run run_embed.py first", file=sys.stderr)
        return 1

    embeddings = np.load(emb_path)
    manifest = pd.read_parquet(man_path)
    print(f"loaded {embeddings.shape[0]} frames, {embeddings.shape[1]} dims")

    keep, cutoff = split_by_sharpness(manifest, args.blur_percentile)
    dropped = int((~keep).sum())
    if not np.isnan(cutoff):
        print(
            f"blur gate: dropped {dropped} frames "
            f"({dropped / len(manifest) * 100:.1f}%) below Laplacian variance "
            f"{cutoff:.1f}\n"
        )
        embeddings = embeddings[keep]
        manifest = manifest.loc[keep].reset_index(drop=True)
    else:
        print("blur gate: no sharpness column — run run_embed.py again\n")

    worker_key = (manifest["factory"] + "/" + manifest["worker"]).to_numpy()
    factory_key = manifest["factory"].to_numpy()

    per_worker = measure_by_group(embeddings, worker_key)
    per_factory = measure_by_group(embeddings, factory_key)
    global_result = measure(embeddings)

    worker_curve = summarise(per_worker)
    factory_curve = summarise(per_factory)

    print("REDUNDANCY — share of frames whose content already appeared earlier\n")
    print("  WITHIN WORKER (same person's own footage)")
    print(format_curve(worker_curve))
    print("\n  WITHIN FACTORY (across workers in one plant)")
    print(format_curve(factory_curve))
    print("\n  GLOBAL (whole sample)")
    print(format_curve(global_result.curve))

    headline = global_result.curve.get(HEADLINE_THRESHOLD, float("nan"))
    print(
        f"\nHEADLINE: {headline * 100:.1f}% of sampled frames at "
        f"sim >= {HEADLINE_THRESHOLD} duplicate earlier content."
    )
    print(
        "NOTE: threshold not yet human-calibrated. Review the example pairs "
        "before quoting this number anywhere."
    )

    examples = global_result.pairs_at(HEADLINE_THRESHOLD, limit=args.examples)
    example_rows = [
        {
            "similarity": round(sim, 4),
            "frame": manifest.iloc[i][["factory", "worker", "clip", "timestamp_sec"]].to_dict(),
            "matches_earlier": manifest.iloc[j][["factory", "worker", "clip", "timestamp_sec"]].to_dict(),
            "same_worker": bool(worker_key[i] == worker_key[j]),
        }
        for i, j, sim in examples
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": args.tag,
        "blur_percentile_dropped": args.blur_percentile,
        "blur_cutoff_laplacian_var": cutoff,
        "n_frames_dropped_as_blur": dropped,
        "n_frames": int(embeddings.shape[0]),
        "n_clips": int(manifest["clip"].nunique()),
        "n_workers": int(manifest["worker"].nunique()),
        "n_factories": int(manifest["factory"].nunique()),
        "headline_threshold": HEADLINE_THRESHOLD,
        "curves": {
            "within_worker": {str(k): v for k, v in worker_curve.items()},
            "within_factory": {str(k): v for k, v in factory_curve.items()},
            "global": {str(k): v for k, v in global_result.curve.items()},
        },
        "examples": example_rows,
    }
    out_path = OUT_DIR / f"report_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
