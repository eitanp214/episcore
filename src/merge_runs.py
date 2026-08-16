"""Concatenate embedding runs that cover different workers.

A long run is network-bound and dies mid-corpus; resuming with
--worker-offset produces a second partial run. This stitches them back into
one dataset so the report sees the whole corpus.

Order matters. Redundancy is defined against *earlier* frames, so runs are
concatenated in the order given and that order becomes corpus order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge embedding runs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="source tags, in order")
    parser.add_argument("--out", required=True, help="destination tag")
    args = parser.parse_args()

    vectors, frames = [], []
    for tag in args.inputs:
        emb = EMB_DIR / f"{tag}.npy"
        man = EMB_DIR / f"{tag}.parquet"
        if not emb.exists():
            print(f"missing {emb}", file=sys.stderr)
            return 1
        v = np.load(emb)
        m = pd.read_parquet(man)
        if len(v) != len(m):
            print(f"{tag}: {len(v)} vectors vs {len(m)} manifest rows", file=sys.stderr)
            return 1
        vectors.append(v)
        frames.append(m)
        print(f"  {tag}: {len(v)} frames, workers={sorted(m['worker'].unique())}")

    matrix = np.concatenate(vectors, axis=0)
    manifest = pd.concat(frames, ignore_index=True)

    dupes = manifest.duplicated(subset=["clip", "frame_pos"]).sum()
    if dupes:
        print(
            f"WARNING: {dupes} duplicate (clip, frame_pos) rows across inputs — "
            f"overlapping runs would inflate redundancy",
            file=sys.stderr,
        )

    np.save(EMB_DIR / f"{args.out}.npy", matrix)
    manifest.to_parquet(EMB_DIR / f"{args.out}.parquet", index=False)
    print(
        f"\n{len(matrix)} frames, {manifest['clip'].nunique()} clips, "
        f"{manifest['worker'].nunique()} workers -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
