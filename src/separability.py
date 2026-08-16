"""Can the embedding identify which workstation a frame came from?

The cross-factory control answers "does CLIP collapse industrial interiors
into one blob?" That control needs gated data. This asks a sharper version
of the same question using only what is already on disk: if the embedding
had collapsed, it could not tell eight stations inside ONE factory apart.

Chance is 12.5% for eight workers. Accuracy well above that means the
embedding carries fine-grained discriminative structure on this footage,
which is the property the redundancy metric depends on.

Split is by CLIP, not by frame. Frames from one clip are seconds apart and
near-identical; letting them straddle the split would leak the answer and
produce a meaningless 99%.

CAVEAT, stated in the output as well: each worker has their own fisheye
intrinsics, so a classifier could be reading the lens signature rather than
the scene. That does not undermine the collapse question -- a collapsed
embedding could not separate on either signal -- but it does mean this is
not a substitute for the cross-factory control, where intrinsics vary within
each group too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR  # noqa: E402

SEED = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Workstation separability test.")
    parser.add_argument("--tag", type=str, default="factory051_v2")
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    embeddings = np.load(EMB_DIR / f"{args.tag}.npy").astype(np.float32)
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")

    labels = manifest["worker"].to_numpy()
    groups = manifest["clip"].to_numpy()
    classes = np.unique(labels)
    chance = 1.0 / len(classes)

    print(f"{len(embeddings)} frames | {len(classes)} workers | "
          f"{len(np.unique(groups))} clips")
    print(f"chance accuracy: {chance * 100:.1f}%\n")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=SEED)
    train_idx, test_idx = next(splitter.split(embeddings, labels, groups))
    print(f"train {len(train_idx)} frames / test {len(test_idx)} frames "
          f"(split by clip — no clip appears on both sides)")

    model = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    model.fit(embeddings[train_idx], labels[train_idx])
    pred = model.predict(embeddings[test_idx])
    acc = accuracy_score(labels[test_idx], pred)

    print(f"\nworkstation identification accuracy: {acc * 100:.1f}%")
    print(f"lift over chance: {acc / chance:.1f}x")

    cm = confusion_matrix(labels[test_idx], pred, labels=classes)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("\nper-worker recall:")
    for cls, rec, n in zip(classes, per_class, cm.sum(axis=1)):
        bar = "#" * int(round(rec * 40))
        print(f"  w{cls}  {rec * 100:5.1f}%  n={n:4d}  {bar}")

    print("\n--- interpretation ---")
    if acc > 0.6:
        print(
            f"The embedding separates {len(classes)} workstations inside a single\n"
            f"factory at {acc * 100:.0f}% ({acc / chance:.0f}x chance). It is not\n"
            "collapsing this footage into an undifferentiated region."
        )
    elif acc > 2 * chance:
        print("Above chance but weak. The embedding discriminates only coarsely here.")
    else:
        print(
            "AT OR NEAR CHANCE. The embedding cannot separate stations in one\n"
            "factory, which means the redundancy figure is measuring its blind\n"
            "spot. Do not publish; switch to DINOv2 or undistort the fisheye."
        )

    print(
        "\nCAVEAT: per-worker fisheye intrinsics differ, so some of this signal\n"
        "may be lens rather than scene. This answers the collapse question, but\n"
        "it does not replace the cross-factory control."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
