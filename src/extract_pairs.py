"""Render side-by-side frame pairs so the threshold can be set by eye.

The redundancy curve is steep between 0.88 and 0.94, so the headline number
moves a lot depending on where the line is drawn. Picking that line by taste
is exactly what makes a published figure attackable. Instead: pull real pairs
at several thresholds, look at them, and set the threshold at the point where
a human agrees the content is genuinely repeated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, OUT_DIR  # noqa: E402
from dedup import measure  # noqa: E402
from frames import sample_frames  # noqa: E402
from subset_source import SUBSET_REPO  # noqa: E402

PAIRS_DIR = OUT_DIR / "pairs"
BANDS = [(0.86, 0.88), (0.88, 0.90), (0.90, 0.92), (0.92, 0.94), (0.94, 1.01)]
THUMB_W = 480


def fetch_frame(clip: str, frame_pos: int) -> Image.Image | None:
    """Re-decode one clip and return the frame at the recorded position.

    Sampling is deterministic (keyframes, fixed interval), so frame_pos from
    the manifest indexes the same frame the embedding came from.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"datasets/{SUBSET_REPO}/data/{clip}.mp4"
    try:
        with fs.open(path, "rb") as handle:
            blob = handle.read()
    except Exception:  # noqa: BLE001
        return None

    frames = sample_frames(blob)
    if frame_pos >= len(frames):
        return None
    return frames.images[frame_pos]


def side_by_side(left: Image.Image, right: Image.Image, caption: str) -> Image.Image:
    def thumb(img: Image.Image) -> Image.Image:
        ratio = THUMB_W / img.width
        return img.resize((THUMB_W, int(img.height * ratio)), Image.LANCZOS)

    a, b = thumb(left), thumb(right)
    height = max(a.height, b.height)
    canvas = Image.new("RGB", (THUMB_W * 2 + 12, height + 26), "black")
    canvas.paste(a, (0, 26))
    canvas.paste(b, (THUMB_W + 12, 26))
    ImageDraw.Draw(canvas).text((6, 7), caption, fill="white")
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Render pairs for calibration.")
    parser.add_argument("--tag", type=str, default="factory051")
    parser.add_argument("--per-band", type=int, default=3)
    args = parser.parse_args()

    embeddings = np.load(EMB_DIR / f"{args.tag}.npy")
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")
    result = measure(embeddings)

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    written = 0

    for low, high in BANDS:
        in_band = np.flatnonzero((result.max_sim >= low) & (result.max_sim < high))
        if in_band.size == 0:
            print(f"band {low:.2f}-{high:.2f}: empty")
            continue

        picks = rng.choice(in_band, size=min(args.per_band, in_band.size), replace=False)
        print(f"band {low:.2f}-{high:.2f}: {in_band.size} frames, sampling {picks.size}")

        for i in picks:
            j = int(result.match_idx[i])
            sim = float(result.max_sim[i])
            row_a, row_b = manifest.iloc[i], manifest.iloc[j]

            img_a = fetch_frame(row_a["clip"], int(row_a["frame_pos"]))
            img_b = fetch_frame(row_b["clip"], int(row_b["frame_pos"]))
            if img_a is None or img_b is None:
                print(f"  skip {i}<->{j}: could not fetch")
                continue

            same_worker = row_a["worker"] == row_b["worker"]
            caption = (
                f"sim={sim:.3f}  "
                f"L: w{row_a['worker']} {row_a['clip'][-5:]} @{row_a['timestamp_sec']:.0f}s   "
                f"R: w{row_b['worker']} {row_b['clip'][-5:]} @{row_b['timestamp_sec']:.0f}s   "
                f"{'same worker' if same_worker else 'DIFFERENT WORKER'}"
            )
            out = PAIRS_DIR / f"sim{sim:.3f}_{i}_{j}.jpg"
            side_by_side(img_a, img_b, caption).save(out, quality=88)
            written += 1
            print(f"  wrote {out.name}")

    print(f"\n{written} pairs in {PAIRS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
