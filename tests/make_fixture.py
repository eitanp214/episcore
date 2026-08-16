"""Generate a synthetic corpus with known redundancy, for end-to-end testing.

Unit tests pin the maths. This pins the product: if the tool cannot tell a
deliberately repetitive operator from a deliberately varied one on footage
whose answer is known by construction, it is not going to be trusted on
footage where the answer is not.

Three operators, decreasing repetition:
  repetitive  — cycles a small set of scenes, so most clips repeat earlier ones
  mixed       — half repeats, half new
  diverse     — a new scene every clip

Expected ordering of measured redundancy: repetitive > mixed > diverse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import av
import numpy as np

W, H, FPS, SECONDS = 320, 240, 10, 3
GOP = 10  # a keyframe per second, so keyframe-only sampling sees several


def scene(seed: int) -> np.ndarray:
    """A visually distinct still: dominant hue, blocks, and fixed grain."""
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 220, size=3)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = base

    for _ in range(6):
        x0, y0 = rng.integers(0, W - 60), rng.integers(0, H - 60)
        w, h = rng.integers(40, 110), rng.integers(40, 90)
        colour = rng.integers(0, 255, size=3)
        img[y0:y0 + h, x0:x0 + w] = colour

    grain = rng.integers(-18, 18, size=(H, W, 3))
    return np.clip(img.astype(int) + grain, 0, 255).astype(np.uint8)


def write_clip(path: Path, still: np.ndarray, jitter_seed: int) -> None:
    """Encode a clip that drifts slightly, as real footage of one scene does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(jitter_seed)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.gop_size = GOP
        stream.options = {"crf": "28", "preset": "veryfast"}

        for i in range(FPS * SECONDS):
            shift = int(round(6 * np.sin(i / 5.0))) + int(rng.integers(-2, 3))
            frame_arr = np.roll(still, shift, axis=1)
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def build(root: Path, clips_per_operator: int = 8) -> None:
    root.mkdir(parents=True, exist_ok=True)

    # repetitive: three scenes on a loop
    pool = [scene(s) for s in (101, 102, 103)]
    for i in range(clips_per_operator):
        write_clip(root / "repetitive" / f"clip{i:02d}.mp4", pool[i % 3], 900 + i)

    # mixed: alternates between a fixed scene and a fresh one
    anchor = scene(201)
    for i in range(clips_per_operator):
        still = anchor if i % 2 == 0 else scene(210 + i)
        write_clip(root / "mixed" / f"clip{i:02d}.mp4", still, 950 + i)

    # diverse: a new scene each time
    for i in range(clips_per_operator):
        write_clip(root / "diverse" / f"clip{i:02d}.mp4", scene(300 + i), 980 + i)

    total = sum(1 for _ in root.rglob("*.mp4"))
    print(f"wrote {total} clips under {root}")
    print("expected redundancy ordering: repetitive > mixed > diverse")


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/episcore/data/fixture")
    build(dest)
