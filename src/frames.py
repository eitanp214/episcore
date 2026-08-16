"""Decode sampled frames out of in-memory H.265 clips.

Only keyframes are decoded. H.265 keyframes land every 1-4 seconds, which
is already close to the sampling rate redundancy analysis needs, and asking
the decoder to skip non-key frames is roughly an order of magnitude cheaper
than decoding every frame and discarding 59 of every 60.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import av
import numpy as np
from PIL import Image

from config import FRAME_INTERVAL_SEC, MAX_FRAMES_PER_CLIP

av.logging.set_level(av.logging.ERROR)


# Laplacian kernel for the sharpness estimate.
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
_SHARPNESS_WIDTH = 320


@dataclass
class FrameSet:
    """Sampled frames from one clip, with timestamp and sharpness per frame."""

    images: list[Image.Image]
    timestamps: list[float]
    sharpness: list[float]

    def __len__(self) -> int:
        return len(self.images)


def laplacian_variance(image: Image.Image) -> float:
    """Sharpness estimate: variance of the Laplacian.

    Motion-blurred frames are a real confound here. Blur destroys detail, so
    two blurred frames collapse toward each other in embedding space and
    register as duplicates of one another even when they show different
    things. Measured inspection found a 0.906 pair that matched purely on
    shared blur, ranking above a 0.872 pair that was genuinely the same
    workstation.

    Reported raw. The cutoff is chosen as a percentile of the corpus during
    analysis rather than fixed here, since the value scales with resolution
    and scene content.
    """
    grey = image.convert("L")
    ratio = _SHARPNESS_WIDTH / grey.width
    small = np.asarray(
        grey.resize((_SHARPNESS_WIDTH, max(1, int(grey.height * ratio))), Image.BILINEAR),
        dtype=np.float32,
    )
    if small.shape[0] < 3 or small.shape[1] < 3:
        return 0.0

    windows = np.lib.stride_tricks.sliding_window_view(small, (3, 3))
    response = np.einsum("ijkl,kl->ij", windows, _LAPLACIAN)
    return float(response.var())


def sample_frames(
    video: bytes,
    interval_sec: float = FRAME_INTERVAL_SEC,
    max_frames: int = MAX_FRAMES_PER_CLIP,
) -> FrameSet:
    """Return keyframes spaced at least `interval_sec` apart.

    Returns an empty FrameSet on undecodable input rather than raising: a
    corrupt clip in a 192,900-clip corpus is a data point, not a crash.
    """
    images: list[Image.Image] = []
    timestamps: list[float] = []
    sharpness: list[float] = []

    try:
        with av.open(io.BytesIO(video)) as container:
            if not container.streams.video:
                return FrameSet([], [], [])

            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            stream.thread_type = "AUTO"

            last_kept = None
            for frame in container.decode(stream):
                ts = float(frame.time or 0.0)
                if last_kept is not None and ts - last_kept < interval_sec:
                    continue

                image = frame.to_image()
                images.append(image)
                timestamps.append(ts)
                sharpness.append(laplacian_variance(image))
                last_kept = ts

                if len(images) >= max_frames:
                    break
    except (av.AVError, ValueError, MemoryError):
        return FrameSet(images, timestamps, sharpness)

    return FrameSet(images, timestamps, sharpness)


def to_array(frames: FrameSet) -> np.ndarray:
    """Stack frames as uint8 HWC for downstream preprocessing."""
    if not frames.images:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack([np.asarray(img) for img in frames.images])
