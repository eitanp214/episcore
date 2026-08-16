"""CLIP image embeddings, batched for a small GPU."""

from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image

from config import (
    BATCH_SIZE,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    DEVICE_PREFERENCE,
    EMBED_DIM,
)


def resolve_device() -> torch.device:
    if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Embedder:
    """Wraps open_clip so callers hand over PIL images and get unit vectors.

    Vectors are L2-normalised on the way out, so cosine similarity is a plain
    dot product everywhere downstream — including inside FAISS, which only
    offers inner product.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or resolve_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Embed images, returning (N, EMBED_DIM) float16."""
        if not images:
            return np.empty((0, EMBED_DIM), dtype=np.float16)

        out: list[np.ndarray] = []
        for start in range(0, len(images), BATCH_SIZE):
            chunk = images[start : start + BATCH_SIZE]
            batch = torch.stack([self.preprocess(img) for img in chunk])
            batch = batch.to(self.device, non_blocking=True)

            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype(np.float16))

        return np.concatenate(out, axis=0)
