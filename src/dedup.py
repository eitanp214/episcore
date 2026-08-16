"""Measure how much of a corpus is content that already appeared earlier.

Definition used throughout: a frame is *redundant at threshold t* if its
cosine similarity to any EARLIER frame in the corpus is at least t.

This is deliberately threshold-independent to compute. One pass records each
frame's best match against everything before it; the full redundancy curve
then falls out of that single array. The alternative -- greedy "keep or drop"
-- has to be re-run per threshold and its answer depends on the order in
which drops cascade, which is exactly the kind of hidden knob that makes a
published number indefensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np

from config import SIM_THRESHOLDS

BLOCK = 4096


@dataclass
class RedundancyResult:
    """Per-frame nearest-earlier-neighbour scores plus the derived curve."""

    max_sim: np.ndarray  # (N,) best similarity to any earlier frame
    match_idx: np.ndarray  # (N,) which earlier frame that was, -1 if none
    curve: dict[float, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.max_sim.shape[0])

    def redundant_mask(self, threshold: float) -> np.ndarray:
        return self.max_sim >= threshold

    def pairs_at(self, threshold: float, limit: int = 50) -> list[tuple[int, int, float]]:
        """Sample (frame, its earlier match, similarity) for human review.

        Spread across the ranked list rather than taking the top matches,
        so the eyeballed examples reflect the threshold boundary and not
        just the most obvious duplicates.
        """
        hits = np.flatnonzero(self.redundant_mask(threshold))
        if hits.size == 0:
            return []
        picks = hits[np.linspace(0, hits.size - 1, min(limit, hits.size)).astype(int)]
        return [(int(i), int(self.match_idx[i]), float(self.max_sim[i])) for i in picks]


def measure(embeddings: np.ndarray) -> RedundancyResult:
    """Score every frame against all frames that precede it.

    `embeddings` must be L2-normalised and ordered as collected, so that
    "earlier" means earlier in the corpus.
    """
    x = np.ascontiguousarray(embeddings.astype(np.float32))
    n, dim = x.shape

    max_sim = np.full(n, -1.0, dtype=np.float32)
    match_idx = np.full(n, -1, dtype=np.int64)

    index = faiss.IndexFlatIP(dim)

    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        block = x[start:stop]

        # Best match among frames already in the index (strictly earlier).
        if index.ntotal > 0:
            sims, idxs = index.search(block, 1)
            max_sim[start:stop] = sims[:, 0]
            match_idx[start:stop] = idxs[:, 0]

        # Best match among earlier frames inside this same block.
        if block.shape[0] > 1:
            within = block @ block.T
            within[np.triu_indices(block.shape[0], k=0)] = -1.0
            best = within.argmax(axis=1)
            best_sim = within[np.arange(block.shape[0]), best]

            improved = best_sim > max_sim[start:stop]
            max_sim[start:stop] = np.where(improved, best_sim, max_sim[start:stop])
            match_idx[start:stop] = np.where(
                improved, best + start, match_idx[start:stop]
            )

        index.add(block)

    result = RedundancyResult(max_sim=max_sim, match_idx=match_idx)
    result.curve = {
        t: float(np.mean(max_sim >= t)) for t in SIM_THRESHOLDS
    }
    return result


def measure_by_group(
    embeddings: np.ndarray, groups: np.ndarray
) -> dict[str, RedundancyResult]:
    """Run the measurement independently inside each group.

    Scoping matters: redundancy within one worker's own footage, across
    workers in a factory, and across the whole corpus are three different
    claims. Reporting only the global number hides which one is driving it.
    """
    results: dict[str, RedundancyResult] = {}
    for key in np.unique(groups):
        mask = groups == key
        if mask.sum() < 2:
            continue
        results[str(key)] = measure(embeddings[mask])
    return results


def summarise(results: dict[str, RedundancyResult]) -> dict[float, float]:
    """Frame-weighted mean curve across groups.

    Weighted by frame count so a worker with 30 frames cannot swing the
    headline as hard as one with 3,000.
    """
    if not results:
        return {}
    total = sum(r.n_frames for r in results.values())
    return {
        t: sum(r.curve[t] * r.n_frames for r in results.values()) / total
        for t in SIM_THRESHOLDS
    }
