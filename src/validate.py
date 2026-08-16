"""Does the embedding actually discriminate on THIS footage?

The redundancy number is only meaningful if CLIP can tell these scenes
apart in the first place. Two ways that could fail here:

  1. CLIP was trained on web photography, not fisheye industrial video.
     Factory interiors may collapse into a narrow region of the embedding
     space, making everything look alike and inflating redundancy.
  2. Each worker has their own fisheye intrinsics, so identical scenes
     shot by two workers are geometrically different images. That confounds
     any cross-worker comparison.

The check: the corpus spans 85 physically distinct factories. Frames from
different factories MUST be less similar to each other than frames from the
same worker. If those distributions overlap, the metric is measuring the
embedding's blind spot rather than real redundancy, and the run is void.

Run this before quoting any redundancy figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD  # noqa: E402

RNG_SEED = 0
N_PAIRS = 20_000


def _sample_pairs(
    rng: np.random.Generator,
    embeddings: np.ndarray,
    left_pool: np.ndarray,
    right_pool: np.ndarray,
    exclude_identical: bool,
) -> np.ndarray:
    """Cosine similarity for random cross-pairs drawn from two index pools."""
    if left_pool.size == 0 or right_pool.size == 0:
        return np.empty(0, dtype=np.float32)

    i = rng.choice(left_pool, size=N_PAIRS)
    j = rng.choice(right_pool, size=N_PAIRS)
    if exclude_identical:
        keep = i != j
        i, j = i[keep], j[keep]
    if i.size == 0:
        return np.empty(0, dtype=np.float32)

    a = embeddings[i].astype(np.float32)
    b = embeddings[j].astype(np.float32)
    return np.einsum("ij,ij->i", a, b)


def describe(name: str, sims: np.ndarray) -> dict:
    if sims.size == 0:
        return {"scope": name, "n": 0}
    return {
        "scope": name,
        "n": int(sims.size),
        "mean": float(sims.mean()),
        "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)),
        "p99": float(np.percentile(sims, 99)),
        f"frac>={HEADLINE_THRESHOLD}": float(np.mean(sims >= HEADLINE_THRESHOLD)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate embedding separability.")
    parser.add_argument("--tag", type=str, default="run1")
    args = parser.parse_args()

    emb_path = EMB_DIR / f"{args.tag}.npy"
    if not emb_path.exists():
        print(f"missing {emb_path} — run run_embed.py first", file=sys.stderr)
        return 1

    embeddings = np.load(emb_path)
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")
    rng = np.random.default_rng(RNG_SEED)

    factory = manifest["factory"].to_numpy()
    worker = (manifest["factory"] + "/" + manifest["worker"]).to_numpy()
    idx = np.arange(len(manifest))

    factories = np.unique(factory)
    single_factory = factories.size < 2
    if single_factory:
        print(
            "NOTE: only one factory present. Falling back to cross-WORKER as the\n"
            "      control group. This is a weaker test: two workers at adjacent\n"
            "      stations genuinely should look alike, so a high cross-worker\n"
            "      score is ambiguous rather than disqualifying. Verdict below is\n"
            "      PROVISIONAL and is not sufficient to publish a figure.\n"
        )

    # Same worker.
    same_worker_sims = []
    for w in np.unique(worker):
        pool = idx[worker == w]
        if pool.size >= 2:
            same_worker_sims.append(
                _sample_pairs(rng, embeddings, pool, pool, exclude_identical=True)
            )
    same_worker = (
        np.concatenate(same_worker_sims) if same_worker_sims else np.empty(0)
    )

    # Same factory, different worker.
    cross_worker_sims = []
    for f in factories:
        workers_here = np.unique(worker[factory == f])
        if workers_here.size < 2:
            continue
        a_pool = idx[worker == workers_here[0]]
        b_pool = idx[worker == workers_here[1]]
        cross_worker_sims.append(
            _sample_pairs(rng, embeddings, a_pool, b_pool, exclude_identical=False)
        )
    cross_worker = (
        np.concatenate(cross_worker_sims) if cross_worker_sims else np.empty(0)
    )

    # Control group: different factory when available, else different worker.
    if single_factory:
        control = cross_worker
        control_name = "diff_worker (fallback control)"
    else:
        first_half = idx[factory == factories[0]]
        rest = idx[factory != factories[0]]
        control = _sample_pairs(
            rng, embeddings, first_half, rest, exclude_identical=False
        )
        control_name = "diff_factory"

    rows = [
        describe("same_worker", same_worker),
        describe("same_factory_diff_worker", cross_worker),
        describe(control_name, control),
    ]
    table = pd.DataFrame(rows)
    print("\nSIMILARITY DISTRIBUTION BY SCOPE\n")
    print(table.to_string(index=False))

    verdict_ok = True
    reasons: list[str] = []

    if same_worker.size and control.size:
        gap = float(np.mean(same_worker)) - float(np.mean(control))
        print(f"\nseparation (same-worker mean - control mean): {gap:+.3f}")
        if gap < 0.10:
            verdict_ok = False
            reasons.append(
                f"separation {gap:.3f} is too small — the embedding barely "
                f"distinguishes the control group from same-worker footage"
            )

    if control.size:
        false_dupes = float(np.mean(control >= HEADLINE_THRESHOLD))
        print(
            f"control pairs scoring >= {HEADLINE_THRESHOLD}: "
            f"{false_dupes * 100:.2f}%"
            + ("" if single_factory else "  (false duplicates by construction)")
        )
        if false_dupes > 0.01 and not single_factory:
            verdict_ok = False
            reasons.append(
                f"{false_dupes * 100:.2f}% of physically-different-factory pairs "
                f"exceed the threshold; the redundancy figure would be inflated"
            )

    if single_factory:
        label = "PROVISIONAL — needs cross-factory control before publishing"
    else:
        label = "PASS — metric is usable"
    print("\nVERDICT:", label if verdict_ok else "FAIL")
    for reason in reasons:
        print(f"  - {reason}")
    if not verdict_ok:
        print(
            "\n  Do not publish a redundancy number from this run. Options:\n"
            "    - undistort fisheye using the per-worker intrinsics.json\n"
            "    - swap CLIP for DINOv2, which is stronger on scene geometry\n"
            "    - raise HEADLINE_THRESHOLD until diff-factory false dupes < 1%"
        )
    return 0 if verdict_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
