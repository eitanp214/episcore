"""How much does the cross-factory separation depend on WHICH factories were sampled?

The separation fell from 0.213 at four factories to 0.155 at six. Two readings
fit that: a sampling effect settling toward its true value, or a lucky first
draw drifting toward the 0.10 pass threshold. Publishing without telling them
apart would mean publishing a number known to be moving.

Downloading more factories is the direct answer but the infrastructure keeps
killing long runs. This is the cheap one: leave out one factory at a time and
watch the statistic move. Wide swings mean the estimate is hostage to which
factories happened to be drawn; tight ones mean six is already enough.

Also reports every leave-one-out value against the pre-registered 0.10 floor,
since what matters is not the point estimate but whether any plausible sample
would have failed the gate.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD  # noqa: E402

SEED = 0
N_PAIRS = 20_000
FLOOR = 0.10


def separation(embeddings: np.ndarray, factory: np.ndarray, rng) -> tuple[float, float]:
    """(separation, share of cross-factory pairs above threshold)."""
    idx = np.arange(len(factory))
    same, cross = [], []

    for f in np.unique(factory):
        pool = idx[factory == f]
        if pool.size >= 2:
            i = rng.choice(pool, size=N_PAIRS // max(len(np.unique(factory)), 1))
            j = rng.choice(pool, size=i.size)
            keep = i != j
            if keep.any():
                a = embeddings[i[keep]].astype(np.float32)
                b = embeddings[j[keep]].astype(np.float32)
                same.append(np.einsum("ij,ij->i", a, b))

    factories = np.unique(factory)
    for f1, f2 in itertools.combinations(factories, 2):
        p1, p2 = idx[factory == f1], idx[factory == f2]
        if p1.size and p2.size:
            n = max(N_PAIRS // max(len(list(itertools.combinations(factories, 2))), 1), 1)
            a = embeddings[rng.choice(p1, size=n)].astype(np.float32)
            b = embeddings[rng.choice(p2, size=n)].astype(np.float32)
            cross.append(np.einsum("ij,ij->i", a, b))

    if not same or not cross:
        return float("nan"), float("nan")

    same_v = np.concatenate(same)
    cross_v = np.concatenate(cross)
    return float(same_v.mean() - cross_v.mean()), float(np.mean(cross_v >= HEADLINE_THRESHOLD))


def main() -> int:
    parser = argparse.ArgumentParser(description="Factory-level stability of separation.")
    parser.add_argument("--tag", type=str, default="crossfactory_all")
    args = parser.parse_args()

    embeddings = np.load(EMB_DIR / f"{args.tag}.npy")
    manifest = pd.read_parquet(EMB_DIR / f"{args.tag}.parquet")
    factory = manifest["factory"].to_numpy()
    factories = np.unique(factory)

    print(f"{len(embeddings)} frames across {len(factories)} factories: "
          f"{', '.join(factories)}\n")

    rng = np.random.default_rng(SEED)
    full_sep, full_share = separation(embeddings, factory, rng)
    print(f"all {len(factories)} factories:  separation {full_sep:+.3f}   "
          f"cross-factory >= {HEADLINE_THRESHOLD}: {full_share * 100:.2f}%\n")

    print("LEAVE ONE OUT")
    seps = []
    for f in factories:
        mask = factory != f
        rng = np.random.default_rng(SEED)
        sep, share = separation(embeddings[mask], factory[mask], rng)
        seps.append(sep)
        flag = "  <-- would FAIL" if sep < FLOOR else ""
        print(f"  without f{f}:  {sep:+.3f}   cross >= thr {share * 100:.2f}%{flag}")

    seps = np.array([s for s in seps if not np.isnan(s)])
    print(f"\n  range {seps.min():+.3f} to {seps.max():+.3f}   "
          f"spread {seps.max() - seps.min():.3f}   sd {seps.std(ddof=1):.3f}")

    n = len(seps)
    jack_se = float(np.sqrt((n - 1) / n * np.sum((seps - seps.mean()) ** 2)))
    lo = full_sep - 1.96 * jack_se
    print(f"  jackknife SE {jack_se:.3f}   95% lower bound {lo:+.3f}")

    print("\n--- verdict ---")
    if seps.min() >= FLOOR and lo >= FLOOR:
        print(
            f"STABLE. Every leave-one-out sample clears the {FLOOR} floor, and so\n"
            f"does the lower bound of the interval. The gate is not resting on\n"
            f"which factories happened to be drawn."
        )
    elif seps.min() >= FLOOR:
        print(
            f"BORDERLINE. All leave-one-out samples clear {FLOOR}, but the interval's\n"
            f"lower bound ({lo:+.3f}) does not. More factories are needed before the\n"
            f"claim is safe to publish."
        )
    else:
        print(
            f"UNSTABLE. At least one leave-one-out sample falls below {FLOOR}. The\n"
            f"pass depends on which factories were sampled. Do not publish; extend\n"
            f"the sample or switch encoder."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
