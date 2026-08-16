"""EpiScore — score a local video corpus for redundancy and usability.

    python src/score.py /path/to/clips
    python src/score.py /path/to/clips --pattern "(?P<operator>op\\d+)" --json out.json

Produces a per-operator scorecard. Per-operator, not per-corpus, because
measured redundancy varies about 6x between operators on the same factory
floor -- a single corpus number averages away the thing a buyer would act on.

The report refuses to state a redundancy figure until it has checked that the
embedding can actually discriminate on the supplied footage. A number derived
from an embedding that sees everything as identical is not a small error, it
is a fabrication, so that check gates the output rather than annotating it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HEADLINE_THRESHOLD, SIM_THRESHOLDS  # noqa: E402
from dedup import measure  # noqa: E402
from embed import Embedder  # noqa: E402
from frames import sample_frames  # noqa: E402
from local_source import describe, discover, stream_unit  # noqa: E402

SEPARABILITY_FLOOR = 2.0  # times chance
BLUR_PERCENTILE = 20.0

# Share of RANDOM frame pairs allowed to clear the threshold before the
# similarity scale is judged saturated. On real footage this sits near zero
# (0.16% within an operator, 0.00% across operators). If unrelated frames
# routinely score as duplicates, "duplicate" has stopped meaning anything on
# this corpus and no redundancy figure derived from it is worth reporting.
SATURATION_CEILING = 0.05
RANDOM_PAIRS = 20_000


def saturation(embeddings: np.ndarray, threshold: float) -> dict:
    """How often do unrelated frames already clear the threshold?"""
    n = len(embeddings)
    if n < 4:
        return {"share": 0.0, "median": float("nan"), "pairs": 0}

    rng = np.random.default_rng(0)
    i = rng.integers(0, n, size=RANDOM_PAIRS)
    j = rng.integers(0, n, size=RANDOM_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]

    a = embeddings[i].astype(np.float32)
    b = embeddings[j].astype(np.float32)
    sims = np.einsum("ij,ij->i", a, b)
    return {
        "share": float(np.mean(sims >= threshold)),
        "median": float(np.median(sims)),
        "pairs": int(sims.size),
    }


def collect(units, embedder, max_clips: int | None) -> tuple[np.ndarray, pd.DataFrame]:
    vectors, rows = [], []
    for n, unit in enumerate(units, start=1):
        clips = 0
        for sample in stream_unit(unit):
            if max_clips and clips >= max_clips:
                break
            frames = sample_frames(sample.video)
            clips += 1
            if len(frames) == 0:
                continue
            vectors.append(embedder.embed(frames.images))
            for pos, (ts, sharp) in enumerate(zip(frames.timestamps, frames.sharpness)):
                rows.append({
                    "operator": unit.worker, "clip": sample.key,
                    "frame_pos": pos, "timestamp_sec": ts, "sharpness": sharp,
                    "path": sample.meta.get("path"),
                })
        print(f"  [{n}/{len(units)}] {unit.worker}: {clips} clips", flush=True)

    if not vectors:
        return np.empty((0, 512), dtype=np.float16), pd.DataFrame()
    return np.concatenate(vectors, axis=0), pd.DataFrame(rows)


def separability(embeddings: np.ndarray, manifest: pd.DataFrame) -> dict | None:
    """Can a linear model name the operator from a frame? Gates the report."""
    operators = manifest["operator"].unique()
    if len(operators) < 2:
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit

    labels = manifest["operator"].to_numpy()
    groups = manifest["clip"].to_numpy()
    if len(np.unique(groups)) < len(operators) * 2:
        return None

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    try:
        train, test = next(splitter.split(embeddings, labels, groups))
    except ValueError:
        return None
    if len(np.unique(labels[train])) < 2:
        return None

    model = LogisticRegression(max_iter=2000)
    model.fit(embeddings[train].astype(np.float32), labels[train])
    acc = accuracy_score(labels[test], model.predict(embeddings[test].astype(np.float32)))
    chance = 1.0 / len(operators)
    return {"accuracy": float(acc), "chance": chance, "lift": float(acc / chance)}


def redundancy(embeddings: np.ndarray, threshold: float) -> float:
    if len(embeddings) < 2:
        return 0.0
    return float(np.mean(measure(embeddings).max_sim >= threshold))


def bar(value: float, scale: float, width: int = 26) -> str:
    return "#" * max(0, int(round(value / max(scale, 1e-9) * width)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a video corpus for redundancy and usability.")
    parser.add_argument("path", type=Path, help="directory of video files")
    parser.add_argument("--pattern", default=None,
                        help=r"regex with a named group 'operator', e.g. '(?P<operator>op\d+)'")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="cap clips per operator (for a quick look)")
    parser.add_argument("--threshold", type=float, default=HEADLINE_THRESHOLD)
    parser.add_argument("--blur-percentile", type=float, default=BLUR_PERCENTILE)
    parser.add_argument("--json", type=Path, default=None, help="write report JSON here")
    args = parser.parse_args()

    started = time.time()
    try:
        units = discover(args.path, args.pattern, args.max_clips)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not units:
        print(f"error: no video files found under {args.path}", file=sys.stderr)
        return 1

    print(f"EPISCORE\ncorpus: {args.path}\n{describe(units)}\n")
    print("sampling and embedding ...", flush=True)
    embedder = Embedder()
    embeddings, manifest = collect(units, embedder, args.max_clips)

    if manifest.empty:
        print("error: no frames could be decoded", file=sys.stderr)
        return 1

    # --- gate ------------------------------------------------------------
    print("\nVALIDATION")

    sat = saturation(embeddings, args.threshold)
    print(f"  random-pair baseline:  median {sat['median']:.3f}, "
          f"{sat['share'] * 100:.1f}% already >= {args.threshold}", end="")
    if sat["share"] > SATURATION_CEILING:
        print("   FAIL")
        print(
            f"\n  The similarity scale is saturated on this footage: unrelated\n"
            f"  frames clear the duplicate threshold {sat['share'] * 100:.0f}% of the "
            f"time. Every clip\n"
            f"  would score as a duplicate of every other, so a redundancy figure\n"
            f"  here would describe the embedding, not the corpus. No score reported.\n"
            f"\n  Usual causes: footage the encoder was not trained on (synthetic\n"
            f"  patterns, heavy distortion, near-uniform scenes), or too few frames."
        )
        return 2
    print("   PASS")

    sep = separability(embeddings, manifest)
    if sep is None:
        print("  operator separability: not testable "
              "(needs >= 2 operators and enough clips)")
        print("  -> proceeding, but the figures below are UNVERIFIED on this footage")
        gate = "untested"
    elif sep["lift"] < SEPARABILITY_FLOOR:
        print(f"  operator separability: {sep['accuracy'] * 100:.1f}% "
              f"vs {sep['chance'] * 100:.1f}% chance ({sep['lift']:.1f}x)   FAIL")
        print("\n  The embedding cannot tell your operators apart, which means it is\n"
              "  not resolving this footage. Any redundancy figure would measure that\n"
              "  failure rather than your corpus. No score reported.")
        return 2
    else:
        print(f"  operator separability: {sep['accuracy'] * 100:.1f}% "
              f"vs {sep['chance'] * 100:.1f}% chance ({sep['lift']:.1f}x)   PASS")
        print("  -> the embedding resolves this footage; figures below are meaningful")
        gate = "pass"

    # --- usability -------------------------------------------------------
    sharp = manifest["sharpness"].to_numpy()
    cutoff = float(np.percentile(sharp, args.blur_percentile))
    keep = sharp >= cutoff
    print(f"\nUSABILITY\n  frames sampled       {len(manifest):>7,}")
    print(f"  dropped as blurred   {int((~keep).sum()):>7,}  "
          f"({(~keep).mean() * 100:.1f}%)")

    emb_k = embeddings[keep]
    man_k = manifest.loc[keep].reset_index(drop=True)

    # --- redundancy ------------------------------------------------------
    corpus = redundancy(emb_k, args.threshold)
    print(f"\nREDUNDANCY at similarity >= {args.threshold}")
    print(f"  corpus               {corpus * 100:.1f}%")

    per_op = {}
    for op, group in man_k.groupby("operator", sort=True):
        per_op[str(op)] = redundancy(emb_k[group.index.to_numpy()], args.threshold)

    if len(per_op) > 1:
        vals = np.array(list(per_op.values()))
        top = vals.max()
        print("\n  BY OPERATOR")
        for op, val in sorted(per_op.items(), key=lambda kv: -kv[1]):
            flag = ""
            if val == vals.max():
                flag = "  <- highest"
            elif val == vals.min():
                flag = "  <- lowest"
            print(f"    {op:<20} {val * 100:5.1f}%  {bar(val, top)}{flag}")
        ratio = vals.max() / max(vals.min(), 1e-9)
        print(f"\n  spread {vals.min() * 100:.1f}% - {vals.max() * 100:.1f}%  "
              f"({ratio:.1f}x)")
        if ratio >= 2:
            print("  -> priced per hour, these operators are not the same purchase")

    curve = {f"{t:.2f}": redundancy(emb_k, t) for t in SIM_THRESHOLDS}

    report = {
        "corpus": str(args.path),
        "operators": len(units),
        "clips": int(man_k["clip"].nunique()),
        "frames_sampled": int(len(manifest)),
        "frames_scored": int(len(man_k)),
        "blur_dropped_share": float((~keep).mean()),
        "threshold": args.threshold,
        "validation": {"gate": gate, **(sep or {})},
        "redundancy_corpus": corpus,
        "redundancy_by_operator": per_op,
        "redundancy_curve": curve,
        "elapsed_sec": round(time.time() - started, 1),
    }

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    print(f"\ndone in {report['elapsed_sec']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
